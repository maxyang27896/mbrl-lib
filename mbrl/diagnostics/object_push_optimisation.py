from __future__ import print_function
import argparse
import copy 
import cv2
import multiprocessing as mp
import pathlib
import pickle
import time
from typing import Sequence, Tuple, cast
import os 

import gym.wrappers
import numpy as np
import omegaconf
import skvideo.io
import torch

import mbrl.planning;
import mbrl.util
from mbrl.util.env import EnvHandler
from mbrl.util.pybullet import FreezePybullet, PybulletEnvHandler

import tactile_gym.rl_envs
from tactile_gym.sb3_helpers.params import import_parameters

# produce a display to render image
from pyvirtualdisplay import Display
_display = Display(visible=False, size=(1400, 900))
_ = _display.start()

from contextlib import contextmanager
import sys, os


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

env__: gym.Env
handler__: EnvHandler

def init(env_name: str, seed: int):
    global env__
    global handler__
    with suppress_stdout():
        handler__ = mbrl.util.create_handler_from_str(env_name)
        env__ = handler__.make_env_from_str(env_name)
        env__.seed(seed)
        env__.reset()


def evaluate_all_action_sequences(
    action_sequences: Sequence[Sequence[np.ndarray]],
    pool: mp.Pool,  # type: ignore
    current_state: Tuple,
) -> torch.Tensor:

    res_objs = [
        pool.apply_async(evaluate_sequence_fn, (sequence, current_state))  # type: ignore
        for sequence in action_sequences
    ]
    res = [res_obj.get() for res_obj in res_objs]
    return torch.tensor(res, dtype=torch.float32)


def evaluate_sequence_fn(action_sequence: np.ndarray, current_state: Tuple) -> float:
    global env__
    global handler__
    # obs0__ is not used (only here for compatibility with rollout_env)
    obs0 = env__.observation_space.sample()
    env = {"env": env__}
    handler__.set_env_state(current_state, env)
    _, rewards_, _ = handler__.rollout_env(
        env, obs0, -1, agent=None, plan=action_sequence
    )
    return rewards_.sum().item()


def get_random_trajectory(horizon):
    global env__
    return [env__.action_space.sample() for _ in range(horizon)]

if __name__ == "__main__":
    control_filename = 'saved_control'
    work_dir = os.path.join(os.getcwd(), control_filename)

    algo_name = 'ppo'
    env_name = 'object_push-v0'
    rl_params, algo_params, augmentations = import_parameters(env_name, algo_name)
    rl_params["env_modes"][ 'observation_mode'] = 'tactile_pose_goal_excluded_data'
    rl_params["env_modes"][ 'control_mode'] = 'TCP_position_control'
    rl_params["env_modes"][ 'terminate_early']  = True
    rl_params["env_modes"][ 'use_contact'] = True
    rl_params["env_modes"][ 'traj_type'] = 'point'
    rl_params["env_modes"][ 'task'] = "goal_pos"
    rl_params["env_modes"]['planar_states'] = True

    env_kwargs={
        'show_gui':False,
        'show_tactile':False,
        'max_steps':rl_params["max_ep_len"],
        'image_size':rl_params["image_size"],
        'env_modes':rl_params["env_modes"],
    }

    mp.set_start_method("spawn")
    handler_env_name = "pybulletgym___" + env_name
    handler = mbrl.util.create_handler_from_str(handler_env_name)
    eval_env = handler.make_env_from_str(handler_env_name, **env_kwargs)
    seed = 0
    eval_env.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    current_obs = eval_env.reset()
    render = True

    num_processes = 1
    samples_per_process = 128
    optimizer_type = "cem"
    num_steps = 1000

    if optimizer_type == "cem":
        optimizer_cfg = omegaconf.OmegaConf.create(
            {
                "_target_": "mbrl.planning.CEMOptimizer",
                "device": "cpu",
                "num_iterations": 5,
                "elite_ratio": 0.1,
                "population_size": num_processes * samples_per_process,
                "alpha": 0.1,
                "lower_bound": "???",
                "upper_bound": "???",
            }
        )
    elif optimizer_type == "mppi":
        optimizer_cfg = omegaconf.OmegaConf.create(
            {
                "_target_": "mbrl.planning.MPPIOptimizer",
                "num_iterations": 5,
                "gamma": 1.0,
                "population_size": num_processes * samples_per_process,
                "sigma": 0.95,
                "beta": 0.1,
                "lower_bound": "???",
                "upper_bound": "???",
                "device": "cpu",
            }
        )
    elif optimizer_type == "icem":
        optimizer_cfg = omegaconf.OmegaConf.create(
            {
                "_target_": "mbrl.planning.ICEMOptimizer",
                "num_iterations": 2,
                "elite_ratio": 0.1,
                "population_size": num_processes * samples_per_process,
                "population_decay_factor": 1.25,
                "colored_noise_exponent": 2.0,
                "keep_elite_frac": 0.1,
                "alpha": 0.1,
                "lower_bound": "???",
                "upper_bound": "???",
                "return_mean_elites": "true",
                "device": "cpu",
            }
        )
    else:
        raise ValueError

    controller = mbrl.planning.TrajectoryOptimizer(
            optimizer_cfg,
            eval_env.action_space.low,
            eval_env.action_space.high,
            40,
        )

    # Record video
    if render:
        record_every_n_frames = 3
        render_img = eval_env.render(mode="rgb_array")
        render_img_size = (render_img.shape[1], render_img.shape[0])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            os.path.join(work_dir, "optimal_policy.mp4"),
            fourcc,
            24.0,
            render_img_size,
        )
    
    num_processes = 1
    with mp.Pool(
        processes=num_processes, initializer=init, initargs=[handler_env_name, seed]
    ) as pool__:

        # initialise paramters
        total_reward__ = 0
        frames = []
        max_population_size = optimizer_cfg.population_size
        if isinstance(controller.optimizer, mbrl.planning.ICEMOptimizer):
            max_population_size += controller.optimizer.keep_elite_size
        value_history = np.zeros(
            (num_steps, max_population_size, optimizer_cfg.num_iterations)
        )
        values_sizes = []  # for icem
        eval_env_dict = {"env": eval_env}
        # state_before = eval_env_dict["env"].get_tactile_pose_obs_goal_exluded()

        # Start control loop
        for t in range(3):
            if render:
                frames.append(eval_env.render(mode="rgb_array"))
            start = time.time()
            current_state__ = handler.get_current_state(
                        eval_env_dict
                    )

            def trajectory_eval_fn(action_sequences):
                return evaluate_all_action_sequences(
                    action_sequences,
                    pool__,
                    current_state__,
                )
            
            best_value = [-np.inf]  # this is hacky, sorry

            def compute_population_stats(_population, values, opt_step):
                value_history[t, : len(values), opt_step] = values.numpy()
                values_sizes.append(len(values))
                best_value[0] = max(best_value[0], values.max().item())

            plan = controller.optimize(
                trajectory_eval_fn, callback=compute_population_stats
            )

            # state_after = eval_env.get_tactile_pose_obs_goal_exluded()

            action__ = plan[0]
            next_obs__, reward__, done__, _ = eval_env.step(action__)

            total_reward__ += reward__

            # state_after_action_dict = eval_env_dict["env"].get_tactile_pose_obs_goal_exluded()
            # state_after_action_env = eval_env.get_tactile_pose_obs_goal_exluded()


            # Record video at every n trials
            if render and t % record_every_n_frames == 0:
                render_img = eval_env.render(mode="rgb_array")
                render_img = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
                out.write(render_img)

            print(
                f"step: {t}, time: {time.time() - start: .3f}, "
                f"reward: {reward__: .3f}, pred_value: {best_value[0]: .3f}, "
                f"total_reward: {total_reward__: .3f}"
            )

        # output_dir = pathlib.Path(work_dir)
        # output_dir = output_dir / handler_env_name / optimizer_type
        # pathlib.Path.mkdir(output_dir, exist_ok=True, parents=True)

         # release video at every n trials
        if render:
            out.release()


        ########## Test algorithm ##########
        # eval_env_dict = {"env": eval_env}
        # current_state = handler.get_current_state(
        #         eval_env_dict
        #     )
        # state_before = eval_env_dict["env"].get_tactile_pose_obs_goal_exluded()

        # action_sequences = torch.from_numpy(np.array([eval_env.action_space.sample() for _ in range(15)])).float()
        # action_sequences = torch.tile(action_sequences, (5,1,1))

        # rewards_ = evaluate_all_action_sequences(
        #     action_sequences,
        #     pool__,
        #     current_state
        # )
        # state_after = eval_env_dict["env"].get_tactile_pose_obs_goal_exluded()

        # print(rewards_)
        # print(state_before)
        # print(state_after)
        # print(state_after_action_env)
        # print(state_after_action_dict)
        ########## Test algorithm ##########