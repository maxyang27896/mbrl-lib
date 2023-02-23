from IPython import display
import argparse
import cv2
import copy
from collections import deque
import gym
import hydra.utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys, shutil
import torch
import omegaconf
import time
import torch
import torch.nn as nn
from typing import Optional, Sequence, cast

import mbrl
import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util
import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac_pranz24
import mbrl.util.math as math_util

from mbrl.planning.sac_wrapper import SACAgent
from mbrl.util.plot_and_save_push_data import plot_and_save_training, plot_and_save_push_plots, clear_and_create_dir
from mbrl.util.eval_agent import eval_and_save_vid

import tactile_gym.rl_envs
from tactile_gym.sb3_helpers.params import import_parameters
from stable_baselines3.common.torch_layers import NatureCNN
from tactile_gym.sb3_helpers.custom.custom_torch_layers import CustomCombinedExtractor, ImpalaCNN

from pyvirtualdisplay import Display
_display = Display(visible=False, size=(1400, 900))
_ = _display.start()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

data_columns = [
    'trial',
    'trial_steps', 
    'time_steps', 
    'tcp_x',
    'tcp_y',
    'tcp_z',
    'contact_x', 
    'contact_y', 
    'contact_z', 
    'tcp_Rz', 
    'contact_Rz', 
    'goal_x', 
    'goal_y', 
    'goal_Rz', 
    'rewards', 
    'contact', 
    'dones',
]

def plot_and_save(y_data, x_data=None, title=None, xlabel=None, ylabel=None, work_dir=None):

    fig, ax = plt.subplots(3, 2, figsize=(14, 10))

    for i in range(3):
        for j in range(2):
            
            if (2 * i + j) == 5:
                break
            if not x_data:
                ax[i, j].plot(y_data[2 * i + j])
            else:   
                ax[i, j].plot(x_data[2 * i + j], y_data[2 * i + j])
            
            if title:
                ax[i, j].set_title(title[2 * i + j])
            if xlabel:
                ax[i, j].set_xlabel(xlabel[2 * i + j])
            if ylabel:
                ax[i, j].set_ylabel(ylabel[2 * i + j])

    if not work_dir:
        work_dir = os.getcwd()

    fig.savefig(os.path.join(work_dir, "losses.png"))
    plt.close(fig)

def rollout_model_and_populate_sac_buffer(
    model_env: models.ModelEnvPushing,
    replay_buffer: mbrl.util.ReplayBuffer,
    agent: SACAgent,
    sac_buffer: mbrl.util.ReplayBuffer,
    sac_samples_action: bool,
    rollout_horizon: int,
    batch_size: int,
):

    batch = replay_buffer.sample(batch_size)
    initial_obs, initial_act, *_ = cast(mbrl.types.TransitionBatch, batch).astuple()
    model_state = model_env.reset(
        initial_obs_batch=cast(np.ndarray, initial_obs),
        return_as_np=True,
    )
    accum_dones = np.zeros(initial_obs.shape[0], dtype=bool)
    obs = initial_obs
    stacked_act = initial_act
    model_env.reset_batch_goals(batch_size)
    for i in range(rollout_horizon):
        action = agent.act(obs, action=None, sample=sac_samples_action, batched=True)
        stacked_act = np.concatenate([stacked_act, action], axis=action.ndim - 1)
        stacked_act = stacked_act[:, -initial_act.shape[-1]:]
        pred_next_obs, pred_rewards, pred_dones, model_state = model_env.step(
            stacked_act, model_state, sample=True
        )
        pred_next_obs = np.concatenate([obs, pred_next_obs], axis=pred_next_obs.ndim - 1)
        pred_next_obs = pred_next_obs[:, -initial_obs.shape[-1]:]
        sac_buffer.add_batch(
            obs[~accum_dones],
            action[~accum_dones],
            pred_next_obs[~accum_dones],
            pred_rewards[~accum_dones, 0],
            pred_dones[~accum_dones, 0],
        )
        obs = pred_next_obs
        accum_dones |= pred_dones.squeeze()


# def rollout_model_and_populate_sac_buffer(
#     model_env: models.ModelEnvPushing,
#     replay_buffer: mbrl.util.ReplayBuffer,
#     agent: SACAgent,
#     sac_buffer: mbrl.util.ReplayBuffer,
#     sac_samples_action: bool,
#     rollout_horizon: int,
#     batch_size: int,
# ):

#     batch = replay_buffer.sample(batch_size)
#     initial_obs, initial_act, *_ = cast(mbrl.types.TransitionBatch, batch).astuple()
#     model_state = model_env.reset(
#         initial_obs_batch=cast(np.ndarray, initial_obs),
#         return_as_np=True,
#     )
#     accum_dones = np.zeros(initial_obs.shape[0], dtype=bool)
#     obs = initial_obs
#     stacked_act = initial_act
#     model_env.reset_batch_goals(batch_size, sample_goals=True)
#     model_env.update_step_data(obs)
#     for i in range(rollout_horizon):
#         agent_obs = model_env.get_agent_obs(obs)
#         action = agent.act(agent_obs, action=None, sample=sac_samples_action, batched=True)
#         stacked_act = np.concatenate([stacked_act, action], axis=action.ndim - 1)
#         stacked_act = stacked_act[:, -initial_act.shape[-1]:]
#         pred_next_obs, pred_rewards, pred_dones, model_state = model_env.step(
#             stacked_act, model_state, sample=True
#         )
#         pred_next_obs = np.concatenate([obs, pred_next_obs], axis=pred_next_obs.ndim - 1)
#         pred_next_obs = pred_next_obs[:, -initial_obs.shape[-1]:]
#         agent_next_obs = model_env.get_agent_obs(pred_next_obs)
#         sac_buffer.add_batch(
#             agent_obs[~accum_dones],
#             action[~accum_dones],
#             agent_next_obs[~accum_dones],
#             pred_rewards[~accum_dones, 0],
#             pred_dones[~accum_dones, 0],
#         )
#         obs = pred_next_obs
#         accum_dones |= pred_dones.squeeze()


def maybe_replace_sac_buffer(
    sac_buffer: Optional[mbrl.util.ReplayBuffer],
    obs_shape: Sequence[int],
    act_shape: Sequence[int],
    new_capacity: int,
    seed: int,
) -> mbrl.util.ReplayBuffer:
    if sac_buffer is None or new_capacity != sac_buffer.capacity:
        if sac_buffer is None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = sac_buffer.rng
        new_buffer = mbrl.util.ReplayBuffer(new_capacity, obs_shape, act_shape, rng=rng)
        if sac_buffer is None:
            return new_buffer
        obs, action, next_obs, reward, done, *_ = sac_buffer.get_all().astuple()
        new_buffer.add_batch(obs, action, next_obs, reward, done)
        return new_buffer
    return sac_buffer

def replay_to_sac_buffer(replay_buffer, sac_obs_shape, sac_act_shape, rng):
    new_sac_buffer = mbrl.util.ReplayBuffer(replay_buffer.num_stored, sac_obs_shape, sac_act_shape, rng=rng)
    obs, action, next_obs, reward, done, *_ = replay_buffer.get_all().astuple()

    # Get next_obs and action states
    next_obs = np.concatenate([obs[:, next_obs.shape[-1]:], next_obs], axis=1)
    action = action[:, -sac_act_shape[-1]:]

    # Fill new buffer
    new_sac_buffer.add_batch(obs, action, next_obs, reward, done)

    return new_sac_buffer

def train_agent(
    env_kwargs: dict,
    rl_params: dict,
    cfg_dict: dict,
    agent_cfg_dict: dict,
    work_dir: str,
):
    # Create work directory
    clear_and_create_dir(work_dir)
    # training_result_directory = os.path.join(work_dir, "training_result")

    # training environment
    env = gym.make(env_name, **env_kwargs)
    seed = 0
    env.seed(seed)
    rng = np.random.default_rng(seed=0)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    history_len = env.states_stacked_len
    step_obs_shape = (obs_shape[-1], )
    step_act_shape = act_shape
    stacked_obs_shape = (obs_shape[-1] * history_len, )
    stacked_act_shape = (history_len * act_shape[-1], )

    # evaluation environment
    num_eval_episodes = 7
    eval_env_kwargs = copy.deepcopy(env_kwargs)
    eval_env_kwargs["env_modes"]['eval_mode'] = True
    eval_env_kwargs["env_modes"]['eval_num'] = num_eval_episodes
    eval_env = gym.make(env_name, **eval_env_kwargs)

    agent_cfg = omegaconf.OmegaConf.create(agent_cfg_dict)
    planning.complete_agent_cfg(env, agent_cfg)
    agent = SACAgent(
        cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(agent_cfg))
    )

    # create the dynamics model
    cfg = omegaconf.OmegaConf.create(cfg_dict)
    dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    model_env = models.ModelEnvPushing(env, dynamics_model, termination_fn=None, reward_fn=None, generator=generator)

    # -------------- Create initial overrides. dataset --------------
    replay_buffer = common_util.create_replay_buffer(
        cfg,
        stacked_obs_shape,
        stacked_act_shape,
        rng=rng,
        next_obs_shape=(obs_shape[-1], )
    )
    
    # common_util.rollout_agent_trajectories(
    #     env,
    #     initial_buffer_size,
    #     mbrl.planning.RandomAgent(env),
    #     {},
    #     replay_buffer=replay_buffer,
    # )
    common_util.rollout_agent_trajectories(
        env,
        initial_buffer_size,
        agent,
        {"sample": True, "batched": False},
        replay_buffer=replay_buffer,
        stacking=cfg.algorithm.using_history_of_obs
    )

    # ------------------ Training finished ------------------
    # Save losses 
    train_losses = [0.0]
    val_scores = [0.0]
    policy_losses = [0.0]
    qf1_losses = [0.0]
    qf2_losses = [0.0]

    def train_callback(_model, _total_calls, _epoch, tr_loss, val_score, _best_val):
        train_losses.append(tr_loss)
        val_scores.append(val_score.mean().item())   # this returns val score per ensemble model

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    rollout_batch_size = (
        cfg.overrides.effective_model_rollouts_per_step * cfg.overrides.freq_train_model
    )
    trains_per_epoch = int(
        np.ceil(cfg.overrides.epoch_length / cfg.overrides.freq_train_model)
    )
    updates_made = 0
    env_steps = 0
    model_env = models.ModelEnvPushing(
        env, dynamics_model, termination_fn=None, reward_fn=None, generator=generator
    )
    model_trainer = models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
    )

    best_eval_reward = -np.inf
    epoch = 0
    sac_buffer = None
    all_train_rewards = [0]
    all_eval_rewards = [0]
    total_steps_train = [0]
    total_steps_eval = [0]
    goal_reached = [0]
    trial_push_result = []
    trial = 0
    while env_steps < cfg.overrides.num_steps:
        rollout_length = int(
            math_util.truncated_linear(
                *(cfg.overrides.rollout_schedule + [epoch + 1])
            )
        )
        sac_buffer_capacity = rollout_length * rollout_batch_size * trains_per_epoch
        sac_buffer_capacity *= cfg.overrides.num_epochs_to_retain_sac_buffer
        sac_buffer = maybe_replace_sac_buffer(
            sac_buffer, stacked_obs_shape, act_shape, sac_buffer_capacity, rng
        )
        obs, done = None, False
        for steps_epoch in range(cfg.overrides.epoch_length):
            if steps_epoch == 0 or done:
                if done:
                    # save goal reached data during training
                    if env.single_goal_reached:
                        goal_reached.append(trial_reward)
                    else:
                        goal_reached.append(0)

                    # Save data to csv and plot
                    all_train_rewards.append(trial_reward)
                    total_steps_train.append(steps_trial + total_steps_train[-1])
                    trial_time = time.time() - start_trial_time

                    # Save data to csv and plot
                    # trial_push_result = np.array(trial_push_result)
                    # plot_and_save_training(env, trial_push_result, trial, data_columns, training_result_directory)
                    trial_push_result = []

                    # Save and plot training curve 
                    training_result = np.stack((total_steps_train[1:], all_train_rewards[1:]), axis=-1)
                    pd.DataFrame(training_result).to_csv(os.path.join(work_dir, "{}_result.csv".format("train_curve")))
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(total_steps_train[1:], all_train_rewards[1:], 'bs-', total_steps_train[1:], goal_reached[1:], 'rs')
                    ax.set_xlabel("Samples")
                    ax.set_ylabel("Trial reward")
                    fig.savefig(os.path.join(work_dir, "output_train.png"))        
                    plt.close(fig)

                    trial += 1

                obs, done = env.reset(), False
                stacked_act = deque(np.zeros((env.states_stacked_len, *env.action_space.shape)), maxlen=env.states_stacked_len)
                trial_reward = 0.0
                trial_pb_steps = 0.0
                steps_trial = 0
                start_trial_time = time.time()

                (tcp_pos_workframe, 
                tcp_rpy_workframe,
                cur_obj_pos_workframe, 
                cur_obj_rpy_workframe) = env.get_obs_workframe()
                trial_push_result.append(np.hstack([trial, 
                                                    steps_trial, 
                                                    trial_pb_steps,
                                                    tcp_pos_workframe, 
                                                    cur_obj_pos_workframe, 
                                                    tcp_rpy_workframe[2],
                                                    cur_obj_rpy_workframe[2],
                                                    env.goal_pos_workframe[0:2], 
                                                    env.goal_rpy_workframe[2],
                                                    trial_reward, 
                                                    False,
                                                    done]))
                            

            # --- Doing env step and adding to model dataset ---
            next_obs, reward, done, info = common_util.step_env_and_add_to_buffer_stacked(
                env, obs, agent, {}, replay_buffer, stacked_action=stacked_act
            )

            # --------------- Model Training -----------------
            if (env_steps + 1) % cfg.overrides.freq_train_model == 0:
                dynamics_model.update_normalizer(replay_buffer.get_all())  # update normalizer stats            
                dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
                    replay_buffer,
                    batch_size=cfg.overrides.model_batch_size,
                    val_ratio=cfg.overrides.validation_ratio,
                    ensemble_size=len(dynamics_model),
                    shuffle_each_epoch=True,
                    bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
                )
                # print("Training model...")
                model_trainer.train(
                    dataset_train, 
                    dataset_val=dataset_val, 
                    num_epochs=cfg.overrides.num_epochs_train_model, 
                    patience=cfg.overrides.patience, 
                    callback=train_callback,
                    silent=True)
                # print("Model training done.")
                
                
                # --------- Rollout new model and store imagined trajectories --------
                # Batch all rollouts for the next freq_train_model steps together
                rollout_model_and_populate_sac_buffer(
                    model_env,
                    replay_buffer,
                    agent,
                    sac_buffer,
                    cfg.algorithm.sac_samples_action,
                    rollout_length,
                    rollout_batch_size,
                )

            # --------------- Agent Training -----------------
            # print("Training agent...")
            for _ in range(cfg.overrides.num_sac_updates_per_step):
                use_real_data = rng.random() < cfg.algorithm.real_data_ratio
                replay_sac_buffer = replay_to_sac_buffer(replay_buffer, stacked_obs_shape, act_shape, rng)
                which_buffer = replay_sac_buffer if use_real_data else sac_buffer
                if (env_steps + 1) % cfg.overrides.sac_updates_every_steps != 0 or len(
                    which_buffer
                ) < cfg.overrides.sac_batch_size:
                    
                    qf1_loss = 0
                    qf2_loss = 0
                    policy_loss = 0
                    # print("Agent training skipped. Buffer not big enough")
                    break  # only update every once in a while
                (
                    qf1_loss,
                    qf2_loss,
                    policy_loss,
                    _,
                    _,
                ) = agent.sac_agent.update_parameters(
                    which_buffer,
                    cfg.overrides.sac_batch_size,
                    updates_made,
                    logger=None,
                    reverse_mask=True,
                )
                updates_made += 1
            # print("Agent training done.")

            qf1_losses.append(qf1_loss)
            qf2_losses.append(qf2_loss)
            policy_losses.append(policy_loss)

            # ------ Epoch ended (evaluate and save model) ------
            if (env_steps + 1) % cfg.overrides.epoch_length == 0:
                avg_reward = eval_and_save_vid(eval_env, agent, n_eval_episodes=num_eval_episodes)
                all_eval_rewards.append(avg_reward)
                total_steps_eval.append(total_steps_train[-1])

                eval_result = np.stack((total_steps_eval[1:], all_eval_rewards[1:]), axis=-1)
                pd.DataFrame(eval_result).to_csv(os.path.join(work_dir, "{}_result.csv".format("eval_curve")))
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(total_steps_eval[1:], all_eval_rewards[1:], 'bs-')
                ax.set_xlabel("Samples")
                ax.set_ylabel("Eval reward")
                fig.savefig(os.path.join(work_dir, "output_eval.png"))        
                plt.close(fig)

                print(
                    f"Epoch: {epoch}. "
                    f"SAC buffer size: {len(sac_buffer)}. "
                    f"Real buffer size: {len(replay_buffer)}. "
                    f"Rollout length: {rollout_length}. "
                    f"Steps: {env_steps}"
                    f"Average reward: {avg_reward} "
                )
                
                if avg_reward > best_eval_reward:
                    best_eval_reward = avg_reward
                    agent.sac_agent.save_checkpoint(
                        ckpt_path=os.path.join(work_dir, "sac.pth")
                    )
                    print("Saved best model")
                epoch += 1

            env_steps += 1
            obs = next_obs
            trial_reward += reward
            trial_pb_steps += info["num_of_pb_steps"]
            steps_trial += 1

            # Save data for plotting training performances
            (tcp_pos_workframe, 
            tcp_rpy_workframe,
            cur_obj_pos_workframe, 
            cur_obj_rpy_workframe) = env.get_obs_workframe()
            trial_push_result.append(np.hstack([trial,
                                            steps_trial,
                                            trial_pb_steps * env._sim_time_step,
                                            tcp_pos_workframe, 
                                            cur_obj_pos_workframe, 
                                            tcp_rpy_workframe[2],
                                            cur_obj_rpy_workframe[2],
                                            env.goal_pos_workframe[0:2], 
                                            env.goal_rpy_workframe[2],
                                            trial_reward, 
                                            info["tip_in_contact"],
                                            done]))
            
        # Plot losses at the end of each epoch
        data = [train_losses, val_scores, qf1_losses, qf2_losses, policy_losses]
        ylabels = ["Train loss", "Val score", "QF1 loss", "QF2 loss", "Policy loss"]
        plot_and_save(data, ylabel=ylabels, work_dir=work_dir)


if __name__ == '__main__':
    
    work_dir = os.path.join(os.getcwd(), 'trained_mbpo')

     # Load the environment 
    algo_name = 'ppo'
    env_name = 'object_push-v0'
    rl_params, algo_params, augmentations = import_parameters(env_name, algo_name)
    rl_params["max_ep_len"] = 1000    
    rl_params["env_modes"][ 'observation_mode'] = 'goal_aware_tactile_pose_relative_data'
    rl_params["env_modes"][ 'control_mode'] = 'TCP_position_control'
    rl_params["env_modes"]['movement_mode'] = 'TyRz'
    rl_params["env_modes"]['traj_type'] = 'point'
    rl_params["env_modes"]['task'] = "goal_pos"
    rl_params["env_modes"]['planar_states'] = True
    rl_params["env_modes"]['use_contact'] = True
    rl_params["env_modes"]['terminate_early']  = True
    rl_params["env_modes"]['terminate_terminate_early'] = True

    rl_params["env_modes"]['rand_init_orn'] = True
    # rl_params["env_modes"]['rand_init_pos_y'] = True
    # rl_params["env_modes"]['rand_obj_mass'] = True

    rl_params["env_modes"]['additional_reward_settings'] = 'john_guide_off_normal'
    rl_params["env_modes"]['terminated_early_penalty'] =  0.0
    rl_params["env_modes"]['reached_goal_reward'] = 0.0 
    rl_params["env_modes"]['max_no_contact_steps'] = 1000
    rl_params["env_modes"]['max_tcp_to_obj_orn'] = 180/180 * np.pi
    rl_params["env_modes"]['importance_obj_goal_pos'] = 1.0
    rl_params["env_modes"]['importance_obj_goal_orn'] = 1.0
    rl_params["env_modes"]['importance_tip_obj_orn'] = 1.0
    rl_params["env_modes"]["x_speed_ratio"] = 5.0
    rl_params["env_modes"]["y_speed_ratio"] = 5.0
    rl_params["env_modes"]["Rz_speed_ratio"] = 5.0

    rl_params["env_modes"]['mpc_goal_orn_update'] = True
    rl_params["env_modes"]['goal_orn_update_freq'] = 'every_step'


    # set limits and goals
    TCP_lims = np.zeros(shape=(6, 2))
    TCP_lims[0, 0], TCP_lims[0, 1] = -0.1, 0.4  # x lims
    TCP_lims[1, 0], TCP_lims[1, 1] = -0.3, 0.3  # y lims
    TCP_lims[2, 0], TCP_lims[2, 1] = -0.0, 0.0  # z lims
    TCP_lims[3, 0], TCP_lims[3, 1] = -0.0, 0.0  # roll lims
    TCP_lims[4, 0], TCP_lims[4, 1] = -0.0, 0.0  # pitch lims
    TCP_lims[5, 0], TCP_lims[5, 1] = -180 * np.pi / 180, 180 * np.pi / 180  # yaw lims

    # goal parameter
    goal_edges = [(0, -1), (0, 1), (1, 0)] # Top bottom and stright
    # goal_edges = [(1, 0)]
    goal_x_max = np.float64(TCP_lims[0, 1] * 0.8).item()
    goal_x_min = 0.0 # np.float64(TCP_lims[0, 0] * 0.6).item()
    goal_y_max = np.float64(TCP_lims[1, 1] * 0.6).item()
    goal_y_min = np.float64(TCP_lims[1, 0] * 0.6).item()
    goal_ranges = [goal_x_min, goal_x_max, goal_y_min, goal_y_max]

    rl_params["env_modes"]['tcp_lims'] = TCP_lims.tolist()
    rl_params["env_modes"]['goal_edges'] = goal_edges
    rl_params["env_modes"]['goal_ranges'] = goal_ranges


    # define parameters
    num_trials = 1000
    initial_buffer_size = 10000
    trial_length = 1000
    buffer_size = num_trials * trial_length

    cfg_dict = {
        # dynamics model configuration
        "dynamics_model": {
            "_target_": "mbrl.models.GaussianMLP",
            "device": device,
            "num_layers": 3,
            "ensemble_size": 5,
            "hid_size": 200,
            "in_size": "???",
            "out_size": "???",
            "deterministic": False,
            "propagation_method": "fixed_model",
            "learn_logvar_bounds": False,
            # can also configure activation function for GaussianMLP
            # "activation_fn_cfg": {
            #     "_target_": "torch.nn.LeakyReLU",
            #     "negative_slope": 0.01
            # }
            "activation_fn_cfg": {
                "_target_": "torch.nn.SiLU",
            }
        },
        # options for training the dynamics model
        "algorithm": {
            "learned_rewards": False,
            "target_is_delta": True,
            "normalize": True,
            "target_normalize": True,
            "dataset_size": buffer_size,
            "initial_dataset_size": initial_buffer_size,
            "using_history_of_obs": True,
            "sac_samples_action": True,
            "real_data_ratio": 0.0,
        },
        # these are experiment specific options
        "overrides": {
            "trial_length": trial_length,
            "epoch_length": trial_length,
            "num_steps": num_trials * trial_length,
            "patience": 5,
            "num_epochs_train_model": None,
            "model_lr": 0.001,
            "model_wd": 0.00005,
            "model_batch_size": 32,
            "validation_ratio": 0.0,
            "freq_train_model": 500,
            "effective_model_rollouts_per_step": 200,
            "rollout_schedule": [10, 100, 1, 20],
            "num_sac_updates_per_step": 20,
            "sac_updates_every_steps": 1,
            "num_epochs_to_retain_sac_buffer": 1,
            "sac_batch_size": 512,
        }
    }

    agent_cfg_dict = {
        "_target_": "mbrl.third_party.pytorch_sac_pranz24.sac.SAC",
        "num_inputs": "???",
        "action_space": {
            "_target_": "gym.spaces.Box",
            "low": "???",
            "high": "???",
            "shape": "???",
        },

        "args": {
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.05,
            "policy": "Gaussian",
            "target_update_interval": 1,
            "automatic_entropy_tuning": False,
            "target_entropy": 0.1,
            "hidden_size": 256,
            "device": device,
            "lr": 0.0001,
        }
    }

    env_kwargs={
        'show_gui':False,
        'show_tactile':False,
        'states_stacked_len': 1,
        'max_steps':rl_params["max_ep_len"],
        'image_size':rl_params["image_size"],
        'env_modes':rl_params["env_modes"],
    }


    train_agent(
        env_kwargs=env_kwargs,
        rl_params=rl_params,
        cfg_dict=cfg_dict,
        agent_cfg_dict=agent_cfg_dict,
        work_dir=work_dir,
    )