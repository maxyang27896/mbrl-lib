from calendar import TUESDAY
from IPython import display
import argparse
import cv2
import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys, shutil
import torch
import omegaconf
import time
import torch

import mbrl.env.cartpole_continuous as cartpole_env
import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns
import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util
import mbrl.util as util
from mbrl.util.plot_and_save_push_data import plot_and_save_training

import tactile_gym.rl_envs
from tactile_gym.sb3_helpers.params import import_parameters
from tactile_gym.rl_envs.nonprehensile_manipulation.object_push.object_push_env import get_states_from_obs

from pyvirtualdisplay import Display
_display = Display(visible=False, size=(1400, 900))
_ = _display.start()

def train_and_plot(num_trials, model_filename):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Define model working directorys
    # model_filename = 'training_model'
    work_dir = os.path.join(os.getcwd(), model_filename)
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    else:
        for filename in os.listdir(work_dir):
            file_path = os.path.join(work_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Load the environment 
    # env_name = 'object_push-v0'
    # env_kwargs_file = 'env_kwargs'
    # env_kwargs_dir = os.path.join(os.getcwd(), "training_cfg", env_kwargs_file)
    # env_kwargs = omegaconf.OmegaConf.load(env_kwargs_dir)
    algo_name = 'ppo'
    env_name = 'object_push-v0'
    rl_params, algo_params, augmentations = import_parameters(env_name, algo_name)
    rl_params["max_ep_len"] = 2000    
    rl_params["env_modes"][ 'observation_mode'] = 'tactile_pose_relative_data'
    rl_params["env_modes"][ 'control_mode'] = 'TCP_position_control'
    rl_params["env_modes"]['movement_mode'] = 'TyRz'
    rl_params["env_modes"]['traj_type'] = 'point'
    rl_params["env_modes"]['task'] = "goal_pos"
    rl_params["env_modes"]['planar_states'] = True
    rl_params["env_modes"]['use_contact'] = True
    rl_params["env_modes"]['terminate_early']  = True
    rl_params["env_modes"]['terminate_terminate_early'] = True

    rl_params["env_modes"]['additional_reward_settings'] = 'john_guide_off_normal'
    rl_params["env_modes"]['terminated_early_penalty'] =  -100
    rl_params["env_modes"]['reached_goal_reward'] = 100
    rl_params["env_modes"]['importance_obj_goal_pos'] = 5.0
    rl_params["env_modes"]['importance_obj_goal_orn'] = 1.0
    rl_params["env_modes"]['importance_tip_obj_orn'] = 1.0

    rl_params["env_modes"]['mpc_goal_orn_update'] = False
    rl_params["env_modes"]['goal_orn_update_freq'] = 'every_step'


    # set limits and goals
    TCP_lims = np.zeros(shape=(6, 2))
    TCP_lims[0, 0], TCP_lims[0, 1] = -0.1, 0.3  # x lims
    TCP_lims[1, 0], TCP_lims[1, 1] = -0.3, 0.3  # y lims
    TCP_lims[2, 0], TCP_lims[2, 1] = -0.0, 0.0  # z lims
    TCP_lims[3, 0], TCP_lims[3, 1] = -0.0, 0.0  # roll lims
    TCP_lims[4, 0], TCP_lims[4, 1] = -0.0, 0.0  # pitch lims
    TCP_lims[5, 0], TCP_lims[5, 1] = -180 * np.pi / 180, 180 * np.pi / 180  # yaw lims

    # goal parameter
    goal_edges = [(0, -1), (0, 1), (1, 0)] # Top bottom and stright
    # goal_edges = [(1, 0)]
    goal_x_max = np.float64(TCP_lims[0, 1] * 0.6).item()
    goal_x_min = 0.0 # np.float64(TCP_lims[0, 0] * 0.6).item()
    goal_y_max = np.float64(TCP_lims[1, 1] * 0.6).item()
    goal_y_min = np.float64(TCP_lims[1, 0] * 0.6).item()
    goal_ranges = [goal_x_min, goal_x_max, goal_y_min, goal_y_max]

    rl_params["env_modes"]['tcp_lims'] = TCP_lims.tolist()
    rl_params["env_modes"]['goal_edges'] = goal_edges
    rl_params["env_modes"]['goal_ranges'] = goal_ranges

    env_kwargs={
        'show_gui':False,
        'show_tactile':False,
        'max_steps':rl_params["max_ep_len"],
        'image_size':rl_params["image_size"],
        'env_modes':rl_params["env_modes"],
    }

    env = gym.make(env_name, **env_kwargs)
    seed = 0
    env.seed(seed)
    rng = np.random.default_rng(seed=0)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # Load dynamics model and model environment
    # config_file = 'cfg_dict'
    # config_dir = os.path.join(os.getcwd(), "training_cfg", config_file)
    # cfg = omegaconf.OmegaConf.load(config_dir)
    # # num_trials = 80
    # trial_length= cfg.overrides.trial_length
    # ensemble_size = cfg.dynamics_model.ensemble_size
    # cfg.overrides.num_steps = num_trials * cfg.overrides.trial_length
    # cfg.algorithm.dataset_size = num_trials * cfg.overrides.trial_length

    trial_length = env._max_steps
    ensemble_size = 5
    initial_buffer_size = 2000
    buffer_size = num_trials * trial_length
    target_normalised = True
    cfg_dict = {
        # dynamics model configuration
        "dynamics_model": {
            "_target_": "mbrl.models.GaussianMLP",
            "device": device,
            "num_layers": 3,
            "ensemble_size": ensemble_size,
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
            "target_normalize": target_normalised,
            "dataset_size": buffer_size
        },
        # these are experiment specific options
        "overrides": {
            "trial_length": trial_length,
            "num_steps": num_trials * trial_length,
            "model_batch_size": 32,
            "validation_ratio": 0.05
        }
    }
    cfg = omegaconf.OmegaConf.create(cfg_dict)

    dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    model_env = models.ModelEnvPushing(env, dynamics_model, termination_fn=None, reward_fn=None, generator=generator)

    # Load agent
    # agent_config_file = 'agent_cfg'
    # agent_config_dir = os.path.join(os.getcwd(), "training_cfg", agent_config_file)
    # agent_cfg = omegaconf.OmegaConf.load(agent_config_dir)

    optimizer_type = "icem"
    if optimizer_type == "cem":
        optimizer_cfg = {
                "_target_": "mbrl.planning.CEMOptimizer",
                "num_iterations": 5,
                "elite_ratio": 0.1,
                "population_size": 500,
                "alpha": 0.1,
                "device": device,
                "lower_bound": "???",
                "upper_bound": "???",
                "return_mean_elites": True,
                "clipped_normal": False
            }
    elif optimizer_type == "mppi":
        optimizer_cfg = {
                "_target_": "mbrl.planning.MPPIOptimizer",
                "num_iterations": 5,
                "gamma": 10.0,
                "population_size": 500,
                "sigma": 1.0,
                "beta": 0.7,
                "lower_bound": "???",
                "upper_bound": "???",
                "device": device,
            }

    elif optimizer_type == "icem":
        optimizer_cfg = {
                "_target_": "mbrl.planning.ICEMOptimizer",
                "num_iterations": 5,
                "elite_ratio": 0.1,
                "population_size": 500,
                "population_decay_factor": 1.25,
                "colored_noise_exponent": 2.0,
                "keep_elite_frac": 0.1,
                "alpha": 0.1,
                "lower_bound": "???",
                "upper_bound": "???",
                "device": device,
            }
    else:
        raise ValueError

    agent_cfg = omegaconf.OmegaConf.create({
        # this class evaluates many trajectories and picks the best one
        "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
        "planning_horizon": 15,
        "replan_freq": 1,
        "verbose": False,
        "action_lb": "???",
        "action_ub": "???",
        # this is the optimizer to generate and choose a trajectory
        "optimizer_cfg": optimizer_cfg
    })

    # Create agent 
    agent = planning.create_trajectory_optim_agent_for_model(
        model_env,
        agent_cfg,
        num_particles=20
    )

    # create buffer
    replay_buffer = common_util.create_replay_buffer(cfg, obs_shape, act_shape, rng=rng)
    common_util.rollout_agent_trajectories(
        env,
        2000, # initial exploration steps
        planning.RandomAgent(env),
        {}, # keyword arguments to pass to agent.act()
        replay_buffer=replay_buffer,
        trial_length=trial_length
    )

    # Create a trainer for the model
    model_trainer = models.ModelTrainer(dynamics_model, optim_lr=1e-3, weight_decay=5e-5)

    # Saving config files
    config_filename = 'cfg_dict'
    config_dir = os.path.join(work_dir, config_filename)
    omegaconf.OmegaConf.save(config=cfg, f=config_dir) 
    loaded = omegaconf.OmegaConf.load(config_dir)
    assert cfg == loaded

    agent_config_filename = 'agent_cfg'
    agent_config_dir = os.path.join(work_dir, agent_config_filename)
    omegaconf.OmegaConf.save(config=agent_cfg, f=agent_config_dir) 
    loaded = omegaconf.OmegaConf.load(agent_config_dir)
    assert agent_cfg == loaded

    env_kwargs_filename = 'env_kwargs'
    env_kwargs_dir = os.path.join(work_dir, env_kwargs_filename)
    omegaconf.OmegaConf.save(config=env_kwargs, f=env_kwargs_dir) 
    loaded = omegaconf.OmegaConf.load(env_kwargs_dir)
    assert env_kwargs == loaded

    ######### Main PETS loop #############
    all_rewards = [0]
    total_steps = [0]
    goal_reached = [0]
    training_result = []
    plan_time = 0.0
    train_time = 0.0

    # Save training data
    save_model_freqency = 5
    data_columns = ['trial','trial_steps', 'time_steps', 'tcp_x','tcp_y','tcp_z','contact_x', 'contact_y', 'contact_z', 'tcp_Rz', 'contact_Rz', 'goal_x', 'goal_y', 'goal_Rz', 'rewards', 'contact', 'dones']
    training_result_directory = os.path.join(work_dir, "training_result")
    record_video = True
    record_video_frequency = 5
    
    # parametes
    train_losses = [0.0]
    val_scores = [0.0]

    def train_callback(_model, _total_calls, _epoch, tr_loss, val_score, _best_val):
        train_losses.append(tr_loss)
        val_scores.append(val_score.mean().item())   # this returns val score per ensemble model
        

    for trial in range(num_trials):
        # Reset 
        obs = env.reset()    
        agent.reset()
        done = False
        trial_reward = 0.0
        trial_pb_steps = 0.0
        steps_trial = 0
        start_trial_time = time.time()

        # Record video
        if record_video and (trial+1) % record_video_frequency == 0:
            record_every_n_frames = 3
            render_img = env.render(mode="rgb_array")
            render_img_size = (render_img.shape[1], render_img.shape[0])
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                os.path.join(work_dir, "training_policy_trial_{}.mp4".format((trial+1))),
                fourcc,
                24.0,
                render_img_size,
            )
        
        (tcp_pos_workframe, 
        tcp_rpy_workframe,
        cur_obj_pos_workframe, 
        cur_obj_rpy_workframe) = env.get_obs_workframe()
        training_result.append(np.hstack([trial, 
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
        while not done:

            if steps_trial == 0:
                # --------------- Model Training -----------------
                dynamics_model.update_normalizer(replay_buffer.get_all())  # update normalizer stats            
                dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
                    replay_buffer,
                    batch_size=cfg.overrides.model_batch_size,
                    val_ratio=cfg.overrides.validation_ratio,
                    ensemble_size=ensemble_size,
                    shuffle_each_epoch=True,
                    bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
                )
                
                start_train_time = time.time()
                model_trainer.train(
                    dataset_train, 
                    dataset_val=dataset_val, 
                    num_epochs=50, 
                    patience=50, 
                    callback=train_callback,
                    silent=True)
                train_time = time.time() - start_train_time

                # save model at regular frequencies
                if (trial+1) % save_model_freqency  == 0 and trial >= 29:
                    model_dir = os.path.join(work_dir, 'model_trial_{}'.format(trial+1))
                    os.makedirs(model_dir, exist_ok=True)
                    dynamics_model.save(str(model_dir))

                replay_buffer.save(work_dir)

            # --- Doing env step using the agent and adding to model dataset ---
            # start_plan_time = time.time()
            next_obs, reward, done, info = common_util.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer)
            # plan_time = time.time() - start_plan_time

            obs = next_obs
            trial_reward += reward
            trial_pb_steps += info["num_of_pb_steps"]
            steps_trial += 1

            # Save data for plotting training performance
            (tcp_pos_workframe, 
            tcp_rpy_workframe,
            cur_obj_pos_workframe, 
            cur_obj_rpy_workframe) = env.get_obs_workframe()
            training_result.append(np.hstack([trial,
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
            
            # Record video at every n trials
            if record_video and (trial+1) % record_video_frequency == 0 and steps_trial % record_every_n_frames == 0:
                render_img = env.render(mode="rgb_array")
                render_img = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
                out.write(render_img)

            if steps_trial == trial_length:
                break

        all_rewards.append(trial_reward)
        total_steps.append(steps_trial + total_steps[-1])
        trial_time = time.time() - start_trial_time

        # Save data to csv and plot
        training_result = np.array(training_result)
        plot_and_save_training(env, training_result, trial, data_columns, training_result_directory)
        training_result = []

        # save goal reached data during training
        if env.single_goal_reached:
            goal_reached.append(trial_reward)
        else:
            goal_reached.append(0)

        # release video at every n trials
        if record_video and (trial+1) % record_video_frequency == 0:
            out.release()

        print("Trial {}, total steps {}, rewards {}, goal reached {}, time elapsed {}".format(trial+1, steps_trial, all_rewards[-1], env.single_goal_reached, trial_time))

        # Plot training curve 
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(total_steps[1:], all_rewards[1:], 'bs-', total_steps[1:], goal_reached[1:], 'rs')
        ax.set_xlabel("Samples")
        ax.set_ylabel("Trial reward")
        fig.savefig(os.path.join(work_dir, "output.png"))
        plt.close(fig)

    # Save all data 
    # training_result = np.array(training_result)
    # data_columns = ['trial','trial_steps', 'time_steps', 'tcp_x','tcp_y','tcp_z','contact_x', 'contact_y', 'contact_z', 'tcp_Rz', 'contact_Rz', 'goal_x', 'goal_y', 'goal_z', 'rewards', 'contact', 'dones']

    # # Plot the training results
    # training_result_directory = os.path.join(work_dir, "training_result")
    # plot_and_save_push_plots(env, training_result, data_columns, num_trials, training_result_directory, "training")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_trials",
        type=int,
        default=30,
        help="Number of training trials.",
    )
    parser.add_argument(
        "--model_filename",
        type=str,
        default="training_model",
        help="Specify the folder name to which to save the results in.",
    )

    args = parser.parse_args()
    train_and_plot(
        args.num_trials,
        args.model_filename
    )