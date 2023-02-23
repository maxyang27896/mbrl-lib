from IPython import display
import argparse
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys, shutil
import torch
import omegaconf
import time
import torch

import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util
from mbrl.util.eval_agent import eval_and_save_vid

import tactile_gym.rl_envs

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def clear_and_create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    else:
        for filename in os.listdir(dir):
            file_path = os.path.join(dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def evaluate_and_plot(model_filename, model_number, num_test_trials):

    # model_filename = 'training_model'
    # model_number = 50
    work_dir = os.path.join(os.getcwd(), model_filename)
    # work_dir = r"/home/qt21590/Documents/Projects/tactile_gym_mbrl/training_model/stochastic_icem_H25"
    if model_number:
        model_dir = os.path.join(work_dir, 'model_trial_{}'.format(model_number))
        evaluation_result_directory = os.path.join(work_dir, "evaluation_result_model_{}".format(model_number))
    else:
        model_dir = os.path.join(work_dir, 'best_model')
        evaluation_result_directory = os.path.join(work_dir, "evaluation_result_best_model")

    clear_and_create_dir(evaluation_result_directory)

    # Load the environment 
    env_name = 'object_push-v0'
    env_kwargs_file = 'env_kwargs'
    env_kwargs_dir = os.path.join(work_dir, env_kwargs_file)
    env_kwargs = omegaconf.OmegaConf.load(env_kwargs_dir)
    env_kwargs["env_modes"]['eval_mode'] = True
    env_kwargs["env_modes"]['eval_num'] = num_test_trials
    # env_kwargs["env_modes"]['goal_list'] = [[0.1, 0.18]]


    ####### Other evaluation tasks ##########
    # env_kwargs["env_modes"]['traj_type'] = 'simplex'
    # env_kwargs["env_modes"]['additional_reward_settings'] = 'default'
    # env_kwargs["env_modes"]['terminate_using_center'] = True
    # env_kwargs["env_modes"]['importance_obj_goal_pos'] = 1.0
    # env_kwargs["env_modes"]['importance_obj_goal_orn'] = 1.0
    # env_kwargs["env_modes"]['importance_tip_obj_orn'] = 1.0
    # env_kwargs["env_modes"]['x_speed_ratio'] = 1.0
    # env_kwargs["env_modes"]['terminated_early_penalty'] =  -100
    # env_kwargs["env_modes"]['reached_goal_reward'] = 100
    # env_kwargs["env_modes"]['mpc_goal_orn_update'] = True

    env = gym.make(env_name, **env_kwargs)
    seed = 0
    env.seed(seed)
    rng = np.random.default_rng(seed=0)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # Get cfg and agent cfg
    config_file = 'cfg_dict'
    config_dir = os.path.join(work_dir, config_file)
    cfg = omegaconf.OmegaConf.load(config_dir)
    trial_length= cfg.overrides.trial_length

    agent_config_file = 'agent_cfg'
    agent_config_dir = os.path.join(work_dir, agent_config_file)
    agent_cfg = omegaconf.OmegaConf.load(agent_config_dir)
    # agent_cfg['planning_horizon'] = 40
    # agent_cfg['optimizer_cfg']['population_size'] = 500
    # agent_cfg['optimizer_cfg']['num_iterations'] = 5

    # Re-map device
    map_location = None
    if cfg['dynamics_model']['device'] != device:
        cfg['dynamics_model']['device'] = device
        agent_cfg['optimizer_cfg']['device'] = device
        map_location = torch.device(device)
        
    dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape, model_dir)
    model_env = models.ModelEnvPushing(env, dynamics_model, termination_fn=None, reward_fn=None, generator=generator)
    
    # Create agent 
    agent = planning.create_trajectory_optim_agent_for_model(
        model_env,
        agent_cfg,
        num_particles=20
    )

    # Evaluate agent
    evaluate_time = time.time()
    avg_reward = eval_and_save_vid(
        env,
        agent,
        n_eval_episodes=num_test_trials,
        trial_length=trial_length,
        save_and_plot_flag=True,
        save_vid=True,
        render=True,
        data_directory=evaluation_result_directory,
        print_ep_reward=True,
    )

    print("The average reward over {} episodes is {}, time elapsed {}".format(
    num_test_trials, 
    avg_reward,
    time.time() - evaluate_time)   
    )   


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument(
        "--model_filename",
        type=str,
        default="training_model",
        help="Specify the folder name to which to evaluate the results in",
    )
    parser.add_argument(
        "--model_num",
        type=int,
        default=0,
        help="Model number to test for evaluation.",
    )
    parser.add_argument(
        "--eval_trials",
        type=int,
        default=12,
        help="Number of evaluation trials.",
    )
    args = parser.parse_args()
    evaluate_and_plot(
        args.model_filename,
        args.model_num,
        args.eval_trials,
    )