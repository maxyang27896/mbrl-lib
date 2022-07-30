from IPython import display
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

import tactile_gym.rl_envs
from tactile_gym.sb3_helpers.params import import_parameters

from pyvirtualdisplay import Display
_display = Display(visible=False, size=(1400, 900))
_ = _display.start()

def get_states_from_obs(obs):
    if "goal_excluded" in env.observation_mode:
        if env.planar_states == True:
            # tcp_pos_workframe = np.zeros(3)
            # tcp_orn_workframe = np.zeros(4)
            cur_obj_pos_workframe = np.zeros(3)
            # cur_obj_orn_workframe = np.zeros(4)

            # tcp_pos_workframe[0:2] = obs[0:2]
            # tcp_orn_workframe[2:4] = obs[2:4]
            cur_obj_pos_workframe[0:2]= obs[4:6]
            # cur_obj_orn_workframe[:, 2:4] = obs[6:8]
        else:   
            # tcp_pos_workframe = obs[0:3]
            # tcp_orn_workframe = obs[0:4]
            cur_obj_pos_workframe = obs[4:7]
            # cur_obj_orn_workframe = obs[7:11]

    else:
        if env.planar_states == True: 
            # tcp_pos_to_goal_workframe = np.zeros(3)
            # tcp_orn_to_goal_workframe = np.zeros(4)
            cur_obj_pos_to_goal_workframe = np.zeros(3)
            # cur_obj_orn_to_goal_workframe = np.zeros(4)

            # tcp_pos_to_goal_workframe[0:2] = obs[0:2]
            # tcp_orn_to_goal_workframe[2:4] = obs[0:2]
            cur_obj_pos_to_goal_workframe[0:2]= obs[2:4]
            # cur_obj_orn_to_goal_workframe[2:4] = obs[4:6]
        else:
            # tcp_pos_to_goal_workframe = obs[0:3]
            # tcp_orn_to_goal_workframe = obs[0:4]
            cur_obj_pos_to_goal_workframe = obs[4:7]
            # cur_obj_orn_to_goal_workframe = obs[7:11]

        # tcp_pos_workframe = obs[0:3] + env.goal_pos_workframe
        cur_obj_pos_workframe = cur_obj_pos_to_goal_workframe + env.goal_pos_workframe

    return cur_obj_pos_workframe

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Define model working directorys
model_filename = 'training_model'
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
env_name = 'object_push-v0'
env_kwargs_file = 'env_kwargs'
env_kwargs_dir = os.path.join(os.getcwd(), "training_cfg", env_kwargs_file)
env_kwargs = omegaconf.OmegaConf.load(env_kwargs_dir)

env = gym.make(env_name, **env_kwargs)
seed = 0
env.seed(seed)
rng = np.random.default_rng(seed=0)
generator = torch.Generator(device=device)
generator.manual_seed(seed)
obs_shape = env.observation_space.shape
act_shape = env.action_space.shape

# Load dynamics model and model environment
config_file = 'cfg_dict'
config_dir = os.path.join(os.getcwd(), "training_cfg", config_file)
cfg = omegaconf.OmegaConf.load(config_dir)
num_trials = 30
trial_length= cfg.overrides.trial_length
ensemble_size = cfg.dynamics_model.ensemble_size
cfg.overrides.num_steps = num_trials * cfg.overrides.trial_length
cfg.algorithm.dataset_size = num_trials * cfg.overrides.trial_length
dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)
model_env = models.ModelEnvPushing(env, dynamics_model, termination_fn=None, reward_fn=None, generator=generator)

# Load agent
agent_config_file = 'agent_cfg'
agent_config_dir = os.path.join(os.getcwd(), "training_cfg", agent_config_file)
agent_cfg = omegaconf.OmegaConf.load(agent_config_dir)

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
    
    tcp_pos_workframe, _, _, _, _ = env.robot.arm.get_current_TCP_pos_vel_workframe()
    cur_obj_pos_workframe = get_states_from_obs(obs)
    training_result.append(np.hstack([trial, 
                                    steps_trial, 
                                    trial_pb_steps,
                                    tcp_pos_workframe, 
                                    cur_obj_pos_workframe, 
                                    env.goal_pos_workframe, 
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

            if work_dir is not None:
                dynamics_model.save(str(work_dir))
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
        tcp_pos_workframe, _, _, _, _ = env.robot.arm.get_current_TCP_pos_vel_workframe()
        cur_obj_pos_workframe = get_states_from_obs(obs)
        training_result.append(np.hstack([trial,
                                        steps_trial,
                                        trial_pb_steps * env._sim_time_step,
                                        tcp_pos_workframe, 
                                        cur_obj_pos_workframe, 
                                        env.goal_pos_workframe, 
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

    # save goal reached data during training
    if env.single_goal_reached:
        goal_reached.append(trial_reward)
    else:
        goal_reached.append(0)

    # release video at every n trials
    if record_video and (trial+1) % record_video_frequency == 0:
        out.release()

    print("Trial {}, total steps {}, rewards {}, goal reached {}, time elapsed {}".format(trial+1, steps_trial, all_rewards[-1], env.single_goal_reached, trial_time))


# Plot results
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(total_steps[1:], all_rewards[1:], 'bs-', total_steps[1:], goal_reached[1:], 'rs')
ax.set_xlabel("Samples")
ax.set_ylabel("Trial reward")
fig.savefig(os.path.join(work_dir, "output.png"))

def plot_and_save_push_plots(df, trials, directory):
    loss_contact = False
    for trial in range(trials):
        fig_xy, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.query("trial==@trial")["tcp_x"], df.query("trial==@trial")["tcp_y"], "bs", label='tcp psosition')
        ax.plot(df.query("trial==@trial").query("contact==@loss_contact")["tcp_x"], df.query("trial==@trial").query("contact==@loss_contact")["tcp_y"], "g+", markersize=20)
        ax.plot(df.query("trial==@trial")["contact_x"], df.query("trial==@trial")["contact_y"], "rs", label='contact psosition')
        ax.plot(df.query("trial==@trial").query("contact==@loss_contact")["contact_x"], df.query("trial==@trial").query("contact==@loss_contact")["contact_y"], "gx", markersize=20)
        ax.plot(df.query("trial==@trial")["goal_x"].iloc[0], df.query("trial==@trial")["goal_y"].iloc[0], "x", markersize=20, markeredgecolor="black", label="goal position")
        ax.set_xlabel("x workframe")
        ax.set_ylabel("y workframe")
        ax.set_xlim([env.robot.arm.TCP_lims[0, 0], env.robot.arm.TCP_lims[0, 1]])
        ax.set_ylim([env.robot.arm.TCP_lims[1, 0], env.robot.arm.TCP_lims[1, 1]])
        ax.legend()
        fig_xy.savefig(os.path.join(directory, "workframe_plot_trial_{}.png".format(trial)))
        plt.close(fig_xy)

        fig_time_xy, axs = plt.subplots(1, 2, figsize=(14, 3.75), gridspec_kw={"width_ratios": [1, 1]})
        axs[0].plot(df.query("trial==@trial")["time_steps"], df.query("trial==@trial")["tcp_x"], "bs", label='tcp ')
        axs[0].plot(df.query("trial==@trial").query("contact==@loss_contact")["time_steps"], df.query("trial==@trial").query("contact==@loss_contact")["tcp_x"], "g+", markersize=20)
        axs[0].plot(df.query("trial==@trial")["time_steps"], df.query("trial==@trial")["contact_x"], "rs", label='contact')
        axs[0].plot(df.query("trial==@trial").query("contact==@loss_contact")["time_steps"], df.query("trial==@trial").query("contact==@loss_contact")["contact_x"], "gx", markersize=20)
        axs[0].set_xlabel("Time steps (s)")
        axs[0].set_ylabel("x axis workframe")
        axs[0].set_ylim([env.robot.arm.TCP_lims[0, 0], env.robot.arm.TCP_lims[0, 1]])
        axs[0].legend()
        axs[1].plot(df.query("trial==@trial")["time_steps"], df.query("trial==@trial")["tcp_y"], "bs", label='tcp')
        axs[1].plot(df.query("trial==@trial").query("contact==@loss_contact")["time_steps"], df.query("trial==@trial").query("contact==@loss_contact")["tcp_y"], "g+", markersize=20)
        axs[1].plot(df.query("trial==@trial")["time_steps"], df.query("trial==@trial")["contact_y"], "rs", label='contact')
        axs[1].plot(df.query("trial==@trial").query("contact==@loss_contact")["time_steps"], df.query("trial==@trial").query("contact==@loss_contact")["contact_y"], "gx", markersize=20)
        axs[1].set_xlabel("Time steps (s)")
        axs[1].set_ylabel("y axis workframe")
        axs[1].set_ylim([env.robot.arm.TCP_lims[1, 0], env.robot.arm.TCP_lims[1, 1]])
        axs[1].legend()
        fig_time_xy.savefig(os.path.join(directory, "time_plot_trial_{}.png".format(trial)))
        plt.close(fig_time_xy)

# Save data 
training_result = np.array(training_result)
data_columns = ['trial','trial_steps', 'time_steps', 'tcp_x','tcp_y','tcp_z','contact_x', 'contact_y', 'contact_z', 'goal_x', 'goal_y', 'goal_z', 'rewards', 'contact', 'dones']
df_training = pd.DataFrame(training_result, columns = data_columns)
pd.DataFrame(training_result).to_csv(os.path.join(work_dir, "training_results.csv"))

# Plot the training results
training_result_directory = os.path.join(work_dir, "training_result")
if not os.path.exists(training_result_directory):
    os.mkdir(training_result_directory)
else:
    for filename in os.listdir(training_result_directory):
        file_path = os.path.join(training_result_directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

plot_and_save_push_plots(df_training, num_trials, training_result_directory)
