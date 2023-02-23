from IPython import display
import cv2
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import time

from mbrl.util.plot_and_save_push_data import plot_and_save_push_plots

from pyvirtualdisplay import Display
_display = Display(visible=False, size=(1400, 900))
_ = _display.start()

DATA_COLUMN =  [
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
    'action_y',
    'action_Rz',
    'goal_reached', 
    'rewards', 
    'contact', 
    'dones',
    ]

def eval_and_save_vid(
        eval_env, 
        agent, 
        n_eval_episodes=10,
        trial_length=1000,
        save_and_plot_flag=False, 
        save_vid=False,
        render=False,
        data_directory=None,
        print_ep_reward=False,
        agent_kwargs={},
    ):
    
    if save_vid:
        record_every_n_frames = 3
        render_img = eval_env.render(mode="rgb_array")
        render_img_size = (render_img.shape[1], render_img.shape[0])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            os.path.join(data_directory, "evaluated_policy.mp4"),
            fourcc,
            24.0,
            render_img_size,
        )

    # Initial params
    all_rewards = []
    evaluation_result = []
    goal_reached = []
    for trial in range(n_eval_episodes):
        obs = eval_env.reset()  
        stacked_act = deque(np.zeros((eval_env.states_stacked_len, *eval_env.action_space.shape)), maxlen=eval_env.states_stacked_len)
        agent.reset()
        
        done = False
        trial_reward = 0.0
        trial_pb_steps = 0.0
        steps_trial = 0
        start_trial_time = time.time()

        (tcp_pos_workframe, 
        tcp_rpy_workframe,
        cur_obj_pos_workframe, 
        cur_obj_rpy_workframe) = eval_env.get_obs_workframe()
        evaluation_result.append(np.hstack([trial, 
                                            steps_trial, 
                                            trial_pb_steps,
                                            tcp_pos_workframe, 
                                            cur_obj_pos_workframe, 
                                            tcp_rpy_workframe[2],
                                            cur_obj_rpy_workframe[2],
                                            eval_env.goal_pos_workframe[0:2], 
                                            eval_env.goal_rpy_workframe[2],
                                            np.array([0, 0]),
                                            eval_env.goal_updated,
                                            trial_reward, 
                                            False,
                                            done]))

        while not done:

            # --- Doing env step using the agent and adding to model dataset ---
            action = agent.act(obs.reshape(-1), np.array(stacked_act).reshape(-1), **agent_kwargs)
            next_obs, reward, done, info = eval_env.step(action)

            if render:
                render_img = eval_env.render(mode="rgb_array")
            else:
                render_img = None

            stacked_act.append(action)
            obs = next_obs.reshape(-1)
            trial_reward += reward
            trial_pb_steps += info["num_of_pb_steps"]
            steps_trial += 1

            if done:
                current_goal_reached = eval_env.single_goal_reached
            else:
                current_goal_reached = eval_env.goal_updated,
            
            (tcp_pos_workframe, 
            tcp_rpy_workframe,
            cur_obj_pos_workframe, 
            cur_obj_rpy_workframe) = eval_env.get_obs_workframe()
            evaluation_result.append(np.hstack([trial,
                                                steps_trial,
                                                trial_pb_steps * eval_env._sim_time_step,
                                                tcp_pos_workframe, 
                                                cur_obj_pos_workframe, 
                                                tcp_rpy_workframe[2],
                                                cur_obj_rpy_workframe[2],
                                                eval_env.goal_pos_workframe[0:2], 
                                                eval_env.goal_rpy_workframe[2],
                                                action,
                                                current_goal_reached,
                                                trial_reward, 
                                                info["tip_in_contact"],
                                                done]))

            # use record_every_n_frames to reduce size sometimes
            if save_vid and steps_trial % record_every_n_frames == 0:

                # warning to enable rendering
                if render_img is None:
                    sys.exit('Must be rendering to save video')

                render_img = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
                out.write(render_img)

            if steps_trial == trial_length:
                break
        
        # Save episodic data
        all_rewards.append(trial_reward)
        if eval_env.single_goal_reached:
            goal_reached.append(trial_reward)
        else:
            goal_reached.append(0)

        if print_ep_reward:
            print("Terminated at step {} with reward {}, goal reached: {}, time elapsed {}".format(
                steps_trial, 
                trial_reward, 
                eval_env.single_goal_reached,
                time.time() - start_trial_time)
                )
    
    # Save evalutation data 
    if save_vid:
        out.release()

    if save_and_plot_flag:
        evaluation_result = np.array(evaluation_result)
        plot_and_save_push_plots(eval_env, evaluation_result, DATA_COLUMN, n_eval_episodes, data_directory, "evaluation")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(all_rewards, 'bs-', goal_reached, 'rs')
        ax.set_xlabel("Trial")
        ax.set_ylabel("Trial reward")
        fig.savefig(os.path.join(data_directory, "evaluation_output.png"))
        plt.close(fig)

    return np.mean(all_rewards)
