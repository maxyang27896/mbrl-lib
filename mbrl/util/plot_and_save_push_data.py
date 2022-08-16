import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys, shutil


def plot_and_save_push_plots(env, data, data_columns, trials, directory, result_type):

    # plot and save results
    if not os.path.exists(directory):
        os.mkdir(directory)
    else:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    # create dataframe
    df = pd.DataFrame(data, columns = data_columns)
    pd.DataFrame(data).to_csv(os.path.join(directory, "{}_result.csv".format(result_type)))

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