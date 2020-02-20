import math
import traceback

import click
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

"""
plot the performance of algorithms from TensorBoard Log History
"""
DEFAULT_SIZE_GUIDANCE = {
    "scalars": 0,
}

sns.set(style="darkgrid", font_scale=1.2)


# plt.style.use('bmh')

def plot_data(data, x_axis='num steps', y_axis="average reward", hue="algorithm", smooth=1, ax=None, **kwargs):
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[y_axis])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            datum[y_axis] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True, sort=True)

    sns.lineplot(data=data, x=x_axis, y=y_axis, hue=hue, ci='sd', ax=ax, **kwargs)
    ax.legend(loc='best').set_draggable(True)
    """Spining up style"""
    # plt.legend(loc='upper center', ncol=6, handlelength=1,
    #            mode="expand", borderaxespad=0.02, prop={'size': 13})

    xscale = np.max(np.asarray(data[x_axis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.tight_layout(pad=1.2)


def load_event_scalars(log_path):
    feature = log_path.split(os.sep)[-1]
    print(f"Processing logfile: {log_path}")
    if feature.find("_") != -1:
        feature = feature.split("_")[-1]
    try:
        event_acc = EventAccumulator(log_path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            df = pd.DataFrame({feature: values}, index=step)
    # Dirty catch of DataLossError
    except:
        print("Event file possibly corrupt: {}".format(log_path))
        traceback.print_exc()
    return df


def get_env_alg_log(log_path):
    """
    拆分Env
    :param log_path:
    :return:
    """
    alg = log_path.split(os.sep)[-1]
    if alg.find("_") != -1:
        alg = alg.rsplit("_", maxsplit=1)[0]
    env_alg_fulldir = lambda x: os.path.join(log_path, x)
    alg_features = [env_alg_fulldir(fea) for fea in os.listdir(log_path) if os.path.isdir(env_alg_fulldir(fea))]
    df = pd.concat([load_event_scalars(feature) for feature in alg_features], axis=1)
    if "num steps" in df:
        df["num steps"] = df["num steps"].cumsum()
    else:
        df["num steps"] = (np.ones((1, df.shape[0])) * 3000).cumsum()
    df["algorithm"] = [alg] * df.shape[0]
    return df


def plot_all_logs(log_dir=None, x_axis=None, y_axis=None, hue=None, smooth=1, env_filter_func=None,
                  alg_filter_func=None):
    if y_axis is None:
        y_axis = ['min reward', 'average reward', 'max reward', 'total reward']

    basedir = os.path.dirname(log_dir)  # ../log/
    fulldir = lambda x: os.path.join(basedir, x)  # ../log/Ant-v3/
    envs_logdirs = sorted(
        [fulldir(x) for x in os.listdir(basedir) if os.path.isdir(fulldir(x))])  # [../log/Ant-v3/, ../log/Hopper-v3/]
    if env_filter_func:
        envs_logdirs = sorted(filter(env_filter_func, envs_logdirs))
    print("All envs are: ", envs_logdirs)

    num_envs = len(envs_logdirs)
    sub_plot_height = math.floor(math.sqrt(num_envs))
    sub_plot_width = num_envs // sub_plot_height

    envs_fulldir = lambda env_dir, alg_dir: os.path.join(env_dir, alg_dir)
    for y_ax in y_axis:
        k = 0
        fig, axes = plt.subplots(sub_plot_height, sub_plot_width, figsize=(7 * sub_plot_width, 5 * sub_plot_height))
        for env_dir in envs_logdirs:
            if sub_plot_width == 1 and sub_plot_height == 1:
                ax = axes
            else:
                ax = axes[k // sub_plot_width][k % sub_plot_width]

            env_id = env_dir.split(os.sep)[-1]
            env_alg_dirs = sorted(
                filter(os.path.isdir, [envs_fulldir(env_dir, alg_dir) for alg_dir in os.listdir(env_dir)]))
            if alg_filter_func:
                env_alg_dirs = sorted(filter(alg_filter_func, env_alg_dirs))
            print(env_alg_dirs)

            env_log_df = [get_env_alg_log(env_alg_dir) for env_alg_dir in env_alg_dirs]
            make_plot(data=env_log_df, x_axis=x_axis, y_axis=y_ax, smooth=smooth, title=env_id, hue=hue, ax=ax)
            k += 1
    plt.show()


def make_plot(data, x_axis=None, y_axis=None, title=None, hue=None, smooth=1, estimator='mean', ax=None):
    estimator = getattr(np, estimator)
    plot_data(data, x_axis=x_axis, y_axis=y_axis, hue=hue, smooth=smooth, ax=ax, estimator=estimator)
    if title:
        ax.set_title(title)


# @click.command()
# @click.option("--log_dir", type=str, default="../log/", help="Directory to load tensorboard logs")
# @click.option("--x_axis", type=str, default="num steps", help="X axis data")
# @click.option("--y_axis", type=list, default=["average reward"], help="Y axis data(can be multiple)")
# @click.option("--hue", type=str, default="algorithm", help="Hue for legend")
# def main(log_dir, x_axis, y_axis, hue):
def main(log_dir='../log/', x_axis='num steps', y_axis=['average reward'], hue='algorithm',
         env_filter_func=None, alg_filter_func=None):
    """
    1.遍历所有环境, 对每个环境下所有算法的log信息进行绘图
    2.对每个环境下的所有算法数据载入一个 data frame
    :return:
    """
    plot_all_logs(log_dir=log_dir, x_axis=x_axis, y_axis=y_axis, hue=hue,
                  smooth=11,
                  env_filter_func=env_filter_func,
                  alg_filter_func=alg_filter_func)


if __name__ == "__main__":
    # env_filter_func = lambda x: x.split(os.sep)[-1] == "BipedalWalker-v2"
    # alg_filter_func = lambda x: x.split(os.sep)[-1].rsplit("_")[0] in ["PPO", "TRPO"]
    main(env_filter_func=None, alg_filter_func=None)
