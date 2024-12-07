import wandb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib
import argparse
import difflib
import numpy as np
import tqdm
import torch
from copy import deepcopy
from collections import defaultdict
from joypy import joyplot
import pandas as pd


from matplotlib import colormaps as cm
# from matplotlib import cm
from scipy.optimize import minimize
from matplotlib import rcParams

############################################################
# PLOTTING UTILS

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.style"] = "normal"
plt.rcParams["font.weight"] = "ultralight"

# Customize the tick settings
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.size"] = 3
plt.rcParams["ytick.major.size"] = 3
plt.rcParams["xtick.minor.size"] = 2
plt.rcParams["ytick.minor.size"] = 2
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5
plt.rcParams["xtick.minor.width"] = 0.5
plt.rcParams["ytick.minor.width"] = 0.5

plt.rcParams["font.size"] = 23
plt.rcParams["figure.figsize"] = [4, 5]

matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
matplotlib.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

def thousands_formatter(x, pos):
    """Formatter to display numbers in the format of 10k, 20k, 30k, etc."""
    # return f'{int(x/1000)}k'
    if x == 0:
        return "0"
    if x < 1000:
        return f"{int(x)}"
    # if x < 10000:
    #     return f"{x/1000:.1f}k"
    if x < 1000000:
        return f"{int(x/1000)}k"

    return f"{x/1000000:.1f}M"

C = "$\mathcal{C}$"
def delay_filter(delay):
    return lambda x: x.config["online/delay"] == delay

def repeat_filter(repeat):
    return lambda x: x.config["online/batch_repeat"] == repeat

def concat_filter(filters):
    return lambda x: all([f(x) for f in filters])

def or_filter(filters):
    return lambda x: any([f(x) for f in filters])

def set_default_settings(ax=None):
    if ax is None:
        ax = plt.gca()

    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=5)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(thousands_formatter))
    ax.grid(True, linestyle="--", which='major',axis='both',alpha=0.3)

    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def get_run_metrics(run, x_metric="timestep_step", y_metric="next_batch_acc1_step"):
    run_metrics = run.history(
        keys=[x_metric, y_metric],
        x_axis=x_metric,
        samples=1000,
        pandas=False,
    )
    # run_metrics is a list of dicts with values for each metric at each step
    # sort the list by the x_metric
    run_metrics.sort(key=lambda x: x[x_metric])

    x = [row[x_metric] for row in run_metrics]
    y = [row[y_metric] for row in run_metrics]
    return x, y

class Plotter:
    """
    Aligns the annotations of the plot to avoid overlaps
    """

    def __init__(self, ax=None):
        self.ax = ax or plt.gca()
        self.lines = []
        self.last_ys = []
        self.last_xs = []
        self.colors = []
        self.markers = [
            "*",
            "d",
            "X",
            "P",
            "^",
            "v",
            "^",
        ]

    def plot(self, x, y, label=None, color=None, marker=None, **kwargs):
        line_idx = len(self.lines)

        if color is None:
            color = f"C{line_idx}"
        if marker is None:
            marker = self.markers[line_idx]

        line, = self.ax.plot(
            x, y,
            color=color,
            **kwargs
        )
        markersize = 10 if marker != "*" else 12
        self.ax.plot(
            x[-1], y[-1],
            color=color,
            marker=marker,
            markersize=markersize,
            markeredgecolor="black",
            markeredgewidth=0.5,
            alpha=0.8,
            label=label,
            **kwargs
        )
        self.lines.append(line)
        self.last_ys.append(y[-1])
        self.last_xs.append(x[-1])
        self.colors.append(color)
        return line

    def get_fontsize_in_data_coords(self, ax, fontsize):
        """
        Given a Matplotlib Axes object and a font size,
        returns the equivalent font height in data coordinates.

        Parameters:
        - ax: Matplotlib Axes object
        - fontsize: Font size

        Returns:
        - text_height_data_coords: Font height in data coordinates
        """

        # Add temporary text to the plot to measure its height
        temp_text = ax.text(0, 0, "99.9", fontsize=fontsize)

        # Draw the canvas to make sure all elements are rendered
        plt.draw()

        # Get the bounding box of the text
        bbox = temp_text.get_window_extent()

        # Convert the bounding box to data coordinates
        bbox_data = bbox.transformed(ax.transData.inverted())

        # Calculate text height in data coordinates
        text_height_data_coords = bbox_data.ymax - bbox_data.ymin
        text_width_data_coords = bbox_data.xmax - bbox_data.xmin

        # Remove the temporary text
        temp_text.remove()

        # Redraw the canvas to reflect the removal
        plt.draw()

        return text_width_data_coords, text_height_data_coords

    def annotate(self, fontsize=20, **kwargs):
        def objective(x, original_positions, heights):
            # Objective function to minimize the total "cost" (distance from original position)
            cost = np.sum((x - original_positions)**2)

            # Calculate overlaps and add to cost if there are overlaps
            for i in range(len(x)):
                for j in range(i+1, len(x)):
                    overlap = heights[i] / 2 + heights[j] / 2 - np.abs(x[i] - x[j])
                    if overlap > 0:
                        cost += 10**6 * overlap ** 2  # Large penalty for overlaps

            return cost

        def correct_positions(original_positions, heights):
            if len(original_positions) != len(heights):
                return "Lengths of original_positions and widths should be the same."

            initial_guess = np.array(original_positions)
            eps = 1e-6
            random_offset = np.random.uniform(-eps, eps, len(initial_guess))
            initial_guess += random_offset
            result = minimize(objective, initial_guess, args=(original_positions, heights), method='L-BFGS-B')
            return result.x.tolist()

        text_width, text_height = self.get_fontsize_in_data_coords(self.ax, fontsize*1.2)

        # correct the positions
        corrected_positions = correct_positions(
            self.last_ys, [text_height]*len(self.last_ys)
        )
        pos_x = max(self.last_xs)
        max_pos_y = max(corrected_positions)
        # get current xlim

        # get the new width and height using the new xlim and ylim
        # text_width, text_height = self.get_fontsize_in_data_coords(self.ax, fontsize*1.2)


        for i, (last_y, pos_y) in enumerate(zip(self.last_ys, corrected_positions)):
            non_strippable_space = "\hspace{1cm}"
            self.ax.annotate(
                f"{last_y:>5.1f}".replace(" ", non_strippable_space),
                xy=(pos_x+text_width*.2, pos_y),
                fontsize=fontsize,
                color=self.colors[i],
                va="center",
                ha="left",
            )

            # Also put a marker at the end of the text
            markersize = 10 if self.markers[i] != "*" else 12
            marker_x = pos_x+text_width*1.3
            self.ax.plot(
                marker_x, pos_y,
                color=self.colors[i],
                marker=self.markers[i],
                markersize=markersize,
                markeredgecolor="black",
                markeredgewidth=0.5,
                alpha=0.8,
            )

        ylim = self.ax.get_ylim()
        xlim = self.ax.get_xlim()
        plt.xlim(xlim[0], max(xlim[1], pos_x+text_width*1.1))
        plt.ylim(ylim[0], max(max_pos_y+text_height*1.2, ylim[1]))

# PLOTTING UTILS
############################################################


############################################################
# PLOT FIGURES

def plot_delay_ablation():
    """
    Plot figure 3 from the paper

    Ablation of the delay factor
    """
    runs = defaultdict(list)
    api = wandb.Api()


    def get_run_metrics(run, recompute_online_acc=True):

        # FOR BACKWARD COMPATIBILITY
        x_metric = "timestep"
        if run.summary.get(x_metric) is None:
            x_metric = "timestep_step"
        assert run.summary.get(x_metric) is not None

        # Since the CLOC training resuming didn't work on "next_batch_acc1",
        # we need to recompute the Online Accuracy - in theory they should be the same
        if recompute_online_acc:
            y_metric = "next_batch_acc1_current"
            run_metrics = run.scan_history(
                keys=[x_metric, y_metric],
                page_size=40000,
            )
            run_metrics = list(tqdm.tqdm(
                run_metrics,
                desc=f"Recomputing Online Accuracy for {run.name}",
                total=run.lastHistoryStep//2,
            ))
            # run_metrics is a list of dicts with values for each metric at each step
            # sort the list by the x_metric
            run_metrics.sort(key=lambda x: x[x_metric])

            x = [row[x_metric] for row in run_metrics]
            y = [row[y_metric] for row in run_metrics]

            # cumsum the y values
            y = np.cumsum(y) / np.arange(1, len(y)+1)

        else:

            # FOR BACKWARD COMPATIBILITY
            y_metric = "next_batch_acc1"
            if run.summary.get(y_metric) is None:
                y_metric = "next_batch_acc1_step"
            assert run.summary.get(y_metric) is not None

            run_metrics = run.history(
                keys=[x_metric, y_metric],
                x_axis=x_metric,
                samples=10000,
                pandas=False,
            )
            # run_metrics is a list of dicts with values for each metric at each step
            # sort the list by the x_metric
            run_metrics.sort(key=lambda x: x[x_metric])

            x = [row[x_metric] for row in run_metrics]
            y = [row[y_metric] for row in run_metrics]
        return x, y

    naive_cloc_runs = []
    for naive_cloc_sweep_id in ["wk20q4tx", "k1ht1ugd"]:
        for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+naive_cloc_sweep_id).runs):
            if (
                run.config["online"]["num_supervised"] == 2 and
                run.config["online"]["supervision_source"] == "NR"
            ):
                naive_cloc_runs.append(run)
                print(run.name, run.id)

    naive_cglm_runs = []
    for naive_cglm_sweep_id in ["7tido66v"]:
        for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+naive_cglm_sweep_id).runs):
            if (
                run.config["online"]["num_supervised"] == 8 and
                run.config["online"]["supervision_source"] == "NR"
            ):
                naive_cglm_runs.append(run)

    naive_yearbook_runs = []
    for naive_yearbook_sweep_id in ["y3v64udp"]:
        for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+naive_yearbook_sweep_id).runs):
            if (
                run.config["online.num_supervised"] == 16 and
                run.config["online.supervision_source"] == "NR"
            ):
                naive_yearbook_runs.append(run)


    naive_fmow_runs = []
    for naive_fmow_sweep_id in ["jiow5twf"]:
        for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+naive_fmow_sweep_id).runs):
            if (
                run.config["online.num_supervised"] == 16 and
                run.config["online.supervision_source"] == "NR"
            ):
                naive_fmow_runs.append(run)

    # runs = {
    #     "Continual Localisation (CLOC)": naive_cloc_runs,
    #     "Continual Google Landmarks (CGLM)": naive_cglm_runs,
    #     "Functional Map of the World (FMoW)": naive_fmow_runs,
    #     "Yearbook": naive_yearbook_runs,
    # }

    runs = {
        "CLOC": naive_cloc_runs,
        "CGLM": naive_cglm_runs,
        "FMoW": naive_fmow_runs,
        "Yearbook": naive_yearbook_runs,
    }
    delays = [0, 10, 50, 100]
    colors = cm['Oranges'](np.linspace(0.4, 1, len(delays)))

    block_size = 4.5
    legend_size = .4
    num_rows = 1
    num_cols = 4
    plt.figure(
        figsize=(
            block_size*num_cols,
            block_size*num_rows+legend_size
        )
    )
    for i, (dataset, runs) in enumerate(runs.items()):
        runs.sort(key=lambda x: x.config["online.delay"])


        ax = plt.subplot(num_rows, num_cols, i+1)
        plotter = Plotter(ax)
        for run in tqdm.tqdm(runs):
            recompute_online_acc = False
            x, y = get_run_metrics(run, recompute_online_acc=recompute_online_acc)
            delay = run.config["online.delay"]
            color = colors[delays.index(delay)]
            label = f"$d={delay}$"
            plotter.plot(x, y, label=label, color=color, lw=3)
        plotter.annotate()
        set_default_settings(ax)
        plt.xlabel("Time step")
        if i == 0:
            plt.ylabel("Online Accuracy")
        plt.title(dataset)

    # plot the legend
    ax = plt.subplot(num_rows, num_cols, 1)
    plt.figlegend(
        *ax.get_legend_handles_labels(),
        loc='lower center',
        ncol=len(delays),
        fontsize=22
    )
    # Take the legend size into account
    # plt.tight_layout(rect=[0, 0, 1.0, 1.0-legend_size/(block_size*num_rows+legend_size)])
    plt.tight_layout(rect=[0, legend_size/(block_size*num_rows+legend_size), 1.0, 1.0])
    # plt.subplots_adjust(bottom=.2)

    plt.savefig(f"fig-delay-ablation.pdf")


    # colors = colors = cm.viridis(np.linspace(0, 1, len(runs)))
    # Oranges
    # colors = cm['Oranges'](np.linspace(0.4, 1, len(runs)))

    # sort runs by delay


    # plt.figure(figsize=(5, 5))
    # plotter = Plotter()
    # for i, run in enumerate(tqdm.tqdm(runs)):
    #     x, y = get_run_metrics(run)
    #     label = f'$d$={run.config["online.delay"]}'
    #     plotter.plot(x, y, label=label, color=colors[~i], lw=3)
    # plotter.annotate(fontsize=22)

    # set_default_settings()
    # plt.xlabel("Time step")
    # plt.ylabel("Online Accuracy")
    # title = "Continual Localization (CLOC)" if dataset == "CLOC" else "Continual Google Landmarks (CGLM)"
    # plt.title(title, fontsize=20)
    # plt.legend(framealpha=0.5, fontsize=17.5, loc="lower right")
    # plt.tight_layout()
    # plt.savefig(f"fig-delay-ablation-{dataset}.pdf")
    # plt.savefig(f"fig-delay-ablation-{dataset}.svg", format="svg")
    # plt.close()


def plot_fig_compute():
    """
    Plot fig:compute from the paper

    Ablation of the Compute factor
    """

    for dataset in ["CLOC", "CGLM"]:
        api = wandb.Api()

        plt.figure(figsize=(3*4, 5))
        # sweep_id = "[author1]/onlineCL-cs1/op6hwx8n"
        if dataset == "CLOC":
            sweep_id = "[author1]/onlineCL-cs1/w8kv7s1d"
        elif dataset == "CGLM":
            # sweep_id = "[author1]/onlineCL-CGLM/4pefs3rt" # OLD
            sweep_id = "[author1]/onlineCL-cs1/74737bfd" # NEW

        iter_str = "online/batch_repeat" if dataset == "CLOC" else "online.num_supervised"
        runs = list(api.sweep(sweep_id).runs)
        # sweep_ids = ["[author1]/onlineCL-cs1/2wf30m5i", "[author1]/onlineCL-cs1/bpb21z4w", "[author1]/onlineCL-cs1/6mj3k6tv", "[author1]/onlineCL-cs1/z12dp3iz"]
        # runs = []
        # for sweep_id in sweep_ids:
            # sweep = api.sweep(sweep_id)
            # runs += list(sweep.runs)


        subplots = ["a", "b", "c"]
        if dataset == "CLOC":
            # Cs = [1, 2, 4, 8]
            Cs = [1, 8]
        elif dataset == "CGLM":
            # Cs = [4, 8, 16, 32]
            Cs = [4, 32]
        delays = {"a": 10, "b": 50, "c": 100}

        # colors = cm['Oranges'](np.linspace(0.6, 0.4, len(Cs)))
        # colors = cm['Oranges']([0.7, 0.4, 0.4, 0.4])
        # baseline_color = cm['Oranges'](1.0)
        blue_color = "#7eb0d5"
        orange_color = cm['Oranges'](0.5)
        # lws = np.linspace(3, 2, len(Cs))
        lws = [3, 3]
        # lws = [3, 1, 1, 1]
        # linetypes = [":", "-.", "--", "-"]
        # linetypes = ["-", "-", "-", "-"]
        # linetypes = ["-", "--"]

        for plot_idx, subplot in enumerate(subplots):
            ax = plt.subplot(1, 3, plot_idx+1)
            delay = delays[subplot]
            baseline_low_run = [run for run in runs if run.config["online.delay"] == 0 and run.config[iter_str] == Cs[0]][0]
            baseline_high_run = [run for run in runs if run.config["online.delay"] == 0 and run.config[iter_str] == Cs[-1]][0]
            d_runs = [run for run in runs if run.config["online.delay"] == delay and run.config[iter_str] in Cs]

            # sort runs by repeat
            d_runs.sort(key=lambda x: x.config[iter_str], reverse=True)


            plotter = Plotter()
            x, y = get_run_metrics(baseline_high_run)
            label = f'{C}={baseline_high_run.config[iter_str]} w/o delay'
            plotter.plot(x, y, color=orange_color, label=label, linestyle="--", lw=lws[0])

            x, y = get_run_metrics(baseline_low_run)
            label = f'{C}={baseline_low_run.config[iter_str]} w/o delay'
            plotter.plot(x, y, color=blue_color, label=label, linestyle="--", lw=lws[0])

            for i, run in enumerate(tqdm.tqdm(d_runs)):
                x, y = get_run_metrics(run)
                label = f'{C}={run.config[iter_str]}'
                color = blue_color if run.config[iter_str] == Cs[0] else orange_color
                plotter.plot(x, y, color=color, label=label, linestyle="-", lw=lws[i])
            # plt.ylim(0, 30)
            plotter.annotate()
            # plt.ylim(0, 30)
            set_default_settings()


            # plt.xlim(-3000, 33000)
            plt.xlabel("Time step")
            # only show the y label on the leftmost plot
            if plot_idx == 0:
                plt.ylabel("Online Accuracy")

            plt.title(f"{dataset.upper()} ($d$={delay})")

        # plot the legend
        ax = plt.subplot(1, 3, 1)
        plt.figlegend(
            *ax.get_legend_handles_labels(), loc='lower center', ncol=6,
            fontsize=22
        )

        plt.tight_layout(rect=[0, 0.075, 1.0, 1.0])
        plt.savefig(f"fig-compute-{dataset}.pdf")
        plt.savefig(f"fig-compute-{dataset}.svg", format="svg")
        plt.close()


def plot_unsupervised():
    """
    2x3 plot featuring the 3 categories of unsupervised methods: SSL, TTA, and OURS
    on 2 different datasets: CLOC and CGLM

    on each dataset, plot the 3 methods with 3 different delays (d=10, 50, 100)
    """

    def get_run_metrics(run, recompute_online_acc=True):

        # FOR BACKWARD COMPATIBILITY
        x_metric = "timestep"
        if run.summary.get(x_metric) is None:
            x_metric = "timestep_step"
        assert run.summary.get(x_metric) is not None

        # Since the CLOC training resuming didn't work on "next_batch_acc1",
        # we need to recompute the Online Accuracy - in theory they should be the same
        if recompute_online_acc:
            y_metric = "next_batch_acc1_current"
            run_metrics = run.scan_history(
                keys=[x_metric, y_metric],
                page_size=40000,
            )
            run_metrics = list(tqdm.tqdm(
                run_metrics,
                desc=f"Recomputing Online Accuracy for {run.name}",
                total=run.lastHistoryStep//2,
            ))
            # run_metrics is a list of dicts with values for each metric at each step
            # sort the list by the x_metric
            run_metrics.sort(key=lambda x: x[x_metric])

            x = [row[x_metric] for row in run_metrics]
            y = [row[y_metric] for row in run_metrics]

            # cumsum the y values
            y = np.cumsum(y) / np.arange(1, len(y)+1)

        else:

            # FOR BACKWARD COMPATIBILITY
            y_metric = "next_batch_acc1"
            if run.summary.get(y_metric) is None:
                y_metric = "next_batch_acc1_step"
            assert run.summary.get(y_metric) is not None

            run_metrics = run.history(
                keys=[x_metric, y_metric],
                x_axis=x_metric,
                samples=10000,
                pandas=False,
            )
            # run_metrics is a list of dicts with values for each metric at each step
            # sort the list by the x_metric
            run_metrics.sort(key=lambda x: x[x_metric])

            x = [row[x_metric] for row in run_metrics]
            y = [row[y_metric] for row in run_metrics]
        return x, y

    # if not os.path.exists("runs.pt"):
    api = wandb.Api()

    #############################
    # GATHER THE RUNS
    #############################
    #############################
    # NAIVE RUNS
    #############################
    # cloc_naive_sweep_id = "w8kv7s1d" # 40K memory
    # cloc_naive_sweep_id = "wk20q4tx" # 2^19 memory
    # cloc_naive_runs = list(api.sweep("[author1]/onlineCL-cs1/"+cloc_naive_sweep_id).runs)
    # # filter batch repeat = 2
    # cloc_naive_runs = [run for run in cloc_naive_runs if run.config["online"]["num_supervised"] == 2]

    naive_cloc_runs = []
    for naive_cloc_sweep_id in ["wk20q4tx", "k1ht1ugd"]:
        for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+naive_cloc_sweep_id).runs):
            if (
                run.config["online"]["num_supervised"] == 2 and
                run.config["online"]["supervision_source"] == "NR"
            ):
                naive_cloc_runs.append(run)
                print(run.name, run.id)

    naive_cglm_runs = []
    for naive_cglm_sweep_id in ["7tido66v"]:
        for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+naive_cglm_sweep_id).runs):
            if (
                run.config["online"]["num_supervised"] == 8 and
                run.config["online"]["supervision_source"] == "NR"
            ):
                naive_cglm_runs.append(run)

    naive_yearbook_runs = []
    for naive_yearbook_sweep_id in ["y3v64udp"]:
        for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+naive_yearbook_sweep_id).runs):
            if (
                run.config["online.num_supervised"] == 16 and
                run.config["online.supervision_source"] == "NR"
            ):
                naive_yearbook_runs.append(run)


    naive_fmow_runs = []
    for naive_fmow_sweep_id in ["jiow5twf"]:
        for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+naive_fmow_sweep_id).runs):
            if (
                run.config["online.num_supervised"] == 16 and
                run.config["online.supervision_source"] == "NR"
            ):
                naive_fmow_runs.append(run)

    #############################
    # SSL RUNS
    #############################
    # filter num_supervised = 1
    ssl_cloc_runs = []
    # for ssl_cloc_sweep_id in ["iz12g5er"]: # 40K memory
    for ssl_cloc_sweep_id in ["c8fw690u"]: # 2^19 memory
        for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+ssl_cloc_sweep_id).runs):
            if (
                run.config["online"]["num_supervised"] == 1 and
                run.config["online"]["num_unsupervised"] == 1
            ):
                ssl_cloc_runs.append(run)
                print(run.name, run.id)

    ssl_cglm_sweep_id = "3b703oc4"
    # the baseline does 4 supervised and 0 unsupervised
    # so we filter for 2 supervised and 2 unsupervised
    ssl_cglm_runs = []
    for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+ssl_cglm_sweep_id).runs):
        if (
            run.config["online"]["num_supervised"] == 4 and
            run.config["online"]["num_unsupervised"] == 4
        ):
            ssl_cglm_runs.append(run)


    ssl_yearbook_sweep_id = "xp2q91jl"
    ssl_yearbook_runs = []
    for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+ssl_yearbook_sweep_id).runs):
        if (
            run.config["online.num_supervised"] == 8 and
            run.config["online.num_unsupervised"] == 8 and
            run.config["category"] in ["SSL"] and
            run.config["data/dataset"] in ["yearbook"]
        ):
            ssl_yearbook_runs.append(run)

    ssl_fmow_sweep_id = "xp2q91jl"
    ssl_fmow_runs = []
    for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+ssl_fmow_sweep_id).runs):
        if (
            run.config["online.num_supervised"] == 8 and
            run.config["online.num_unsupervised"] == 8 and
            run.config["category"] in ["SSL"] and
            run.config["data.dataset"] in ["fmow"]
        ):
            ssl_fmow_runs.append(run)

    #############################
    # TTA RUNS
    #############################
    # filter num_supervised = 1
    tta_cloc_runs = []
    # for tta_cloc_sweep_id in ["k28p9bly", "kxe7t9ri"]: # 40K memory
    for tta_cloc_sweep_id in ["4f81t0jf", "y00h1828"]: # 2^19 memory
        for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+tta_cloc_sweep_id).runs):
            if (
                run.config["online"]["num_supervised"] == 1 and
                run.config["online"]["num_unsupervised"] == 1
            ):
                tta_cloc_runs.append(run)
                print(run.name, run.id)

    tta_cglm_sweep_id = "98pnh3hq"
    # filter num_supervised = 1
    tta_cglm_runs = []
    for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+tta_cglm_sweep_id).runs):
        if (
            run.config["online"]["num_supervised"] == 7 and
            run.config["online"]["num_unsupervised"] == 1
        ):
            tta_cglm_runs.append(run)


    tta_yearbook_sweep_id = "ps54oew5"
    tta_yearbook_runs = []
    for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+tta_yearbook_sweep_id).runs):
        if (
            run.config["online.num_supervised"] == 15 and
            run.config["online.num_unsupervised"] == 1 and
            run.config["category"] in ["TTA"] and
            run.config["data.dataset"] in ["yearbook"]
        ):
            tta_yearbook_runs.append(run)


    tta_fmow_sweep_id = "ps54oew5"
    tta_fmow_runs = []
    for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+tta_fmow_sweep_id).runs):
        if (
            run.config["online.num_supervised"] == 15 and
            run.config["online.num_unsupervised"] == 1 and
            run.config["category"] in ["TTA"] and
            run.config["data.dataset"] in ["fmow"]
        ):
            tta_fmow_runs.append(run)


    #############################
    # PSEUDO RUNS
    #############################
    pseudo_cloc_runs = []
    for pseudo_cloc_sweep_id in ["gh4ex456"]:
        for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+pseudo_cloc_sweep_id).runs):
            if (
                run.config["online/num_supervised"] == 1 and
                run.config["online/num_unsupervised"] == 1
            ):
                pseudo_cloc_runs.append(run)

    pseudo_cglm_runs = []
    for pseudo_cglm_sweep_id in ["qqfgkbkb"]:
        for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+pseudo_cglm_sweep_id).runs):
            if (
                run.config["online/num_supervised"] == 4 and
                run.config["online/num_unsupervised"] == 4
            ):
                pseudo_cglm_runs.append(run)

    pseudo_yearbook_runs = []
    for pseudo_yearbook_sweep_id in ["2mpo11s8"]:
        for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+pseudo_yearbook_sweep_id).runs):
            if (
                run.config["online.num_supervised"] == 8 and
                run.config["online.num_unsupervised"] == 8 and
                run.config["category"] in ["pseudo"] and
                run.config["data.dataset"] in ["yearbook"]
            ):
                pseudo_yearbook_runs.append(run)

    pseudo_fmow_runs = []
    for pseudo_fmow_sweep_id in ["01kqdps8"]:
        for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+pseudo_fmow_sweep_id).runs):
            if (
                run.config["online.num_supervised"] == 8 and
                run.config["online.num_unsupervised"] == 8 and
                run.config["category"] in ["pseudo"] and
                run.config["data.dataset"] in ["fmow"]
            ):
                pseudo_fmow_runs.append(run)


    #############################
    # OUR RUNS
    #############################
    # ours_cloc_sweep_id = "rw7ck577" # Two sources
    # ours_cloc_sweep_id = "g5l52r6h" # Three sources
    # filter num_supervised = 2
    ours_cloc_runs = []
    for ours_cloc_sweep_id in ["k903jftm"]:
        for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+ours_cloc_sweep_id).runs):
            if (
                run.config["online"]["num_supervised"] == 2 and
                run.config["online"]["num_unsupervised"] == 1 and
                # run.config["online"]["supervision_source"] in ["NR"]
                # run.config["online"]["supervision_source"] in ["NW"]
                # run.config["online"]["supervision_source"] in ["WR", "RR"]
                # run.config["online"]["supervision_source"] in ["WRR", "WWR", "RRR", "NWR"]
                set(run.config["online"]["supervision_source"]) in [set("NWR")] and
                run.config["method_kwargs"]["alternating_sources"] == ["NR", "WR"]
            ):
                ours_cloc_runs.append(run)
                print(run.name, run.id)



    # ours_cglm_sweep_id = "ls0c8o4x"
    # ours_cglm_sweep_id = "ckh5920v"
    # ours_cglm_sweep_id = "qhu45cxz"
    ours_cglm_sweep_id = "oag65hxw"

    # ours_cglm_sweep_id = " "
    # filter num_supervised = 4
    ours_cglm_runs = []
    for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+ours_cglm_sweep_id).runs):
        if (
            run.config["online"]["num_supervised"] == 8 and
            run.config["online"]["num_unsupervised"] == 1 and
            run.config["online"]["supervision_source"] in ["WR"]
            # run.config["online"]["supervision_source"] in ["WR", "RR"]
            # run.config["online"]["supervision_source"] in ["WRR", "WWR", "RRR", "NWR"]
        ):
            ours_cglm_runs.append(run)

    ours_yearbook_runs = []
    for ours_yearbook_sweep_id in ["2q2t2uh2"]:
        for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+ours_yearbook_sweep_id).runs):
            if (
                run.config["online/num_supervised"] == 16 and
                run.config["online/num_unsupervised"] == 1 and
                run.config["online/supervision_source"] in ["RWN"] and
                run.config["method_kwargs/alternating_sources"] == ["NR", "WR"] and
                run.config["category"] in ["ours"] and
                run.config["data/dataset"] in ["yearbook"]
            ):
                ours_yearbook_runs.append(run)

    ours_fmow_runs = []
    for ours_fmow_sweep_id in ["bw2bmip4"]:
        for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+ours_fmow_sweep_id).runs):
            if (
                run.config["online/num_supervised"] == 16 and
                run.config["online/num_unsupervised"] == 1 and
                run.config["online/supervision_source"] in ["RWN"] and
                run.config["method_kwargs/alternating_sources"] == ["NR", "WR"] and
                run.config["category"] in ["ours"] and
                run.config["data/dataset"] in ["fmow"]
            ):
                ours_fmow_runs.append(run)

    #############################
    # COLLECT THE RUNS
    #############################
    runs = {
        "cloc": {
            "naive": naive_cloc_runs,
            "ssl": ssl_cloc_runs,
            "tta": tta_cloc_runs,
            "pseudo": pseudo_cloc_runs,
            "ours": ours_cloc_runs,
        },
        "cglm": {
            "naive": naive_cglm_runs,
            "ssl": ssl_cglm_runs,
            "tta": tta_cglm_runs,
            "pseudo": pseudo_cglm_runs,
            "ours": ours_cglm_runs,
        },
        "yearbook": {
            "naive": naive_yearbook_runs,
            "ssl": ssl_yearbook_runs,
            "tta": tta_yearbook_runs,
            "pseudo": pseudo_yearbook_runs,
            "ours": ours_yearbook_runs,
        },
        "fmow": {
            "naive": naive_fmow_runs,
            "ssl": ssl_fmow_runs,
            "tta": tta_fmow_runs,
            "pseudo": pseudo_fmow_runs,
            "ours": ours_fmow_runs,
        }
    }

    # define helper functions for extracting config values
    def get_delay(run):
        val1 = run.config.get("online/delay", None)
        if val1 is not None:
            return val1

        val2 = run.config.get("online.delay", None)
        if val2 is not None:
            return val2

        val3 = run.config.get("online", {}).get("delay", None)
        if val3 is not None:
            return val3

    def get_dataset(run):
        val1 = run.config.get("data/dataset", None)
        if val1 is not None:
            return val1
        val2 = run.config.get("data.dataset", None)
        if val2 is not None:
            return val2
        val3 = run.config.get("data", {}).get("dataset", None)
        if val3 is not None:
            return val3

    #############################
    # ARRANGE THE PLOTS
    #############################
    # make subplots


    # datasets = ["cloc", "cglm", "yearbook", "fmow"]
    datasets = ["cloc", "cglm", "fmow", "yearbook"]
    # datasets = ["cloc"]
    # datasets = ["cglm"]
    delays = [10, 50, 100]
    methods = ["naive-wod", "naive", "ours", "ssl", "pseudo", "tta"]
    # methods = ["naive-wod", "naive"]
    # methods = ["naive-wod", "naive", "tta"]
    # methods = ["naive-wod", "naive", "tta", "ssl"]


    colors = {
        # "ssl": "C0",
        # "tta": "C1",
        # "ours": "C2"

        # COLORBREWER
        # "ssl": "#a6cee3",
        # "tta": "#b2df8a",
        # "ours": "#fb9a99"

        # Spring pastels
        "naive-wod": "#ffb55a",
        "naive": "#ffb55a",
        "ssl": "#7eb0d5",
        "tta": "#bd7ebe",
        "pseudo": "#82c596",
        "ours": "#fd7f6f"
    }

    linestyles = {
        "naive-wod": "--",
        "naive": "-",
        "ssl": "-",
        "tta": "-",
        "pseudo": "-",
        "ours": "-"
    }

    labels = {
        "naive-wod": "Naïve without delay",
        "naive": "Naïve",
        "ssl": "S4L",
        "tta": "TTA",
        "pseudo": "Pseudo-Label",
        "ours": "IWMS",
    }

    C_vals = {
        "cloc": 2,
        "cglm": 8,
        "yearbook": 16,
        "fmow": 16,
    }
    dataset_names = {
        "cloc": "CLOC",
        "cglm": "CGLM",
        "yearbook": "Yearbook",
        "fmow": "FMoW",
    }

    # ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#8bd3c7"]

    # linestyles = [
    #     "-",
    #     "--",
    #     "-.",
    #     ":"
    # ]


    block_size = 4.5
    num_rows = len(delays)
    num_cols = len(datasets)
    plt.figure(figsize=(num_cols*block_size, num_rows*block_size+0.5))
    for i, delay in enumerate(delays):
        for j, dataset in enumerate(datasets):
            print(f"Plotting {dataset} {delay}")
            ax = plt.subplot(num_rows, num_cols, i*num_cols+j+1)
            plotter = Plotter(ax=ax)
            for method in methods:
                # method_runs = runs[dataset][method]
                if method == "naive-wod":
                    method_runs = runs[dataset]["naive"]
                else:
                    method_runs = runs[dataset][method]

                if method == "naive-wod":
                    method_runs = [run for run in method_runs if get_delay(run) == 0]
                else:
                    method_runs = [run for run in method_runs if get_delay(run) == delay]


                print(f"Found {len(method_runs)} runs for {method} {dataset} {delay}")
                if len(method_runs) == 0:
                    continue

                # recompute_online_acc = method in ["ours", "ssl"] and dataset == "cloc"
                # recompute_online_acc = True
                recompute_online_acc = False
                method_xys = {run: get_run_metrics(run, recompute_online_acc) for run in method_runs}

                # sort by the y value at the last step
                method_runs.sort(key=lambda x: method_xys[x][1][-1], reverse=True)

                method_runs = method_runs[:1]

                # plot the best run
                for k, run in enumerate(method_runs):
                    label = labels[method]

                    # alpha = 1. if k == 0 else 0.5
                    # linestyle = linestyles[k]
                    x, y = method_xys[run]
                    plotter.plot(
                        x, y,
                        label=label,
                        color=colors[method],
                        lw=3,
                        linestyle=linestyles[method],
                    )

                set_default_settings(ax=ax)
                # ylim = ax.get_ylim()
                # if dataset == "cloc":
                #     plt.ylim(ylim[0], 15)
                # else:
                #     plt.ylim(ylim[0], 23)

            plotter.annotate(fontsize=22)

            # only plot it for the bottom row
            if i == num_rows-1:
                plt.xlabel("Time step")

            # only plot it for the left column
            if j == 0:
                # plt.ylabel("Online Accuracy")
                plt.ylabel(f"Online Accuracy ($d={delay}$)")


            C_val = C_vals[dataset]
            # plt.title(f"~~{dataset.upper()}"+r"\\"+f"$d$={delay}, {C}={C_val}")
            # plt.title(f"~~{dataset.upper()} "+f"($d$={delay}, {C}={C_val})")
            if i == 0:
                plt.title(f"{dataset_names[dataset]} ({C}={C_val})")

    # plot the legend
    ax = plt.subplot(num_rows, num_cols, 1)
    plt.figlegend(*ax.get_legend_handles_labels(), loc='lower center', ncol=6)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.125, hspace=0.1)

    plt.savefig(f"fig-unsupervised.pdf")
    plt.savefig(f"fig-unsupervised.svg", format="svg")
    plt.close()


def plot_fig_sampling_strategy():
    """
    Plot the various sampling strategies for ablation on CGLM

    (here by strategy we mean the supervision source)
    """
    for delay in [10, 100]:
        api = wandb.Api()
        runs = []
        for cglm_sweep_id in ["407pdrs4", "hunyluhz"]:
            for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+cglm_sweep_id).runs):
                if "online" not in run.config:
                    continue
                if run.config["online"]["supervision_source"] in ["WR", "NR", "RR"] and cglm_sweep_id != "407pdrs4":
                    continue
                if run.config["online"]["supervision_source"] in ["WW", "NW", "NN"] and cglm_sweep_id != "hunyluhz":
                    continue

                if (
                    run.config["online"]["num_supervised"] == 8 and
                    run.config["online"]["delay"] == delay
                    # run.config["online"]["sup_buffer_size"] == 40000
                    # run.config["online"]["sup_buffer_size"] == 2**20
                ):
                    runs.append(run)

        if len(runs) == 0:
            raise ValueError("No runs found")

        source_run = {
            # run.config["method_kwargs"]["selection_function"]: run for run in runs
            run.config["online"]["supervision_source"]: run for run in runs
        }

        # sort by last value of y
        run_xys = {run: get_run_metrics(run) for run in runs}
        source_run = {
            k: v for k, v in sorted(
                source_run.items(), key=lambda item: run_xys[item[1]][1][-1], reverse=True
            )
        }

        plt.figure(figsize=(5, 5.5))
        """
        #a6cee3
        #1f78b4
        #b2df8a
        #33a02c
        #fb9a99
        #e31a1c
        """

        colors = ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#8bd3c7"]

        max_x = 0
        last_ys = []
        plotter = Plotter()
        for i, (source, run) in enumerate(source_run.items()):
            x, y = run_xys[run]
            plotter.plot(
                x, y,
                label=source,
                color=colors[i],
                lw=3,
            )

            last_x = x[-1]
            last_y = y[-1]
            last_ys.append(last_y)
            max_x = max(max_x, last_x)
        # ylim = plt.ylim()
        # plt.ylim(ylim[0], 30.2)
        plotter.annotate()
        set_default_settings()
        plt.xlabel("Time step")
        plt.ylabel("Online Accuracy")

        plt.legend(framealpha=0.5, loc="lower right", fontsize=15)
        # plt.legend(framealpha=0.5, loc="lower center", fontsize=15, ncol=2)
        # plt.legend(framealpha=0.5, loc="upper center", fontsize=15, ncol=2)

        plt.title(f"Sampling Strategy ($d$={delay})", fontsize=27)
        plt.tight_layout()
        plt.savefig(f"fig-sampling-strategy-d{delay}.pdf")
        plt.savefig(f"fig-sampling-strategy-d{delay}.svg", format="svg")
        plt.close()



def plot_fig_memory_size():
    """
    Plot the memory size of the best performing runs for our method on CGLM
    for ablation studies
    """

    api = wandb.Api()
    runs = []
    for delay in [10, 100]:
        for cglm_sweep_id in ["dajgyba4", "m8065g0c"]:
            for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+cglm_sweep_id).runs):
                if "online" not in run.config:
                    continue
                if (
                    run.config["online"]["num_supervised"] == 8 and
                    run.config["online"]["num_unsupervised"] == 1 and
                    run.config["online"]["delay"] == delay and
                    run.config["online"]["supervision_source"] in ["WR"] and
                    10000 <= run.config["online"]["sup_buffer_size"] <= 80000
                ):
                    runs.append(run)

        if len(runs) == 0:
            raise ValueError("No runs found")

        # remove duplicates by memory size
        memsize_run = {
            run.config["online"]["sup_buffer_size"]: run for run in runs
        }

        # memsize_run["$2^{15.3}$"] = memsize_run.pop(40000)
        # memsize_run["$2^{15}$"] = memsize_run.pop(40000)
        # memsize_run["$2^{16}$"] = memsize_run.pop(65536)
        # memsize_run["$2^{17}$"] = memsize_run.pop(131072)
        # memsize_run["$2^{18}$"] = memsize_run.pop(262144)

        new_memsize_run = {}
        for k, v in memsize_run.items():
            formatted = thousands_formatter(k, None)
            new_memsize_run[formatted] = v
        memsize_run = new_memsize_run



        # sort by last value of y
        run_xys = {run: get_run_metrics(run) for run in runs}
        memsize_run = {
            k: v for k, v in sorted(
                memsize_run.items(), key=lambda item: run_xys[item[1]][1][-1], reverse=True
            )
        }

        plt.figure()

        # colors = cm["Blues"](np.linspace(0.4, 1.0, len(memsize_run)))
        # colors = ["#68a1c8", "#7eb0d5", "#92c0e0", "#bae0f6"]
        # colors = ["550000", "801515", "AA3939", "D46A6A"]
        # colors = [0E8649, 24A362, 3FB378, 63CB96]
        colors = ["#0e8649", "#24a362", "#3fb378", "#63cb96"][::-1]

        plotter = Plotter()
        for i, (memsize, run) in enumerate(memsize_run.items()):
            x, y = run_xys[run]
            plotter.plot(
                x, y,
                label=memsize,
                color=colors[i],
                lw=3,
            )
        # ylim = plt.ylim()
        # plt.ylim(ylim[0], 20.2)
        plotter.annotate()

        set_default_settings()
        plt.xlabel("Time step")
        plt.ylabel("Online Accuracy")

        plt.legend(framealpha=0.5, loc="lower right", fontsize=18)
        # plt.legend(framealpha=0.5, loc="upper left", fontsize=18)

        plt.title(f"Memory Size ($d$={delay})")
        plt.tight_layout()
        plt.savefig(f"fig-memory-size-d{delay}.pdf")
        plt.close()


def plot_fig_increased_compute():
    """
    Plot the various sampling strategies for ablation on CGLM

    (here by strategy we mean the supervision source)
    """
    for delay in [10, 100]:
        api = wandb.Api()
        runs = []
        for cglm_sweep_id in ["co7dg8y3"]:
            for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+cglm_sweep_id).runs):
                if (
                    run.config["online"]["num_unsupervised"] == 1 and
                    run.config["online"]["delay"] == delay
                    # run.config["online"]["delay"] == 50 and
                    # run.config["method_kwargs"]["queue_size"] == 40000
                ):
                    runs.append(run)

        if len(runs) == 0:
            raise ValueError("No runs found")

        compute_run = {
            run.config["online"]["num_supervised"]: run for run in runs
        }

        # sort by compute
        run_xys = {run: get_run_metrics(run) for run in runs}
        compute_run = {k: v for k, v in sorted(compute_run.items(), key=lambda item: item[0], reverse=True)}

        plt.figure(figsize=(5, 5.5))
        # colors = cm['Greens'](np.linspace(0.4, 1.0, len(compute_run)))
        # colors = ["#86b03d", "#b2e061", "#c6ec7b", "#d0f288"][::-1]
        # colors = ["#C07315", "#E99634", "#FFB55A", "#FFC57D"]
        colors = ["#c07315", "#e99634", "#ffb55a", "#ffc57d"][::-1]
        plotter = Plotter()
        for i, (compute, run) in enumerate(compute_run.items()):
            x, y = run_xys[run]
            plotter.plot(
                x, y,
                label=compute,
                color=colors[i],
                lw=3,
            )
        # ylim = plt.ylim()
        # plt.ylim(ylim[0], 30.2)
        plotter.annotate()
        set_default_settings()
        plt.xlabel("Time step")
        plt.ylabel("Online Accuracy")
        plt.legend(framealpha=0.5, loc="lower right", fontsize=18.5)
        plt.title(f"Compute Budget {C} ($d$={delay})", fontsize=27)
        plt.tight_layout()
        plt.savefig(f"fig-increased-compute-d{delay}.pdf")
        plt.savefig(f"fig-increased-compute-d{delay}.svg", format="svg")
        plt.close()


def plot_fig_temperature():
    """
    Plot the various temperature values for ablation on CGLM
    """

    for delay in [10, 100]:
        api = wandb.Api()
        runs = []
        for cglm_sweep_id in ["3yo8uhrx"]:
            for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+cglm_sweep_id).runs):
                if (
                    run.config["online"]["num_supervised"] == 8 and
                    run.config["online"]["num_unsupervised"] == 1 and
                    run.config["online"]["delay"] == 100 and
                    run.config["online"]["supervision_source"] in ["WR"]
                    # run.config["method_kwargs"]["base_temperature"] == run.config["method_kwargs"]["final_temperature"]
                ):
                    runs.append(run)

        if len(runs) == 0:
            raise ValueError("No runs found")

        temperature_run = {
            run.config["method_kwargs"]["base_temperature"]: run for run in runs
        }

        # sort by last value of y
        run_xys = {run: get_run_metrics(run) for run in runs}
        temperature_run = {
            k: v for k, v in sorted(
                temperature_run.items(), key=lambda item: run_xys[item[1]][1][-1], reverse=True
            )
        }

        plt.figure()
        colors = cm['Purples'](np.linspace(0.4, 1.0, len(temperature_run)))
        plotter = Plotter()
        for i, (temperature, run) in enumerate(temperature_run.items()):
            x, y = run_xys[run]
            plotter.plot(
                x, y,
                label=temperature,
                color=colors[i],
                lw=3,
            )

        plotter.annotate()
        set_default_settings()
        plt.xlabel("Time step")
        plt.ylabel("Online Accuracy")
        plt.legend(framealpha=0.5, loc="lower right", fontsize=18)
        plt.title(f"Temperature ($d$={delay})")
        plt.tight_layout()
        plt.savefig(f"fig-temperature-d{delay}.pdf")


def plot_ssl_breakdown():
    """
    Plot figure 4 from the paper

    Detailed SSL ablation at d=10 C=1
    """

    # batch_repeat = {"a": 1, "b": 5, "c": 10, "d": 20}[subfigure]

    x_metric = "trainer/global_step"
    y_metric = "next_batch_acc1_step"
    def get_run_metrics(run):
        # locally override because there are some runs with faulty X axis values
        run_metrics = run.history(
            keys=[x_metric, y_metric],
            x_axis=x_metric,
            samples=10000,
            pandas=False,
        )
        # run_metrics is a list of dicts with values for each metric at each step
        # sort the list by the x_metric
        run_metrics.sort(key=lambda x: x[x_metric])

        # find the index of the closest step to 30000
        idx = np.argmin(np.abs(np.array([row[x_metric] for row in run_metrics]) - 30000))
        run_metrics = run_metrics[:idx+1]

        x = [row[x_metric] for row in run_metrics]
        y = [row[y_metric] for row in run_metrics]

        return x, y

    for batch_repeat in [1, 5, 10, 20]:
        api = wandb.Api()

        print("Fetching runs")
        sweep_ids = [
            "op6hwx8n", # Naive
            "t7zo3eft", # Old Naive
            "gt2xztjk", # TTA
            "q4hhzx04", # SSL
            "arbac2as", # MOCOv3
            "dp9ybefy", # SAR
        ]

        runs = []
        for sweep_id in sweep_ids:
            sweep = api.sweep("[author1]/onlineCL-cs1/"+sweep_id)
            runs.extend(list(sweep.runs))


        # Baseline d=10
        baseline_d10_run = [run for run in runs if run.config["online/delay"] == 10 and run.config["online/batch_repeat"] == batch_repeat and run.config["category"].lower() == "naive"][0]

        ssl_d10_runs = [run for run in runs if run.config["online/delay"] == 10 and run.config["online/batch_repeat"] == batch_repeat and run.config["category"].upper() == "SSL"]



        # filter ssl runs by method, such that we only have one run per method

        filtered_ssl_d10_runs = []
        duplicate_methods = set()
        for run in ssl_d10_runs:
            if run.config["method"] not in duplicate_methods:
                filtered_ssl_d10_runs.append(run)
                duplicate_methods.add(run.config["method"])

        ssl_d10_runs = filtered_ssl_d10_runs

        # sort by the summary value of the y_metric
        ssl_d10_runs.sort(key=lambda x: x.summary.get(y_metric, 0.), reverse=True)



        # Plot the figure
        plt.figure(figsize=(6, 6))
        plotter = Plotter()
        for i, run in enumerate(ssl_d10_runs):
            x, y = get_run_metrics(run)
            label = f'{run.config["method"]}'
            plotter.plot(x, y, label=label, color=f"C{i}")

        x, y = get_run_metrics(baseline_d10_run)
        label = f'Naive {C}={batch_repeat}'
        plotter.plot(x, y, label=label, color="black", lw=2, linestyle="--")

        plotter.annotate()



        # make the grid dashed
        set_default_settings()

        plt.xlabel("Time step")
        plt.ylabel("Online Accuracy")

        # plt.legend(framealpha=0.5, loc="lower right", fontsize=19)
        plt.legend(framealpha=0.5, loc="lower center", fontsize=19, ncol=2)
        plt.title(f"CLOC ($d=10$, {C}={batch_repeat*2})")
        plt.tight_layout()

        plt.savefig(f"fig-ssl-breakdown-C{2*batch_repeat}.pdf")
        plt.savefig(f"fig-ssl-breakdown-C{2*batch_repeat}.svg", format="svg")




def plot_unsupervised_ssl():
    """
    """

    def get_run_metrics(run, recompute_online_acc=True):

        # FOR BACKWARD COMPATIBILITY
        x_metric = "timestep"
        if run.summary.get(x_metric) is None:
            x_metric = "timestep_step"
        assert run.summary.get(x_metric) is not None

        # Since the CLOC training resuming didn't work on "next_batch_acc1",
        # we need to recompute the Online Accuracy - in theory they should be the same
        if recompute_online_acc:
            y_metric = "next_batch_acc1_current"
            run_metrics = run.scan_history(
                keys=[x_metric, y_metric],
                page_size=40000,
            )
            run_metrics = list(tqdm.tqdm(
                run_metrics,
                desc=f"Recomputing Online Accuracy for {run.name}",
                total=run.lastHistoryStep//2,
            ))
            # run_metrics is a list of dicts with values for each metric at each step
            # sort the list by the x_metric
            run_metrics.sort(key=lambda x: x[x_metric])

            x = [row[x_metric] for row in run_metrics]
            y = [row[y_metric] for row in run_metrics]

            # cumsum the y values
            y = np.cumsum(y) / np.arange(1, len(y)+1)

        else:

            # FOR BACKWARD COMPATIBILITY
            y_metric = "next_batch_acc1"
            if run.summary.get(y_metric) is None:
                y_metric = "next_batch_acc1_step"
            assert run.summary.get(y_metric) is not None

            run_metrics = run.history(
                keys=[x_metric, y_metric],
                x_axis=x_metric,
                samples=10000,
                pandas=False,
            )
            # run_metrics is a list of dicts with values for each metric at each step
            # sort the list by the x_metric
            run_metrics.sort(key=lambda x: x[x_metric])

            x = [row[x_metric] for row in run_metrics]
            y = [row[y_metric] for row in run_metrics]
        return x, y

    # if not os.path.exists("runs.pt"):
    api = wandb.Api()

    #############################
    # GATHER THE RUNS
    #############################
    #############################
    # NAIVE RUNS
    #############################
    cloc_naive_sweep_id = "w8kv7s1d"
    cloc_naive_runs = list(api.sweep("[author1]/onlineCL-cs1/"+cloc_naive_sweep_id).runs)
    # filter batch repeat = 2
    cloc_naive_runs = [run for run in cloc_naive_runs if run.config["online/batch_repeat"] == 2]

    cglm_naive_sweep_id = "74737bfd"
    cglm_naive_runs = list(api.sweep("[author1]/onlineCL-cs1/"+cglm_naive_sweep_id).runs)
    # filter num_supervised = 4
    cglm_naive_runs = [run for run in cglm_naive_runs if run.config["online"]["num_supervised"] == 4]


    #############################
    # SSL RUNS
    #############################
    # filter num_supervised = 1
    ssl_cloc_runs = []
    for ssl_cloc_sweep_id in ["iz12g5er"]:
        for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+ssl_cloc_sweep_id).runs):
            if (
                run.config["online"]["num_supervised"] == 1 and
                run.config["online"]["num_unsupervised"] == 1
            ):
                ssl_cloc_runs.append(run)

    ssl_cglm_sweep_id = "nyk8tlcu"
    # the baseline does 4 supervised and 0 unsupervised
    # so we filter for 2 supervised and 2 unsupervised
    ssl_cglm_runs = []
    for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+ssl_cglm_sweep_id).runs):
        if (
            run.config["online"]["num_supervised"] == 2 and
            run.config["online"]["num_unsupervised"] == 2
        ):
            ssl_cglm_runs.append(run)

    #############################
    runs = {
        "cloc": {
            "naive": cloc_naive_runs,
            "ssl": ssl_cloc_runs,
            # "tta": tta_cloc_runs,
            # "ours": ours_cloc_runs,
        },
        "cglm": {
            "naive": cglm_naive_runs,
            "ssl": ssl_cglm_runs,
            # "tta": tta_cglm_runs,
            # "ours": ours_cglm_runs,
        },
    }
    torch.save(runs, "runs.pt")

    # else:
    #     runs = torch.load("runs.pt")

    # define helper functions for extracting config values
    def get_delay(run):
        val1 = run.config.get("online/delay", None)
        if val1 is not None:
            return val1

        val2 = run.config.get("online.delay", None)
        if val2 is not None:
            return val2

        val3 = run.config.get("online", {}).get("delay", None)
        if val3 is not None:
            return val3

    def get_dataset(run):
        val1 = run.config.get("data/dataset", None)
        if val1 is not None:
            return val1
        val2 = run.config.get("data.dataset", None)
        if val2 is not None:
            return val2
        val3 = run.config.get("data", {}).get("dataset", None)
        if val3 is not None:
            return val3

    #############################
    # ARRANGE THE PLOTS
    #############################
    # make subplots


    datasets = ["cloc", "cglm"]
    # datasets = ["cloc"]
    # datasets = ["cglm"]
    delays = [10, 50, 100]
    methods = ["naive", "ssl"]

    # colors = {
    #     "naive-wod": "#ffb55a",
    #     "naive": "#ffb55a",
    #     # "ssl": "C0",
    #     # "tta": "C1",
    #     # "ours": "C2"

    #     # COLORBREWER
    #     # "ssl": "#a6cee3",
    #     # "tta": "#b2df8a",
    #     # "ours": "#fb9a99"

    #     # Spring pastels
    #     "ssl": "#7eb0d5",
    #     "tta": "#bd7ebe",
    #     "ours": "#fd7f6f"
    # }

    colors = {
        "naive": "#ffb55a",
        "mocov3": "#7eb0d5",
        "ressl": "#bd7ebe",
        "nnbyol": "#fd7f6f",
    }

    linestyles = {
        "naive-wod": "--",
        "naive": "-",
        "ssl": "-",
        "tta": "-",
        "ours": "-"
    }

    labels = {
        "naive": "Naïve",
        "mocov3": "MoCo v3",
        "ressl": "ResSL",
        "nnbyol": "NNBYOL",
    }

    # ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#8bd3c7"]

    # linestyles = [
    #     "-",
    #     "--",
    #     "-.",
    #     ":"
    # ]


    plt.figure(figsize=(13, 10))
    for i, dataset in enumerate(datasets):
        for j, delay in enumerate(delays):
            print(f"Plotting {dataset} {delay}")
            ax = plt.subplot(2, 3, i*3+j+1)
            plotter = Plotter(ax=ax)
            for method in methods:
                # method_runs = runs[dataset][method]
                if method == "naive-wod":
                    method_runs = runs[dataset]["naive"]
                else:
                    method_runs = runs[dataset][method]

                if method == "naive-wod":
                    method_runs = [run for run in method_runs if get_delay(run) == 0]
                else:
                    method_runs = [run for run in method_runs if get_delay(run) == delay]


                print(f"Found {len(method_runs)} runs for {method} {dataset} {delay}")
                if len(method_runs) == 0:
                    continue

                recompute_online_acc = method in ["ours", "ssl"] and dataset == "cloc"
                method_xys = {run: get_run_metrics(run, recompute_online_acc) for run in method_runs}
                # sort by the y value at the last step
                method_runs.sort(key=lambda x: method_xys[x][1][-1], reverse=True)

                if method != "ssl":
                    method_runs = method_runs[:1]

                # plot the best run
                for k, run in enumerate(method_runs):
                    if method == "naive":
                        label = f"Naive"
                    elif method == "naive-wod":
                        label = f"Naive w/o delay"
                    elif method == "ours":
                        label = f"Ours"
                    else:
                        label = f'{run.config["method"].lower()}'
                        # label = f'{method.upper()}'

                    # alpha = 1. if k == 0 else 0.5
                    # linestyle = linestyles[k]
                    x, y = method_xys[run]
                    color = colors[method] if method == "naive" else colors[run.config["method"].lower()]
                    plotter.plot(
                        x, y,
                        label=label,
                        color=color,
                        lw=3,
                        linestyle=linestyles[method],
                    )

                set_default_settings(ax=ax)
                # ylim = ax.get_ylim()
                # if dataset == "cloc":
                #     plt.ylim(ylim[0], 15)
                # else:
                #     plt.ylim(ylim[0], 23)

            plotter.annotate()

            # # only plot it for the bottom row
            if i == 1:
                plt.xlabel("Time step")

            # only plot it for the left column
            if j == 0:
                plt.ylabel("Online Accuracy")


            C_val = 2 if dataset == "cloc" else 8
            # plt.title(f"~~{dataset.upper()}"+r"\\"+f"$d$={delay}, {C}={C_val}")
            plt.title(f"~~{dataset.upper()} "+f"($d$={delay}, {C}={C_val})")
            # if dataset == "cloc":
            #     plt.xticks(rotation=45)

    # plot the legend
    plt.figlegend(*ax.get_legend_handles_labels(), loc='lower center', ncol=5)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.139, hspace=0.3)

    plt.savefig(f"fig-unsupervised-ssl.pdf")
    plt.savefig(f"fig-unsupervised-ssl.svg", format="svg")
    plt.close()




def plot_unsupervised_tta():
    """
    """

    def get_run_metrics(run, recompute_online_acc=True):

        # FOR BACKWARD COMPATIBILITY
        x_metric = "timestep"
        if run.summary.get(x_metric) is None:
            x_metric = "timestep_step"
        assert run.summary.get(x_metric) is not None

        # Since the CLOC training resuming didn't work on "next_batch_acc1",
        # we need to recompute the Online Accuracy - in theory they should be the same
        if recompute_online_acc:
            y_metric = "next_batch_acc1_current"
            run_metrics = run.scan_history(
                keys=[x_metric, y_metric],
                page_size=40000,
            )
            run_metrics = list(tqdm.tqdm(
                run_metrics,
                desc=f"Recomputing Online Accuracy for {run.name}",
                total=run.lastHistoryStep//2,
            ))
            # run_metrics is a list of dicts with values for each metric at each step
            # sort the list by the x_metric
            run_metrics.sort(key=lambda x: x[x_metric])

            x = [row[x_metric] for row in run_metrics]
            y = [row[y_metric] for row in run_metrics]

            # cumsum the y values
            y = np.cumsum(y) / np.arange(1, len(y)+1)

        else:

            # FOR BACKWARD COMPATIBILITY
            y_metric = "next_batch_acc1"
            if run.summary.get(y_metric) is None:
                y_metric = "next_batch_acc1_step"
            assert run.summary.get(y_metric) is not None

            run_metrics = run.history(
                keys=[x_metric, y_metric],
                x_axis=x_metric,
                samples=10000,
                pandas=False,
            )
            # run_metrics is a list of dicts with values for each metric at each step
            # sort the list by the x_metric
            run_metrics.sort(key=lambda x: x[x_metric])

            x = [row[x_metric] for row in run_metrics]
            y = [row[y_metric] for row in run_metrics]
        return x, y

    # if not os.path.exists("runs.pt"):
    api = wandb.Api()

    #############################
    # GATHER THE RUNS
    #############################
    #############################
    # NAIVE RUNS
    #############################
    cloc_naive_sweep_id = "w8kv7s1d"
    cloc_naive_runs = list(api.sweep("[author1]/onlineCL-cs1/"+cloc_naive_sweep_id).runs)
    # filter batch repeat = 2
    cloc_naive_runs = [run for run in cloc_naive_runs if run.config["online/batch_repeat"] == 2]

    cglm_naive_sweep_id = "74737bfd"
    cglm_naive_runs = list(api.sweep("[author1]/onlineCL-cs1/"+cglm_naive_sweep_id).runs)
    # filter num_supervised = 4
    cglm_naive_runs = [run for run in cglm_naive_runs if run.config["online"]["num_supervised"] == 4]


    #############################
    # TTA RUNS
    #############################
    # filter num_supervised = 1
    tta_cloc_runs = []
    for tta_cloc_sweep_id in ["k28p9bly", "kxe7t9ri"]:
        for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+tta_cloc_sweep_id).runs):
            if (
                run.config["online"]["num_supervised"] == 1 and
                run.config["online"]["num_unsupervised"] == 1
            ):
                tta_cloc_runs.append(run)

    # TODO: Missing sweep, need to run it first
    tta_cglm_sweep_id = "8lev423w"
    # filter num_supervised = 1
    tta_cglm_runs = []
    for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+tta_cglm_sweep_id).runs):
        if (
            run.config["online"]["num_supervised"] == 4 and
            run.config["online"]["num_unsupervised"] == 1
        ):
            tta_cglm_runs.append(run)

    #############################
    runs = {
        "cloc": {
            "naive": cloc_naive_runs,
            # "ssl": ssl_cloc_runs,
            "tta": tta_cloc_runs,
            # "ours": ours_cloc_runs,
        },
        "cglm": {
            "naive": cglm_naive_runs,
            # "ssl": ssl_cglm_runs,
            "tta": tta_cglm_runs,
            # "ours": ours_cglm_runs,
        },
    }
    torch.save(runs, "runs.pt")

    # else:
    #     runs = torch.load("runs.pt")

    # define helper functions for extracting config values
    def get_delay(run):
        val1 = run.config.get("online/delay", None)
        if val1 is not None:
            return val1

        val2 = run.config.get("online.delay", None)
        if val2 is not None:
            return val2

        val3 = run.config.get("online", {}).get("delay", None)
        if val3 is not None:
            return val3

    def get_dataset(run):
        val1 = run.config.get("data/dataset", None)
        if val1 is not None:
            return val1
        val2 = run.config.get("data.dataset", None)
        if val2 is not None:
            return val2
        val3 = run.config.get("data", {}).get("dataset", None)
        if val3 is not None:
            return val3

    #############################
    # ARRANGE THE PLOTS
    #############################
    # make subplots


    datasets = ["cloc"]
    # datasets = ["cloc"]
    # datasets = ["cglm"]
    delays = [10, 50, 100]
    methods = ["naive", "tta"]

    # colors = {
    #     "naive-wod": "#ffb55a",
    #     "naive": "#ffb55a",
    #     # "ssl": "C0",
    #     # "tta": "C1",
    #     # "ours": "C2"

    #     # COLORBREWER
    #     # "ssl": "#a6cee3",
    #     # "tta": "#b2df8a",
    #     # "ours": "#fb9a99"

    #     # Spring pastels
    #     "ssl": "#7eb0d5",
    #     "tta": "#bd7ebe",
    #     "ours": "#fd7f6f"
    # }

    colors = {
        "naive": "#ffb55a",
        "sar": "#7eb0d5",
        "tent": "#bd7ebe",
        "cotta": "#fd7f6f",
        "eat": "#8bd3c7",
    }

    linestyles = {
        "naive-wod": "--",
        "naive": "-",
        "ssl": "-",
        "tta": "-",
        "ours": "-"
    }

    # ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#8bd3c7"]

    # linestyles = [
    #     "-",
    #     "--",
    #     "-.",
    #     ":"
    # ]


    plt.figure(figsize=(13, 5))
    for i, dataset in enumerate(datasets):
        for j, delay in enumerate(delays):
            print(f"Plotting {dataset} {delay}")
            ax = plt.subplot(1, len(datasets)*len(delays), i*3+j+1)
            plotter = Plotter(ax=ax)
            for method in methods:
                # method_runs = runs[dataset][method]
                if method == "naive-wod":
                    method_runs = runs[dataset]["naive"]
                else:
                    method_runs = runs[dataset][method]

                if method == "naive-wod":
                    method_runs = [run for run in method_runs if get_delay(run) == 0]
                else:
                    method_runs = [run for run in method_runs if get_delay(run) == delay]


                print(f"Found {len(method_runs)} runs for {method} {dataset} {delay}")
                if len(method_runs) == 0:
                    continue

                recompute_online_acc = method in ["ours", "ssl"] and dataset == "cloc"
                method_xys = {run: get_run_metrics(run, recompute_online_acc) for run in method_runs}
                # sort by the y value at the last step
                method_runs.sort(key=lambda x: method_xys[x][1][-1], reverse=True)

                if method != "tta":
                    method_runs = method_runs[:1]

                # plot the best run
                for k, run in enumerate(method_runs):
                    if method == "naive":
                        label = f"Naïve"
                    elif method == "naive-wod":
                        label = f"Naïve w/o delay"
                    elif method == "ours":
                        label = f"Ours"
                    else:
                        # label = f'{run.config["method"].lower()}'
                        label = {
                            "sar": "SAR",
                            "tent": "TENT",
                            "cotta": "CoTTA",
                            "eat": "EAT",
                        }[run.config["method"].lower()]
                        # label = f'{method.upper()}'

                    # alpha = 1. if k == 0 else 0.5
                    # linestyle = linestyles[k]
                    x, y = method_xys[run]
                    color = colors[method] if method == "naive" else colors[run.config["method"].lower()]
                    plotter.plot(
                        x, y,
                        label=label,
                        color=color,
                        lw=3,
                        linestyle=linestyles[method],
                    )

                set_default_settings(ax=ax)
                # ylim = ax.get_ylim()
                # if dataset == "cloc":
                #     plt.ylim(ylim[0], 15)
                # else:
                #     plt.ylim(ylim[0], 23)

            plotter.annotate()

            # # only plot it for the bottom row
            # if i == 0:
            plt.xlabel("Time step")

            # only plot it for the left column
            if i == 0 and j == 0:
                plt.ylabel("Online Accuracy")


            C_val = 2 if dataset == "cloc" else 8
            # plt.title(f"~~{dataset.upper()}"+r"\\"+f"$d$={delay}, {C}={C_val}")
            plt.title(f"~~{dataset.upper()} "+f"($d$={delay}, {C}={C_val})")
            # if dataset == "cloc":
            #     plt.xticks(rotation=45)

    # plot the legend
    plt.figlegend(*ax.get_legend_handles_labels(), loc='lower center', ncol=5)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.30)

    plt.savefig(f"fig-unsupervised-tta.pdf")
    plt.savefig(f"fig-unsupervised-tta.svg", format="svg")
    plt.close()


def plot_backward_transfer():
    def get_delay(run):
        val1 = run.config.get("online/delay", None)
        if val1 is not None:
            return val1

        val2 = run.config.get("online.delay", None)
        if val2 is not None:
            return val2

        val3 = run.config.get("online", {}).get("delay", None)
        if val3 is not None:
            return val3

    def get_final_value(metrics):
        return metrics[1][-1] if metrics[1] else None

    #############################
    # GATHER THE RUNS
    #############################
    api = wandb.Api()

    rebuttal_project_id = "[author1]/iclr24-rebuttal"
    runs_list = list(tqdm.tqdm(api.runs(f"{rebuttal_project_id}")))

    for bwt_sweep_id in [
        "jnvl5wpx",
        "pyd0s6mf",
        "sekefeyr",
        "9goxx4as",
        "d11svpkq"
    ]:
        runs_list += list(api.sweep("[author1]/onlineCL-cs1/"+bwt_sweep_id).runs)

    def check_if_valid_yearbook(run):
        allowed_sweepids = [
            "v3ut3ye4", # Naive
            "i73m6n5k", # Pseudo
            "5l3rrhij", # TTA
            "2q2t2uh2", # IWM
            "lv2lklmt", # SSL
        ]
        original_runid = run.config["orig_runid"]
        original_run = api.run(f"[author1]/onlineCL-cs1/{original_runid}")
        return original_run.sweep.id in allowed_sweepids

    runs = {}
    for run in runs_list:
        if run.state != "finished":
            print(f"Skipping {run.id} because it is {run.state}")
            continue

        if run.config["data"]["dataset"] == "yearbook":
            if not check_if_valid_yearbook(run):
                print(f"Skipping {run.id} because it is an invalid yearbook exp")
                continue
            else:
                print(f"Keeping {run.id} because it is a valid yearbook exp")

        dataset = run.config["data"]["dataset"]
        method = run.config["method"]
        delay = run.config["online"]["delay"]

        if method == "iwm":
            if run.config["online"]["supervision_source"] == "NR":
                method = "naive"
                if delay == 0:
                    method = "naive-wod"
            else:
                method = "ours"

        if method == "ressl":
            method = "ssl"

        if method == "cotta":
            method = "tta"

        if dataset not in runs:
            runs[dataset] = {}
        if method not in runs[dataset]:
            runs[dataset][method] = {}
        if delay not in runs[dataset][method]:
            runs[dataset][method][delay] = []
        runs[dataset][method][delay].append(run)

    #############################
    # ARRANGE THE PLOTS
    #############################
    datasets = ["cloc", "cglm", "fmow", "yearbook"]
    delays = [10, 50, 100]
    methods = ["naive-wod", "naive", "ssl", "tta", "pseudo", "ours"]

    colors = {
        "naive-wod": "#ffb55a",
        "naive": "#ffb55a",
        "ssl": "#7eb0d5",
        "tta": "#bd7ebe",
        "pseudo": "#82c596",
        "ours": "#fd7f6f"
    }

    linestyles = {
        "naive-wod": "--",
        "naive": "-",
        "ssl": "-",
        "tta": "-",
        "ours": "-",
        "pseudo": "-",
    }

    labels = {
        "naive-wod": "Naïve w/o delay",
        "naive": "Naïve",
        "ssl": "S4L",
        "tta": "TTA",
        "pseudo": "Pseudo-Label",
        "ours": "IWMS",
    }

    dataset_names = {
        "cloc": "CLOC",
        "cglm": "CGLM",
        "fmow": "FMoW",
        "yearbook": "Yearbook",
    }

    block_size = 4.5
    legend_size = 1
    num_rows = len(delays)
    num_cols = len(datasets)
    C_vals = {
        "cloc": 2,
        "cglm": 8,
        "fmow": 16,
        "yearbook": 16,
    }

    plt.figure(figsize=(num_cols*block_size, num_rows*block_size*.95))
    final_values = {}

    for i, delay in enumerate(delays):
        for j, dataset in enumerate(datasets):
            print(f"Plotting {dataset} {delay}")
            ax = plt.subplot(num_rows, num_cols, i*num_cols+j+1)
            plotter = Plotter(ax=ax)
            final_values[dataset] = final_values.get(dataset, {})

            for method in methods:
                if method == "naive-wod":
                    method_run = runs[dataset]["naive-wod"][0][0]
                else:
                    if dataset not in runs or method not in runs[dataset] or delay not in runs[dataset][method]:
                        continue
                    method_runs = runs[dataset][method][delay]
                    print(f"Found {len(method_runs)} runs for {method} {dataset} {delay}")

                    best_run = None
                    for run in method_runs:
                        if best_run is None:
                            best_run = run
                        else:
                            if run.summary["backward_transfer"] > best_run.summary["backward_transfer"]:
                                best_run = run
                    method_run = best_run

                if method == "tta":
                    continue

                method_xy = get_run_metrics(method_run, x_metric="timestep", y_metric="backward_transfer")
                label = labels[method]
                x, y = method_xy
                x = np.array(x)
                y = np.array(y) * 100

                plotter.plot(
                    x, y,
                    label=label,
                    color=colors[method],
                    lw=3,
                    linestyle=linestyles[method],
                )

                final_value = get_final_value(method_xy)
                if final_value is not None:
                    final_values[dataset][(method, delay)] = final_value

                if dataset == "cloc":
                    plt.ylim(5, 15)
                if dataset == "cglm":
                    plt.ylim(20, 63)
                if dataset == "fmow":
                    plt.ylim(49, 59)
                if dataset == "yearbook":
                    plt.ylim(85, 100)

                set_default_settings(ax=ax)

            plotter.annotate(fontsize=22)

            if i == len(delays)-1:
                plt.xlabel("\# Samples in the Past")

            if j == 0:
                plt.ylabel(f"Backward Transfer ($d={delay}$)", fontsize=18)

            C_val = C_vals[dataset]
            if i == 0:
                plt.title(f"{dataset_names[dataset]} ({C}={C_val})")

    ax = plt.subplot(num_rows, num_cols, 1)
    plt.figlegend(*ax.get_legend_handles_labels(), loc='lower center', ncol=5)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.125, hspace=0.1)
    plt.savefig(f"fig-backward.pdf")
    plt.savefig(f"fig-backward.svg", format="svg")
    plt.close()

    #############################
    # GENERATE MARKDOWN TABLE
    #############################


    for dataset in datasets:
        print(f"#### {dataset_names[dataset]} Dataset\n")

        # Print the header with symbols
        print("| Delay (d)       | ★ Naïve w/o delay | ◆ Naïve | ✚ S4L | ✖ Pseudo-Label | ▲ IWMS |")
        print("|-----------------|-------------------|---------|-------|----------------|--------|")

        # Print each delay row
        for i, delay in enumerate(delays):
            if i == 0:
                delay_label = "**d=10** (top)"
            elif i == 1:
                delay_label = "**d=50** (middle)"
            else:
                delay_label = "**d=100** (bottom)"

            values = []
            for method in ["naive-wod", "naive", "ssl", "pseudo", "ours"]:
                value = final_values.get(dataset, {}).get((method, delay), None)
                if value is not None:
                    values.append(f"{100 * value:2.1f} %")
                else:
                    values.append("-")

            # Join the values with the delay label
            print(f"| {delay_label:<15} | {' | '.join(values)} |")

        print()  # Add a newline after each dataset's table



def plot_fig_matching_joyplot():
    api = wandb.Api()

    two_stage_run = {
        "title": "Similarity Matching \\textbf{after} Classification",
        "id": "eflyc61b",
        "key": "histogram/iwm/score/query/avg",
        "filename": "fig-matching-joyplot-two-stage",
    }
    single_shot_run = {
        "title": "Similarity Matching \\textbf{without} Classification",
        "id": "1hzx46mt",
        "key": "histogram/score_avg_across_query",
        "filename": "fig-matching-joyplot-single-shot",
    }
    for run_info in [two_stage_run, single_shot_run]:
        run_id = run_info["id"]
        key = run_info["key"]
        run = api.run(f"[author1]/onlineCL-cs1/{run_id}")


        # Extract the specific histogram data
        hist_data = run.history(keys=["timestep", key])
        df = pd.DataFrame.from_records(hist_data[key])
        df["timestep"] = hist_data["timestep"]

        # Expand the histogram data
        rows = []
        for _, row in df.iterrows():
            bins = row['bins']
            values = row['values']
            timestep = row['timestep']
            bin_centers = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]  # Compute bin centers
            for bin_center, value in zip(bin_centers, values):
                # rows.append({'bin_center': bin_center, 'value': value, 'timestep': timestep})
                for _ in range(int(value)):
                    rows.append({
                        'value': bin_center,
                        'exp_value': np.exp(bin_center),
                        'neglog_value': -np.log(bin_center),
                        'timestep': timestep,
                    })

        expanded_df = pd.DataFrame(rows)

        # Plotting
        # plt.figure(figsize=(6, 5))
        if run_info == two_stage_run:
            # colormap = "C1"
            colormap = cm.Blues_r
        else:
            colormap = cm.Oranges_r

        fig, axes = joyplot(
            data=expanded_df[expanded_df['timestep'] % 300 == 0][expanded_df['timestep'] <= 3000],
            by='timestep',
            # column='log_value',
            # column='exp_value',
            column='value',
            bins=20,
            fade=True,
            figsize=(8, 10),
            # hist=True,
            ylim='own',
            overlap=0.1,
            colormap=colormap,
        )


        # # Set y-axis labels fontsize
        # for ax in axes:
        #     ax.set_ylabel(ax.get_ylabel(), fontsize=12)  # Adjust fontsize as needed
        #     ax.set_xlabel(ax.get_xlabel(), fontsize=12)  # Adjust fontsize as needed


        # # Set fontsizes to normal
        # plt.xticks(fontsize=12)
        # plt.yticks(fontsize=12)
        plt.xlabel("Similarity Score")
        # plt.title("Distribution of Similarity Scores over Time", fontsize=23)
        plt.title(run_info["title"], fontsize=23)

        plt.savefig(f"{run_info['filename']}.pdf")
        plt.savefig(f"{run_info['filename']}.svg", format="svg")


def plot_fig_two_stage_vs_single_shot():
    api = wandb.Api()

    two_stage_run = {
        "title": "Two-stage",
        "id": "eflyc61b",
        "key": "histogram/iwm/score/query/avg",
        "filename": "fig-matching-joyplot-two-stage",
    }
    single_shot_run = {
        "title": "Single-shot",
        "id": "1hzx46mt",
        "key": "histogram/score_avg_across_query",
        "filename": "fig-matching-joyplot-single-shot",
    }

    plt.figure(figsize=(8, 10))
    plotter = Plotter()

    for run_info in [two_stage_run, single_shot_run]:
        run_id = run_info["id"]
        key = run_info["key"]
        run = api.run(f"[author1]/onlineCL-cs1/{run_id}")
        run_xy = get_run_metrics(run)
        x, y = run_xy
        # plt.plot(x, y, label=run_info["title"])
        plotter.plot(x, y, label=run_info["title"], lw=3)

    plotter.annotate(fontsize=23)
    set_default_settings()
    plt.title("Performance of Similarity Matching Policy", fontsize=23)
    plt.xlabel("Time step")
    plt.ylabel("Online Accuracy")

    plt.legend()
    plt.savefig(f"fig-two-stage-vs-single-shot.pdf")
    plt.savefig(f"fig-two-stage-vs-single-shot.svg", format="svg")



if __name__ == "__main__":
    funcs = {
        "delay_ablation": plot_delay_ablation,
        "compute": plot_fig_compute,
        "unsupervised": plot_unsupervised,
        "memory_size": plot_fig_memory_size,
        "sampling_strategy": plot_fig_sampling_strategy,
        "increased_compute": plot_fig_increased_compute,
        "temperature": plot_fig_temperature,
        "ssl_breakdown": plot_ssl_breakdown,
        "unsupervised_ssl": plot_unsupervised_ssl,
        "unsupervised_tta": plot_unsupervised_tta,
        "backward_transfer": plot_backward_transfer,
        "matching_joyplot": plot_fig_matching_joyplot,
        "two_stage_vs_single_shot": plot_fig_two_stage_vs_single_shot,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("func", nargs="+")
    args = parser.parse_args()
    if args.func == ["all"]:
        args.func = funcs.keys()

    # find closest matches
    matched_funcs = []
    for func in args.func:
        if func not in funcs:
            matches = difflib.get_close_matches(func, funcs.keys(), n=1, cutoff=0.1)

            if len(matches) == 0:
                raise ValueError(f"Invalid function {func}")
            else:
                # replace the invalid function with the closest match
                print(f"Automatically replacing {func} with {matches[0]}")
                matched_funcs.append(matches[0])
        else:
            matched_funcs.append(func)

    for func in matched_funcs:
        print(f"Plotting {func}")
        funcs[func]()