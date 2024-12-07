import wandb
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import matplotlib.ticker as mticker
from matplotlib import rcParams
import matplotlib
import numpy as np
import tqdm
import json
import torch
import os

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

plt.rcParams["font.size"] = 25
plt.rcParams["figure.figsize"] = [9, 7]

matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
matplotlib.rcParams['text.usetex'] = True


C = "$\mathcal{C}$"


def thousands_formatter(x, pos):
    """Formatter to display numbers in the format of 10k, 20k, 30k, etc."""
    return f'{int(x/1000)}k'

def delay_filter(delay):
    return lambda x: x.config["online/delay"] == delay

def repeat_filter(repeat):
    return lambda x: x.config["online/batch_repeat"] == repeat

def concat_filter(filters):
    return lambda x: all([f(x) for f in filters])

def or_filter(filters):
    return lambda x: any([f(x) for f in filters])

x_metric = "timestep_step"
y_metric = "next_batch_acc1_step"

def get_run_metrics(run, x_metric=x_metric, y_metric=y_metric):
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

def plot_delay_ablation(subplot):
    """
    Plot figure 3 from the paper

    Ablation of the delay factor
    """

    sweep_id = "[author1]/onlineCL-cs1/w8kv7s1d" if subplot == "a" else "[author1]/onlineCL-cs1/74737bfd"

    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    runs = list(sweep.runs)
    if subplot == "a":
        # only keep runs with C=1
        runs = [run for run in runs if run.config["online/batch_repeat"] == 1]
        runs = [run for run in runs if run.config["online/delay"] in [0, 10, 50, 100]]
    elif subplot == "b":
        # only keep runs with C=1
        runs = [run for run in runs if run.config["online.num_supervised"] == 4]
        runs = [run for run in runs if run.config["online.delay"] in [0, 10, 50, 100]]
    else:
        raise ValueError(f"Invalid subplot {subplot}")




    # colors = colors = cm.viridis(np.linspace(0, 1, len(runs)))
    # Oranges
    colors = cm['Oranges'](np.linspace(0.2, 1, len(runs)))

    # sort runs by delay
    runs.sort(key=lambda x: x.config["online.delay"])


    plt.figure(figsize=(6, 6))
    for i, run in enumerate(tqdm.tqdm(runs)):
        x, y = get_run_metrics(run)
        label = f'$d$={run.config["online.delay"]}'
        plt.plot(x, y, label=label, color=colors[~i], lw=3)


    # make the grid dashed
    plt.grid(True, linestyle="--", which='major',axis='both',alpha=0.3)

    plt.xlabel("Time step")
    plt.ylabel("Online Accuracy")
    title = "Continual Localization (CLOC)" if subplot == "a" else "Continual Google Landmarks (CGLM)"
    plt.title(title, fontsize=24)
    plt.legend(framealpha=0.5, fontsize=18, loc="lower right")


    # set the number of ticks
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(thousands_formatter))

    plt.tight_layout()


    plt.savefig(f"fig-delay-ablation-{subplot}.png")
    plt.savefig(f"fig-delay-ablation-{subplot}.pdf")


def plot_fig_compute(dataset):
    """
    Plot fig:compute from the paper

    Ablation of the Compute factor
    """
    api = wandb.Api()
    plt.figure(figsize=(15, 6))

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
    colors = cm['Oranges']([0.6, 0.6])
    # lws = np.linspace(3, 2, len(Cs))
    lws = [3, 3]
    # lws = [3, 1, 1, 1]
    # linetypes = [":", "-.", "--", "-"]
    # linetypes = ["-", "-", "-", "-"]
    linetypes = ["-", "--", ":", ":"]


    for plot_idx, subplot in enumerate(subplots):
        ax = plt.subplot(1, 3, plot_idx+1)
        delay = delays[subplot]
        baseline_low_run = [run for run in runs if run.config["online.delay"] == 0 and run.config[iter_str] == Cs[0]][0]
        baseline_high_run = [run for run in runs if run.config["online.delay"] == 0 and run.config[iter_str] == Cs[-1]][0]
        d_runs = [run for run in runs if run.config["online.delay"] == delay and run.config[iter_str] in Cs]

        # sort runs by repeat
        d_runs.sort(key=lambda x: x.config[iter_str], reverse=True)


        x, y = get_run_metrics(baseline_high_run, y_metric=y_metric)
        label = f'{C}={baseline_high_run.config[iter_str]} w/o delay'
        plt.plot(x, y, color="black", label=label, linestyle="-", lw=lws[0])

        x, y = get_run_metrics(baseline_low_run, y_metric=y_metric)
        label = f'{C}={baseline_low_run.config[iter_str]} w/o delay'
        plt.plot(x, y, color="black", label=label, linestyle="--", lw=lws[0])

        for i, run in enumerate(tqdm.tqdm(d_runs)):
            x, y = get_run_metrics(run, y_metric=y_metric)
            label = f'{C}={run.config[iter_str]}'
            color = colors[i]
            plt.plot(x, y, color=color, label=label, linestyle=linetypes[i], lw=lws[i], alpha=0.8)


        # make the grid dashed
        plt.grid(True, linestyle="--", which='major',axis='both',alpha=0.3)


        # plt.xlim(-3000, 33000)
        plt.xlabel("Time step")
        # only show the y label on the leftmost plot
        if plot_idx == 0:
            plt.ylabel("Online Accuracy")

        # plt.legend(framealpha=0.5, loc="lower right", fontsize=17)

        plt.title(f"{dataset.upper()} ($d$={delay})")
        # plt.tight_layout()

        # set the number of ticks
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(thousands_formatter))

    # plot the legend
    plt.figlegend(
        *ax.get_legend_handles_labels(), loc='lower center', ncol=6,
        fontsize=22
    )
    # plt.figlegend(*ax.get_legend_handles_labels(), loc='center right', ncol=1)

    # put the legend to the right of the figures
    # plt.subplots_adjust(right=0.85)
    # title = "Continual Localization (CLOC) dataset" if dataset == "CLOC" else "Continual Google Landmarks (CGLM) dataset"
    # plt.suptitle(title, fontsize=24)


    plt.tight_layout(rect=[0, 0.075, 1.0, 1.0])
    # plt.tight_layout()
    # plt.subplots_adjust(bottom=0.22, wspace=0.5, hspace=0.3)

    plt.savefig(f"fig-compute-{dataset}.pdf")
    plt.savefig(f"fig-compute-{dataset}.png")


def plot_fig_fair_compute():
    """
    Plot fig fair compute from the paper

    Ablation of the Compute factor with equal compute
    """
    api = wandb.Api()
    plt.figure(figsize=(15, 5))

    # sweep_id = "[author1]/onlineCL-cs1/op6hwx8n"
    sweep_id = "[author1]/onlineCL-cs1/w8kv7s1d"
    runs = list(api.sweep(sweep_id).runs)

    # sweep_ids = ["[author1]/onlineCL-cs1/2wf30m5i", "[author1]/onlineCL-cs1/bpb21z4w", "[author1]/onlineCL-cs1/6mj3k6tv", "[author1]/onlineCL-cs1/z12dp3iz"]
    # runs = []
    # for sweep_id in sweep_ids:
    #     sweep = api.sweep(sweep_id)
    #     runs += list(sweep.runs)


    # make subplots
    subplots = ["a", "b", "c"]
    for i, subplot in enumerate(subplots):
        ax = plt.subplot(1, 3, i+1)

        delay = {"a": 10, "b": 50, "c": 100}[subplot]

        baseline_run_c1 = [run for run in runs if run.config["online/delay"] == 0 and run.config["online/batch_repeat"] == 1][0]
        baseline_run_c8 = [run for run in runs if run.config["online/delay"] == 0 and run.config["online/batch_repeat"] == 8][0]
        d_run_c1 = [run for run in runs if run.config["online/delay"] == delay and run.config["online/batch_repeat"] == 1][0]
        d_run_c8 = [run for run in runs if run.config["online/delay"] == delay and run.config["online/batch_repeat"] == 8][0]




        # Baseline C=8
        x, y = get_run_metrics(baseline_run_c8, y_metric=y_metric)
        label = f'{C}=8 w/o delay'
        plt.plot(x, y, label=label, color="black", linestyle="-", lw=3)


        # Baseline C=1
        x, y = get_run_metrics(baseline_run_c1, y_metric=y_metric)
        label = f'{C}=1 w/o delay'
        plt.plot(x, y, label=label, color="black", linestyle=":", lw=3)

        # color = {"a": "C0", "b": "C1", "c": "tab:purple"}[subplot]
        color = "C1"
        # Delay C=8
        x, y = get_run_metrics(d_run_c8, y_metric=y_metric)
        label = f'{C}=8'
        plt.plot(x, y, label=label, color=color, linestyle="-", lw=3)

        # Delay C=1
        x, y = get_run_metrics(d_run_c1, y_metric=y_metric)
        label = f'{C}=1'
        plt.plot(x, y, label=label, color=color, linestyle=":", lw=3)


        # make the grid dashed
        plt.grid(True, linestyle="--", which='major',axis='both',alpha=0.3)


        # plt.xlim(-3000, 33000)
        plt.xlabel("Time step")
        plt.ylabel("Online Accuracy")
        # plt.legend(framealpha=0.5, loc="center left", fontsize=17)

        plt.title(f"$d$={delay}")

        # set the number of ticks
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(thousands_formatter))


    # plot the legend
    plt.figlegend(*ax.get_legend_handles_labels(), loc='lower center', ncol=4)

    # avoid overlapping of the legend and the subplots
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, wspace=0.3, hspace=0.3)


    plt.savefig(f"fig-fair-compute.pdf")
    plt.savefig(f"fig-fair-compute.png")



def plot_fig_compute_memory(subplot):
    """
    Plot fig:compute from the paper

    Ablation of the Compute factor
    """

    y_metric = "memory_batch_acc1_step"

    sweep_id = "[author1]/onlineCL-cs1/op6hwx8n"
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    runs = list(sweep.runs)

    delay = {"a": 10, "b": 50, "c": 100}[subplot]

    baseline_run = [run for run in runs if run.config["online/delay"] == 0 and run.config["online/batch_repeat"] == 1][0]
    d_runs = [run for run in runs if run.config["online/delay"] == delay and run.config["online/batch_repeat"] in [1, 2, 4, 8]]

    # sort runs by repeat
    d_runs.sort(key=lambda x: x.config["online/batch_repeat"])

    plt.figure()

    x, y = get_run_metrics(baseline_run, y_metric=y_metric)
    label = f'Naive Baseline'
    plt.plot(x, y, label=label, color="black")
    linetypes = [":", "-.", "--", "-"]
    for i, run in enumerate(tqdm.tqdm(d_runs)):
        x, y = get_run_metrics(run, y_metric=y_metric)
        label = f'$d$={run.config["online/delay"]:3d} ~ {C}={run.config["online/batch_repeat"]}'
        color = "C0" if subplot == "a" else "C1"
        plt.plot(x, y, label=label, color=color, linestyle=linetypes[i])


    # make the grid dashed
    plt.grid(True, linestyle="--", which='major',axis='both',alpha=0.3)

    plt.xlabel("Time step")
    plt.ylabel("Top-1 Accuracy on Seen Samples")
    plt.legend(framealpha=0.5, loc="lower right", fontsize=19)

    plt.title("$d$=10" if subplot == "a" else "$d$=100")

    plt.ylim(-1., 41.)

    # set the number of ticks
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.savefig(f"fig-compute-memory-{subplot}.pdf")
    plt.savefig(f"fig-compute-memory-{subplot}.png")



def plot_fig3(category="SSL"):
    """
    Plot figure 3 from the paper

    Comparison between SSL and TTA

    (a) is for SSL
    """

    def get_run_metrics(run):
        # locally override because there are some runs with faulty X axis values
        x_metric = "timestep_step"
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

    api = wandb.Api()

    # OUTDATED SWEEP IDS
    # "gt2xztjk", # TTA
    # "q4hhzx04", # SSL
    # "arbac2as", # MOCOv3
    # "6thdyupd", # SSL 60k
    # "dp9ybefy", # SAR

    print("Fetching runs")
    sweep_ids = [
        "w8kv7s1d", # Naive
        "95m20lye", # SSL
    ]

    runs = []
    for sweep_id in sweep_ids:
        sweep = api.sweep("[author1]/onlineCL-cs1/"+sweep_id)
        runs.extend(list(sweep.runs))

    # filters = concat_filter([repeat_filter(1)])
    # runs = [run for run in runs if filters(run)]

    # Baseline d=0 with black curve
    # Baseline d=10 with C0 curve
    # Baseline d=100 with C1 curve

    # best method with dashed curve in both d=10 and d=100
    # shaded area for the best vs worst method

    baseline_d0_run = [run for run in runs if run.config.get("online/delay") == 0 and run.config.get("online/batch_repeat") == 2 and run.config["category"].lower() == "naive"][0]
    baseline_d10_run = [run for run in runs if run.config.get("online/delay") == 50 and run.config.get("online/batch_repeat") == 2 and run.config["category"].lower() == "naive"][0]
    # baseline_d100_run = [run for run in runs if run.config.get("online/delay") == 100 and run.config.get("online/batch_repeat") == 1 and run.config["category"].lower() == "naive"][0]

    ssl_d10_runs = [run for run in runs if run.config.get("online/delay") == 50 and run.config.get("online/batch_repeat") == 1 and run.config["category"].lower() == "ssl"]
    # ssl_d100_runs = [run for run in runs if run.config.get("online/delay") == 100 and run.config.get("online/batch_repeat") == 1 and run.config["category"].lower() == "ssl"]


    # select the best and wors method for d=10
    best_d10_run = None
    best_d10_acc = 0.

    worst_d10_run = None
    worst_d10_acc = 100.

    for run in ssl_d10_runs:
        x, y = get_run_metrics(run)
        if y[-1] > best_d10_acc:
            best_d10_acc = y[-1]
            best_d10_run = run
        if y[-1] < worst_d10_acc:
            worst_d10_acc = y[-1]
            worst_d10_run = run

    # select the best and worst method for d=100
    # best_d100_run = None
    # best_d100_acc = 0.

    # worst_d100_run = None
    # worst_d100_acc = 100.

    # for run in ssl_d100_runs:
    #     x, y = get_run_metrics(run)
    #     if y[-1] > best_d100_acc:
    #         best_d100_acc = y[-1]
    #         best_d100_run = run
    #     if y[-1] < worst_d100_acc:
    #         worst_d100_acc = y[-1]
    #         worst_d100_run = run


    # Plot the figure
    plt.figure()
    x, y = get_run_metrics(baseline_d0_run)
    label = f'Naive ($d$={baseline_d0_run.config["online/delay"]})'
    plt.plot(x, y, label=label, color="black")

    x, y = get_run_metrics(baseline_d10_run)
    label = f'Naive ($d$={baseline_d10_run.config["online/delay"]})'
    plt.plot(x, y, label=label, color="C0")

    x, y = get_run_metrics(best_d10_run)
    label = f'Best {category} ($d$={best_d10_run.config["online/delay"]})'
    plt.plot(x, y, label=label, color="C0", linestyle="--")

    # x, y = get_run_metrics(baseline_d100_run)
    # label = f'Naive ($d$={baseline_d100_run.config["online/delay"]})'
    # plt.plot(x, y, label=label, color="C1")


    # x, y = get_run_metrics(best_d100_run)
    # label = f'Best {category} ($d$={best_d100_run.config["online/delay"]})'
    # plt.plot(x, y, label=label, color="C1", linestyle="--")


    # draw the shaded areas
    best_d10_x, best_d10_y = get_run_metrics(best_d10_run)
    worst_d10_x, worst_d10_y = get_run_metrics(worst_d10_run)
    # best_d100_x, best_d100_y = get_run_metrics(best_d100_run)
    # worst_d100_x, worst_d100_y = get_run_metrics(worst_d100_run)

    plt.fill_between(best_d10_x, best_d10_y, worst_d10_y, color="C0", alpha=0.2)
    # plt.fill_between(best_d100_x, best_d100_y, worst_d100_y, color="C1", alpha=0.2)

    # make the grid dashed
    plt.grid(True, linestyle="--", which='major',axis='both',alpha=0.3)


    plt.xlabel("Time step")
    plt.ylabel("Online Accuracy")

    plt.legend(framealpha=0.5, loc="lower right", fontsize=19)

    # set the number of ticks
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)

    # make title bold face
    plt.title(r"\textbf{Self Supervised Learning}" if category == "SSL" else r"\textbf{Test Time Augmentation}")

    plt.savefig(f"fig3-{category}.pdf")
    plt.savefig(f"fig3-{category}.png")


def plot_ssl_breakdown(subfigure):
    """
    Plot figure 4 from the paper

    Detailed SSL ablation at d=10 C=1
    """

    batch_repeat = {"a": 1, "b": 5, "c": 10, "d": 20}[subfigure]


    def get_run_metrics(run):
        # locally override because there are some runs with faulty X axis values
        x_metric = "trainer/global_step"
        run_metrics = run.history(
            keys=[x_metric, y_metric],
            x_axis=x_metric,
            samples=1000,
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
    plt.figure()
    for i, run in enumerate(ssl_d10_runs):
        x, y = get_run_metrics(run)
        label = f'{run.config["method"]}'
        plt.plot(x, y, label=label, color=f"C{i}")

    x, y = get_run_metrics(baseline_d10_run)
    label = f'Naive'
    plt.plot(x, y, label=label, color="black", lw=2, linestyle="--")

    R = "\mathcal{R}"
    plt.title(f"$d=10$, ${R}={batch_repeat}$")

    # make the grid dashed
    plt.grid(True, linestyle="--", which='major',axis='both',alpha=0.3)


    plt.xlabel("Time step")
    plt.ylabel("Online Accuracy")

    plt.legend(framealpha=0.5, loc="lower right", fontsize=19)

    # set the number of ticks
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)

    plt.savefig(f"fig-ssl-breakdown-{subfigure}.pdf")
    plt.savefig(f"fig-ssl-breakdown-{subfigure}.png")



def plot_fig_scale(subfig):
    """
    Compare scaleability of the Naive method and the best performing SSL method

    At a fixed delay, plot the accuracy on multiple C values
    """

    # def get_run_metrics(run):
    #     # locally override because there are some runs with faulty X axis values
    #     x_metric = "timestep_step"
    #     run_metrics = run.history(
    #         keys=[x_metric, y_metric],
    #         x_axis="trainer/global_step",
    #         samples=1000,
    #         pandas=False,
    #     )
    #     # run_metrics is a list of dicts with values for each metric at each step
    #     # sort the list by the x_metric
    #     run_metrics.sort(key=lambda x: x["trainer/global_step"])

    #     # find the index of the closest step to 30000
    #     idx = np.argmin(np.abs(np.array([row[x_metric] for row in run_metrics]) - 30000))
    #     run_metrics = run_metrics[:idx+1]


    #     x = [row[x_metric] for row in run_metrics]
    #     y = [row[y_metric] for row in run_metrics]

    #     return x, y


    delay = {"a": 10, "b": 50, "c": 100}[subfig]

    api = wandb.Api()

    print("Fetching runs")
    sweep_ids = [
        "op6hwx8n", # Naive
        "gt2xztjk", # TTA
        "q4hhzx04", # SSL
        "arbac2as", # MOCOv3
        "dp9ybefy", # SAR
        "t7zo3eft", # Old Naive
    ]

    runs = []
    for sweep_id in sweep_ids:
        sweep = api.sweep("[author1]/onlineCL-cs1/"+sweep_id)
        runs.extend(list(sweep.runs))

    # filter for delay
    runs = [run for run in runs if run.config["online/delay"] == delay]

    # filter for batch repeat
    naive_scale1_run = [run for run in runs if run.config["online/batch_repeat"] == 1 and run.config["category"].lower() == "naive"][0]
    naive_scale2_run = [run for run in runs if run.config["online/batch_repeat"] == 5 and run.config["category"].lower() == "naive"][0]
    naive_scale3_run = [run for run in runs if run.config["online/batch_repeat"] == 10 and run.config["category"].lower() == "naive"][0]

    # Cost of SSL is 2x of Naive
    ssl_scale1_runs = [run for run in runs if run.config["online/batch_repeat"] == 1 and run.config["category"].upper() == "SSL"]
    ssl_scale2_runs = [run for run in runs if run.config["online/batch_repeat"] == 5 and run.config["category"].upper() == "SSL"]
    ssl_scale3_runs = [run for run in runs if run.config["online/batch_repeat"] == 10 and run.config["category"].upper() == "SSL"]

    # Select the best performing run for each C value
    best_scale1_run = None
    best_scale2_run = None
    best_scale3_run = None

    best_scale1_acc = 0
    best_scale2_acc = 0
    best_scale3_acc = 0

    for run in ssl_scale1_runs:
        x, y = get_run_metrics(run)
        if y[-1] > best_scale1_acc:
            best_scale1_acc = y[-1]
            best_scale1_run = run

    for run in ssl_scale2_runs:
        x, y = get_run_metrics(run)
        if y[-1] > best_scale2_acc:
            best_scale2_acc = y[-1]
            best_scale2_run = run

    for run in ssl_scale3_runs:
        x, y = get_run_metrics(run)
        if y[-1] > best_scale3_acc:
            best_scale3_acc = y[-1]
            best_scale3_run = run


    print("Best performing runs")
    print(best_scale1_run.name, best_scale1_acc)
    print(best_scale2_run.name, best_scale2_acc)
    print(best_scale3_run.name, best_scale3_acc)

    # Plot the figure
    plt.figure(figsize=(5, 7))
    # plt.figure()

    # Plot naives with black line
    # c2 - is continuous
    # c4 - is dashed
    # c8 - is dotted

    R = "$\mathcal{R}$"

    lw = 2
    color = {"a": "C0", "b": "C1", "c": "tab:purple"}[subfig]

    x, y = get_run_metrics(naive_scale1_run)
    label = f'Naive ({R}=1)'
    plt.plot(x, y, label=label, color="black", lw=lw, linestyle=":")

    x, y = get_run_metrics(naive_scale2_run)
    label = f'Naive ({R}=5)'
    plt.plot(x, y, label=label, color="black", lw=lw, linestyle="--")

    x, y = get_run_metrics(naive_scale3_run)
    label = f'Naive ({R}=10)'
    plt.plot(x, y, label=label, color="black", lw=lw, linestyle="-")

    # Plot SSL with C0 line
    x, y = get_run_metrics(best_scale1_run)
    label = f'SSL ({R}=1)'
    plt.plot(x, y, label=label, color=color, lw=lw, linestyle=":")

    x, y = get_run_metrics(best_scale2_run)
    label = f'SSL ({R}=5)'
    plt.plot(x, y, label=label, color=color, lw=lw, linestyle="--")

    x, y = get_run_metrics(best_scale3_run)
    label = f'SSL ({R}=10)'
    plt.plot(x, y, label=label, color=color, lw=lw, linestyle="-")

    plt.xlabel("Time step")
    plt.ylabel("Online Accuracy")

    plt.legend(framealpha=0.5, loc="lower right", fontsize=19)

    # set the number of ticks
    plt.ylim(-1., 12.)
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.grid(True, linestyle="--", which='major',axis='both',alpha=0.3)

    plt.title(f"$d={delay}$")
    plt.tight_layout()

    plt.savefig(f"fig-scale-{subfig}.pdf")
    plt.savefig(f"fig-scale-{subfig}.png")

    # make the grid dashed

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
    # OUR RUNS
    #############################
    ours_cloc_sweep_id = "yc8tggf0" # Two sources
    # ours_cloc_sweep_id = "g5l52r6h" # Three sources
    # filter num_supervised = 2
    ours_cloc_runs = []
    for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+ours_cloc_sweep_id).runs):
        if (
            run.config["online"]["num_supervised"] == 2 and
            run.config["online"]["num_unsupervised"] == 1 and
            # run.config["online"]["supervision_source"] in ["NR"]
            run.config["online"]["supervision_source"] in ["WR"]
            # run.config["online"]["supervision_source"] in ["WR", "RR"]
            # run.config["online"]["supervision_source"] in ["WRR", "WWR", "RRR", "NWR"]
        ):
            ours_cloc_runs.append(run)

    # ours_cglm_sweep_id = "ls0c8o4x"
    ours_cglm_sweep_id = "qhu45cxz"
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

    runs = {
        "cloc": {
            "naive": cloc_naive_runs,
            "ssl": ssl_cloc_runs,
            "tta": tta_cloc_runs,
            "ours": ours_cloc_runs,
        },
        "cglm": {
            "naive": cglm_naive_runs,
            "ssl": ssl_cglm_runs,
            "tta": tta_cglm_runs,
            "ours": ours_cglm_runs,
        },
    }
    torch.save(runs, "runs.pt")

    # else:
    #     runs = torch.load("runs.pt")

    # define helper functions for extracting config values
    def get_delay(run):
        val1 = run.config.get("online/delay", None)
        val2 = run.config.get("online.delay", None)
        val3 = run.config.get("online", {}).get("delay", None)
        return val1 or val2 or val3

    def get_dataset(run):
        val1 = run.config.get("data/dataset", None)
        val2 = run.config.get("data.dataset", None)
        val3 = run.config.get("data", {}).get("dataset", None)
        return val1 or val2 or val3

    #############################
    # ARRANGE THE PLOTS
    #############################
    # make subplots


    datasets = ["cloc", "cglm"]
    # datasets = ["cloc"]
    delays = [10, 50, 100]
    methods = ["naive", "ssl", "tta", "ours"]

    colors = {
        "naive": "black",
        # "ssl": "C0",
        # "tta": "C1",
        # "ours": "C2"

        # COLORBREWER
        # "ssl": "#a6cee3",
        # "tta": "#b2df8a",
        # "ours": "#fb9a99"

        # Spring pastels
        "ssl": "#b2df8a",
        "tta": "#fdbf6f",
        "ours": "#7eb0d5"
    }

    # ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#8bd3c7"]

    linestyles = [
        "-",
        "--",
        "-.",
        ":"
    ]


    plt.figure(figsize=(15, 10))
    for i, dataset in enumerate(datasets):
        for j, delay in enumerate(delays):
            print(f"Plotting {dataset} {delay}")
            ax = plt.subplot(2, 3, i*3+j+1)
            for method in methods:
                method_runs = runs[dataset][method]

                method_runs = [run for run in method_runs if get_delay(run) == delay]

                print(f"Found {len(method_runs)} runs for {method} {dataset} {delay}")
                if len(method_runs) == 0:
                    continue

                recompute_online_acc = method in ["ours", "ssl"] and dataset == "cloc"
                method_xys = {run: get_run_metrics(run, recompute_online_acc) for run in method_runs}
                # sort by the y value at the last step
                method_runs.sort(key=lambda x: method_xys[x][1][-1], reverse=True)

                method_runs = method_runs[:1]

                # plot the best run
                for k, run in enumerate(method_runs):
                    if method == "naive":
                        label = f"Naive"
                    elif method == "ours":
                        label = f"Ours"
                    else:
                        # label = f'{run.config["method"].upper()}'
                        label = f'{method.upper()}'

                    # alpha = 1. if k == 0 else 0.5
                    # linestyle = linestyles[k]
                    x, y = method_xys[run]
                    plt.plot(
                        x, y,
                        label=label,
                        color=colors[method],
                        lw=3,
                        # linestyle=linestyle,
                        # marker=markers[k],
                        # markevery=100,
                        # markersize=markersize,
                        # alpha=alpha
                        # markevery=1, markersize=10
                    )

            # make the grid dashed
            plt.grid(True, linestyle="--", which='major',axis='both',alpha=0.3)

            # only plot it for the bottom row
            if i == 1:
                plt.xlabel("Time step")

            # only plot it for the left column
            if j == 0:
                plt.ylabel("Online Accuracy")

            # plt.legend(framealpha=0.5, loc="lower right", fontsize=12)

            # set the number of ticks
            plt.locator_params(axis='x', nbins=5)
            plt.locator_params(axis='y', nbins=5)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(thousands_formatter))

            C_val = 2 if dataset == "cloc" else 8
            plt.title(f"{dataset.upper()} ($d$={delay}, {C}={C_val})")

    # plot the legend
    plt.figlegend(*ax.get_legend_handles_labels(), loc='lower center', ncol=5)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.3)

    plt.savefig(f"fig-unsupervised.pdf")
    plt.savefig(f"fig-unsupervised.png")




def plot_fig_memory_size():
    """
    Plot the memory size of the best performing runs for our method on CGLM
    for ablation studies
    """

    api = wandb.Api()
    cglm_memory_runs = []
    for cglm_sweep_id in ["2duusivr", "10axyklu"]:
        for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+cglm_sweep_id).runs):
            if (
                run.config["online"]["num_supervised"] == 8 and
                run.config["online"]["num_unsupervised"] == 1 and
                run.config["online"]["delay"] == 50 and
                run.config["online"]["supervision_source"] in ["WR"]
            ):
                cglm_memory_runs.append(run)

    # remove duplicates by memory size
    memsize_run = {
        run.config["method_kwargs"]["queue_size"]: run for run in cglm_memory_runs
    }



    # memsize_run["$2^{15.3}$"] = memsize_run.pop(40000)
    memsize_run["$40k$"] = memsize_run.pop(40000)
    memsize_run["$2^{16}$"] = memsize_run.pop(65536)
    memsize_run["$2^{17}$"] = memsize_run.pop(131072)
    memsize_run["$2^{18}$"] = memsize_run.pop(262144)
    memsize_run["$2^{19}$"] = memsize_run.pop(500000)
    # memsize_run["Infinite"] = memsize_run.pop(500000)


    # sort by last value of y
    run_xys = {run: get_run_metrics(run) for run in cglm_memory_runs}
    memsize_run = {
        k: v for k, v in sorted(
            memsize_run.items(), key=lambda item: run_xys[item[1]][1][-1], reverse=True
        )
    }

    plt.figure(figsize=(5, 5))

    colors = cm['Oranges'](np.linspace(0.2, 1, len(memsize_run)))
    for i, (memsize, run) in enumerate(memsize_run.items()):
        x, y = run_xys[run]
        plt.plot(
            x, y,
            label=memsize,
            color=colors[i],
            lw=3,
        )
    # set the number of ticks
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    # plt.xaxis.set_major_formatter(mticker.FuncFormatter(thousands_formatter))
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(thousands_formatter))

    plt.xlabel("Time step")
    plt.ylabel("Online Accuracy")

    plt.legend(framealpha=0.5, loc="lower right", fontsize=19)

    plt.title(f"Memory Size")
    plt.grid(True, linestyle="--", which='major',axis='both',alpha=0.3)

    plt.tight_layout()

    plt.savefig(f"fig-memory-size.pdf")
    plt.savefig(f"fig-memory-size.png")



def plot_fig_sampling_strategy():
    """
    Plot the various sampling strategies for ablation on CGLM

    (here by strategy we mean the supervision source)
    """
    api = wandb.Api()
    runs = []
    for cglm_sweep_id in ["ifjbzy4c"]:
        for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+cglm_sweep_id).runs):
            if (
                run.config["online"]["num_supervised"] == 4 and
                run.config["online"]["num_unsupervised"] == 1 and
                run.config["online"]["delay"] == 100
            ):
                runs.append(run)


    source_run = {
        run.config["online"]["supervision_source"]: run for run in runs
    }

    # sort by last value of y
    run_xys = {run: get_run_metrics(run) for run in runs}
    source_run = {
        k: v for k, v in sorted(
            source_run.items(), key=lambda item: run_xys[item[1]][1][-1], reverse=True
        )
    }

    plt.figure(figsize=(5, 5))

    colors = ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#8bd3c7"]

    max_x = 0
    last_ys = []
    for i, (source, run) in enumerate(source_run.items()):
        x, y = run_xys[run]
        plt.plot(
            x, y,
            label=source,
            color=colors[i],
            lw=3,
        )

        last_x = x[-1]
        last_y = y[-1]
        last_ys.append(last_y)
        max_x = max(max_x, last_x)


    plt.xlim(-0.05*max_x, max_x*1.17)
    # plot the annotations for the last value
    # with taking into consideration the overlapping values
    from scipy.optimize import minimize
    import numpy as np

    def objective(x, original_positions, widths):
        # Objective function to minimize the total "cost" (distance from original position)
        cost = np.sum((x - original_positions)**2)

        # Calculate overlaps and add to cost if there are overlaps
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                overlap = widths[i] / 2 + widths[j] / 2 - np.abs(x[i] - x[j])
                if overlap > 0:
                    cost += 10**6 * overlap ** 2  # Large penalty for overlaps

        return cost

    def correct_positions(original_positions, widths):
        if len(original_positions) != len(widths):
            return "Lengths of original_positions and widths should be the same."

        initial_guess = np.array(original_positions)
        result = minimize(objective, initial_guess, args=(original_positions, widths), method='L-BFGS-B')

        return result.x.tolist()

    # correct the positions
    corrected_last_ys = correct_positions(last_ys, [1.]*len(last_ys))

    for i, last_y in enumerate(corrected_last_ys):
        plt.annotate(
            f"{last_y:.1f}",
            (max_x*1.02, last_y),
            fontsize=13,
            color=colors[i],
            va="center",
        )

    # remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # set the number of ticks
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(thousands_formatter))

    plt.xlabel("Time step")
    plt.ylabel("Online Accuracy")

    # plt.legend(framealpha=0.5, loc="lower center", fontsize=15, ncol=2)
    plt.legend(framealpha=0.5, loc="lower right", fontsize=15, ncol=2)

    plt.title(f"Sampling Strategy")
    plt.grid(True, linestyle="--", which='major',axis='both',alpha=0.3)

    plt.tight_layout()

    plt.savefig(f"fig-sampling-strategy.pdf")
    plt.savefig(f"fig-sampling-strategy.png")



def plot_fig_increased_compute():
    """
    Plot the various sampling strategies for ablation on CGLM

    (here by strategy we mean the supervision source)
    """
    api = wandb.Api()
    runs = []
    for cglm_sweep_id in ["10axyklu"]:
        for run in tqdm.tqdm(api.sweep("[author1]/onlineCL-cs1/"+cglm_sweep_id).runs):
            if (
                run.config["online"]["num_unsupervised"] == 1 and
                # run.config["online"]["delay"] == 100 and
                run.config["online"]["delay"] == 50 and
                run.config["method_kwargs"]["queue_size"] == 500000
            ):
                runs.append(run)


    compute_run = {
        run.config["online"]["num_supervised"]: run for run in runs
    }

    # sort by compute
    run_xys = {run: get_run_metrics(run) for run in runs}
    compute_run = {k: v for k, v in sorted(compute_run.items(), key=lambda item: item[0], reverse=True)}

    plt.figure(figsize=(5, 5))
    colors = cm['Oranges'](np.linspace(0.2, 1, len(compute_run)))

    for i, (compute, run) in enumerate(compute_run.items()):
        x, y = run_xys[run]
        plt.plot(
            x, y,
            label=compute,
            color=colors[i],
            lw=3,
        )
    # set the number of ticks
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(thousands_formatter))

    plt.xlabel("Time step")
    plt.ylabel("Online Accuracy")

    plt.legend(framealpha=0.5, loc="lower right", fontsize=19)

    plt.title(f"Compute Budget {C}")
    plt.grid(True, linestyle="--", which='major',axis='both',alpha=0.3)

    plt.tight_layout()

    plt.savefig(f"fig-increased-compute.pdf")
    plt.savefig(f"fig-increased-compute.png")



###########################
# MAIN
###########################
# plot_unsupervised()
# plot_delay_ablation("a")
# plot_delay_ablation("b")

###########################
# ABLATION SECTION
###########################
plot_fig_sampling_strategy()
# plot_fig_memory_size()
# plot_fig_increased_compute()

###########################
# MISCALLANEOUS
###########################
# plot_fig_compute("CLOC")
# plot_fig_compute("CGLM")

# plot_fig_fair_compute()

# plot_fig_scale("a")
# plot_fig_scale("b")
# plot_fig_scale("c")


# plot_fig_compute_memory("a")
# plot_fig_compute_memory("b")

# plot_fig_compute("a")
# plot_fig_compute("b")
# plot_fig_compute("c")



# plot_ssl_breakdown("a")
# plot_ssl_breakdown("b")
# plot_ssl_breakdown("c")
# plot_ssl_breakdown("d")



# plot_fig3("SSL")
# plot_fig3("TTA")