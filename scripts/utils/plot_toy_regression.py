import matplotlib.pyplot as plt
import matplotlib
import numpy as np


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

def plot_toy_regression():
    pi = np.pi
    X = np.linspace(-pi, pi, 100)
    Y = np.sin(X) + np.random.normal(0, 0.1, 100)
    plt.plot(X, Y, "o", color="black", markersize=5)
    plt.xlabel("timestep")
    plt.ylabel("Regression target")
    plt.savefig("toy_regression.pdf", bbox_inches="tight")
