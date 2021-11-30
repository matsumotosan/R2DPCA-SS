import numpy as np
import matplotlib.pyplot as plt

# TODO : Helper to plot singular value profiles
# TODO : Helper to plot explained variance ratio
# TODO : Helper to plot comparison of occluded images and their reconstructions with different methods

def plot_singular_values(s, filename, title):
    """Helper function for plotting singular values
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, linewidth=1, linestyle="--", color='k', alpha=0.1)
    ax.tick_params(which="both", direction='in', bottom=True, top=True, left=True, right=True)
    ax.plot(np.arange(len(s)) + 1, s)
    ax.set_title(title)
    ax.set_xlabel("Singular Value Index")
    ax.set_xlim(0, len(s) + 1)
    ax.set_ylabel("Singular Value")
    # ax.set_yscale('log')
    # ax.set_ylim()
    plt.savefig("../figures/" + filename, bbox_inches="tight", pad_inches=0.2)


def plot_explained_variance_ratio(evr, filename, title):
    """Helper function for plotting explained variance ratio
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, linewidth=1, linestyle="--", color='k', alpha=0.1)
    ax.tick_params(which="both", direction='in', bottom=True, top=True, left=True, right=True)
    ax.plot(np.arange(len(evr)) + 1, evr)
    ax.set_title(title)
    ax.set_xlabel("Singular Value Index")
    ax.set_xlim(0, len(evr) + 1)
    ax.set_ylabel("Explained Variance Ratio")
    # ax.set_yscale('log')
    # ax.set_ylim()
    plt.savefig("../figures/" + filename, bbox_inches="tight", pad_inches=0.1)