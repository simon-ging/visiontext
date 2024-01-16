import seaborn as sns
from matplotlib import pyplot as plt


def plot_hist(data, bins=None, density=False, title="untitled", figsize=None, show=False):
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.hist(data, density=density, bins=bins)
    sns.kdeplot(data, bw_adjust=1)
    plt.grid()
    if show:
        plt.show()
