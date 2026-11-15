import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

type TimeSeries = ArrayLike


def plot_timeseries(title, x_label="x", y_label="y"):
    plt.figure()
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.plot()
