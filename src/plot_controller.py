import matplotlib.pyplot as plt
import numpy as np


class plot_controller:

    def __init__(self):
        self.fig, self.ax = plt.subplots()


def plot_costs(costs):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title('Costs on sample')
    ax.plot(costs)
    fig.show()
