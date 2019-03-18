import matplotlib.pyplot as plt
import numpy as np


class plot_controller:

    def __init__(self):
        self.fig, self.ax = plt.subplots()


def plot_costs(costs):
    costs_flat = np.array(costs).reshape(-1)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title('Costs on sample')
    ax.plot(costs_flat)
    fig.show()
