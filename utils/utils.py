"""
Takes a dictionary of rewards for each agent and creates a graph
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def plot_data(reward_dict, filename):
    fig, ax = plt.subplots()

    for id, scores in reward_dict.items():
        games_played = len(scores)
        running_avg = np.empty(games_played)
        for t in range(games_played):
            running_avg[t] = np.mean(scores[max(0, t-5):(t+1)])
        x = [i for i in range(games_played)]
        cubic_interploation_model = interp1d(x, running_avg, kind = "cubic")
        X_=np.linspace(min(x), max(x), 30)
        Y_=cubic_interploation_model(X_)
        ax.plot(X_, Y_, label=id)

    ax.set_title('Mean Reward for Each Agent')
    ax.set_xlabel('Number of games played')
    ax.set_ylabel('Mean reward')
    ax.grid()
    ax.legend()
    fig.savefig(filename)



