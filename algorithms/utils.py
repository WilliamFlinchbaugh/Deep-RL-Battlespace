from curses import raw
import matplotlib.pyplot as plt
import numpy as np

def plot_data(reward_dict, filename):
    fig, ax = plt.subplots()

    for id, scores in reward_dict.items():
        games_played = len(scores)
        running_avg = np.empty(games_played)
        for t in range(games_played):
            running_avg[t] = np.mean(scores[max(0, t-5):(t+1)])
        x = [i for i in range(games_played)]
        ax.plot(x, running_avg, label=id)

    ax.set_title('Mean Reward for Each Agent')
    ax.set_xlabel('Number of games played')
    ax.set_ylabel('Mean reward')
    ax.legend()
    fig.savefig(filename)



