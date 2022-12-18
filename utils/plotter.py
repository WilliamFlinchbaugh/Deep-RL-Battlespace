import matplotlib.pyplot as plt
import numpy as np
import os
import json

def plot_scores(score_dict, filename):
    fig, ax = plt.subplots()

    print("Gathering data...")
    red_y = np.array(score_dict['red'], dtype=np.float64)
    blue_y = np.array(score_dict['blue'], dtype=np.float64)

    red_avg = []
    blue_avg = []
    window = 1000
    x = np.arange(0, len(red_y) - window + 1)
    print("Gathering averages over 1000 games...")
    for i in range(len(red_y) - window + 1):
        red_avg.append(np.mean(red_y[i:i+window]))
        blue_avg.append(np.mean(blue_y[i:i+window]))

    # Plot raw averages
    print("Plotting averages over 1000 games...")
    plt.plot(x, red_avg, color='red', alpha=1, label='Red Team')
    plt.plot(x, blue_avg, color='blue', alpha=1, label='Blue Team')

    ax.set_title('Average score over time')
    ax.set_xlabel('Number of games played')
    ax.set_ylabel('Score')
    ax.grid()
    ax.legend()
    fig.savefig(filename)
    print("Finished plotting scores. Saved to", filename)

if __name__ == '__main__':
    model_name = input("Which model to plot?: ")
    scores_file = f'models/{model_name}/scores.json'
    scores = {}
    with open(scores_file, 'r') as f:
        scores = json.load(f)
    plot_scores(scores, f'models/{model_name}/plotted_scores.svg')