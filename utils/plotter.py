import matplotlib.pyplot as plt
import numpy as np

def plot_scores(score_dict, filename):
    fig, ax = plt.subplots()
    red_y = np.array(score_dict['red'], dtype=np.float64)
    blue_y = np.array(score_dict['blue'], dtype=np.float64)

    red_avg = []
    blue_avg = []
    window = 1000
    x = np.arange(0, len(red_y) - window + 1)
    for i in range(len(red_y) - window + 1):
        red_avg.append(np.mean(red_y[i:i+window]))
        blue_avg.append(np.mean(blue_y[i:i+window]))

    # Plot raw averages
    plt.plot(x, red_avg, color='red', alpha=0.3)
    plt.plot(x, blue_avg, color='blue', alpha=0.3)

    # Smooth lines and plot
    red_smooth = []
    blue_smooth = []
    window = 100000
    x = np.arange(0, len(red_y) - window + 1)
    for i in range(len(red_y) - window + 1):
        red_smooth.append(np.mean(red_y[i:i+window]))
        blue_smooth.append(np.mean(blue_y[i:i+window]))

    # Plot smooth averages
    plt.plot(x, red_smooth, color='red', label='Red Team')
    plt.plot(x, blue_smooth, color='blue', label='Blue Team')

    ax.set_title('Average score over time')
    ax.set_xlabel('Number of games played')
    ax.set_ylabel('Score')
    ax.grid()
    ax.legend()
    fig.savefig(filename)

if __name__ == '__main__':
    model_name = input("Which model to plot?: ")
    scores_file = f'../models/{model_name}/scores.json'
    scores = {}
    with open(scores_file, 'r') as f:
        scores = json.load(f)
    plot_scores(scores, f'../models/{model_name}/plotted_scores.svg')