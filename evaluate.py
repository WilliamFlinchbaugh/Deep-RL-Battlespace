import envs.battle_env as battle_env
import instinct.team as instinct
import maddpg.team as maddpg
import matplotlib.pyplot as plt
import json
import os
import shutil
import sys

def merge_dicts(dict1, dict2):
    dict2.update(dict1)
    return dict2

def main():
    model_name = input('Enter model name: ')
    FOLDER = f'models/{model_name}'
    if not os.path.exists(FOLDER):
        print('Model does not exist')
        sys.exit()

    params = {}
    env_config = {}
        
    # Load params and env_config and scores
    with open(f'{FOLDER}/params.json', 'r') as f:
        params = json.load(f)
    with open(f'{FOLDER}/cf.json', 'r') as f:
        env_config = json.load(f)

    env_config['show'] = False

    env = battle_env.parallel_env(**env_config)
    red_agent_list = env.possible_red
    blue_agent_list = env.possible_blue

    obs_len = env.observation_space(red_agent_list[0]).shape[0]
    critic_dims = obs_len * env.n_agents

    # Red team is the maddpg team
    red_team = maddpg.Team(red_agent_list, obs_len, env.n_actions, critic_dims, params['fc1_dims'], params['fc2_dims'], params['buffer_size'], params['batch_size'], params['gamma'], params['lr'], FOLDER)
    red_team.load_models()

    # Blue team is the instinct team
    blue_team = instinct.Team(blue_agent_list, red_agent_list, env)

    wins = {
        "red": 0,
        "blue": 0,
        "tie": 0
    }

    for _ in range(10000):
        observations = env.reset()

        red_obs = {}
        blue_obs = {}
        red_obs_ = {}
        blue_obs_ = {}

        for agent in red_agent_list:
            red_obs[agent] = observations[agent]
        for agent in blue_agent_list:
            blue_obs[agent] = observations[agent]

        observations = env.reset()
        actions = {}

        while not env.env_done:
            actions = merge_dicts(red_team.choose_actions(red_obs), blue_team.choose_actions(blue_obs))
            observations_, rewards, dones, _ = env.step(actions)
            for agent in red_agent_list:
                red_obs_[agent] = observations_[agent]
            for agent in blue_agent_list:
                blue_obs_[agent] = observations_[agent]
            red_obs = red_obs_
            blue_obs = blue_obs_
        
        wins[env.winner] += 1

    print(wins)
    fig, ax = plt.subplots()
    colors = ['red', 'blue', 'grey']
    labels = ['Red Team', 'Blue Team', 'Tie']
    ax.pie(wins.values(), labels=labels, autopct='%1.1f%%', colors=colors)
    ax.axis('equal')
    ax.set_title('Winrates Over 10000 Games')
    plt.show()

if __name__ == '__main__':
    main()