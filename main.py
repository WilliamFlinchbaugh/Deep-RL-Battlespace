import envs.battle_env as battle_env
import maddpg.team as maddpg
import instinct.team as instinct
import numpy as np
import os
import datetime
from utils.utils import plot_data
import sys
import json
import shutil

hyperparams = {
    'gamma': 0.95,
    'lr': 0.001,
    'buffer_size': 400_000,
    'batch_size': 64,
    'fc1_dims': 64,
    'fc2_dims': 64,
    'init_noise': 0.4,
    'final_noise': 0.01,
    'n_explores': 30000,
    'print_interval': 100,
    'save_interval': 100,
    'learn_interval': 100,
    'render_interval': 500,
    'max_episodes': 500_000,
}

env_config = {
    'n_agents': 2, # Number of planes on each team
    'show': False, # Show visuals
    'hit_base_reward': 100, # Reward value for hitting enemy base
    'hit_plane_reward': 100, # Reward value for hitting enemy plane
    'miss_punishment': -2, # Punishment value for missing a shot
    'die_punishment': -10, # Punishment value for a plane dying
    'lose_punishment': -10, # Punishment for losing the game (The goal is to possibly defend the base)
    'fps': 20, # Framerate that the visuals run at
    'continuous_actions': False
}

def merge_dicts(dict1, dict2):
    dict2.update(dict1)
    return dict2

def save_hyperparams(path, param_dict):
    with open(path + '/hyperparams.json', 'w') as f:
        f.write(json.dumps(param_dict))

def save_config(path, config):
    with open(path + '/env_config.json', 'w') as f:
        f.write(json.dumps(config))

if __name__ == '__main__':
    # Menu
    print('1. Override a model')
    print('2. Continue training a model')
    print('3. Train a new model')
    print('4. Quit')
    choice = input('Enter a number: ')
    if choice == '1':
        model_name = input('Enter model name: ')
        model_path = f'models/{model_name}'
        if not os.path.exists(model_path):
            print('Model does not exist')
            sys.exit()
        shutil.rmtree(model_path)
        FOLDER = model_path
        os.makedirs(model_path)
        os.makedirs(f'{model_path}/training_vids')
        save_hyperparams(FOLDER, hyperparams)
        save_config(FOLDER, env_config)

    elif choice == '2':
        model_name = input('Enter model name: ')
        FOLDER = f'models/{model_name}'
        if not os.path.exists(FOLDER):
            print('Model does not exist')
            sys.exit()
        with open(f'{FOLDER}/hyperparams.json', 'r') as f:
            hyperparams = json.load(f)
        with open(f'{FOLDER}/env_config.json', 'r') as f:
            env_config = json.load(f)
        hyperparams['max_episodes'] = int(input('Enter number of games to train: '))
        save_hyperparams(FOLDER, hyperparams)
        save_config(FOLDER, env_config)
            
    elif choice == '3':
        # Create a new folder for the model
        for i in range(1, 100):
            if not os.path.exists(f'models/{i}'):
                FOLDER = f'models/{i}'
                os.makedirs(FOLDER)
                os.makedirs(f'{FOLDER}/training_vids')
                break
        save_hyperparams(FOLDER, hyperparams)
        save_config(FOLDER, env_config)
        
    env = battle_env.parallel_env(**env_config)

    red_agent_list = env.possible_red
    blue_agent_list = env.possible_blue

    obs_len = env.observation_space(red_agent_list[0]).shape[0]
    critic_dims = obs_len * env.n_agents

    red_team = maddpg.Team(red_agent_list, obs_len, env.n_actions, critic_dims, hyperparams['fc1_dims'], hyperparams['fc2_dims'], hyperparams['buffer_size'], hyperparams['batch_size'], hyperparams['gamma'], hyperparams['lr'], FOLDER)
    blue_team = instinct.Team(blue_agent_list, red_agent_list, env)

    steps = 0
    score_dict = {
        "red": [],
        "blue": []
    }

    print(f'\n{" Starting Training ":=^43}')
    start = datetime.datetime.now()

    # Training loop
    for i in range(1, hyperparams['max_episodes']+1):
        now = datetime.datetime.now()
        elapsed = now - start
        estimate = (elapsed.total_seconds() / i * hyperparams['max_episodes']) / 3600
        sys.stdout.write(f"\r{' Episode {game} | %{percent:.1f} | {estimate:.1f} Hours Left '.format(game=i, percent=i/hyperparams['max_episodes']*100, estimate=estimate):=^43}")
        observations = env.reset()

        explore_remaining = max(0, hyperparams['n_explores'] - i) / hyperparams['n_explores']
        explore_scale = hyperparams['init_noise'] + (hyperparams['init_noise'] - hyperparams['final_noise']) * explore_remaining
        explore_scale = round(explore_scale, 2)
        red_team.scale_noise(explore_scale)
        red_team.reset_noise()

        red_score = 0
        blue_score = 0
        red_obs = {}
        blue_obs = {}

        for agent in red_agent_list:
            red_obs[agent] = observations[agent]
        for agent in blue_agent_list:
            blue_obs[agent] = observations[agent]

        if i % hyperparams['render_interval'] == 0 and i > 0:
            env.show = True
            env.start_recording(f'{FOLDER}/training_vids/{i}.mp4')
            # plot_data(score_dict, FOLDER + '/scores.svg')

        elif env.show == True:
            env.export_video()
            env.show = False
            env.close()

        while not env.env_done:
            actions = merge_dicts(red_team.choose_actions(red_obs), blue_team.choose_actions(blue_obs))

            observations_, rewards, dones, _ = env.step(actions)

            red_obs_ = {}
            red_actions = {}
            red_rewards = {}
            red_dones = {}
            blue_obs_ = {}
            blue_actions = {}
            blue_rewards = {}
            blue_dones = {}

            for agent in red_agent_list:
                red_obs_[agent] = observations_[agent]
                red_actions[agent] = actions[agent]
                red_rewards[agent] = rewards[agent]
                red_score += rewards[agent]
                red_dones[agent] = dones[agent]

            for agent in blue_agent_list:
                blue_obs_[agent] = observations_[agent]
                blue_score += rewards[agent]

            red_team.memory.store_transition(red_obs, red_actions, red_rewards, red_obs_, red_dones)

            if steps % hyperparams['learn_interval'] == 0 and steps > 0:
                red_team.learn()

            if steps % hyperparams['save_interval'] == 0 and steps > 0:
                red_team.save_models()

            red_obs = red_obs_
            blue_obs = blue_obs_

            score_dict['red'].append(red_score)
            score_dict['blue'].append(blue_score)
            steps += 1

        if i % hyperparams['print_interval'] == 0:
            now = datetime.datetime.now()
            elapsed = now - start
            s = elapsed.total_seconds()
            hr, rem = divmod(s, 3600)
            min, sec = divmod(rem, 60)

            formatted_elapsed = f'{int(hr):02}:{int(min):02}:{int(sec):02}'
            formatted_time = now.strftime("%I:%M:%S %p")

            avg_red = np.mean(score_dict['red'][-hyperparams['print_interval']:])
            avg_blue = np.mean(score_dict['blue'][-hyperparams['print_interval']:])

            statement = (
                f"\n{'-'*43}\n"
                f"| {('Current Time: ' + formatted_time):<40}|\n"
                f"| {('Elapsed Time: ' + formatted_elapsed):<40}|\n"
                f"| {('Games: ' + str(i)):<40}|\n"
                f"| {('Timesteps: ' + str(steps)):<40}|\n"
                f"| {('Exploration Scale: ' + str(explore_scale)):<40}|\n"
                f"| {('Red Avg Score: ' + str(avg_red)):<40}|\n"
                f"| {('Blue Avg Score: ' + str(avg_blue)):<40}|\n"
                f"{'-'*43}\n"
            )
            print(statement)

    plot_data(score_dict, FOLDER + '/scores.svg')