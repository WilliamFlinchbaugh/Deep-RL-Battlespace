import envs.battle_env as battle_env
import maddpg.team as maddpg
import instinct.team as instinct
import numpy as np
import os
import datetime
import sys
import json
import shutil

params = {
    'gamma': 0.95,
    'lr': 0.001,
    'buffer_size': 1_000_000,
    'batch_size': 512,
    'fc1_dims': 64,
    'fc2_dims': 64,
    'init_noise': 0.4,
    'final_noise': 0.01,
    'curr_noise': 0,
    'n_explores': 30000,
    'print_interval': 100,
    'save_interval': 1000,
    'learn_interval': 100,
    'render_interval': 500,
    'n_games': 500_000,
    'curr_game': 1
}

env_config = {
    'n_agents': 2, # Number of planes on each team
    'show': False, # Show visuals
    'hit_base_reward': 1.0, # Reward value for hitting enemy base
    'hit_plane_reward': 0.9, # Reward value for hitting enemy plane
    'miss_punishment': -0.02, # Punishment value for missing a shot
    'die_punishment': -0.03, # Punishment value for a plane dying
    'lose_punishment': -0.05, # Punishment for losing the game (The goal is to possibly defend the base)
    'fps': 20, # Framerate that the visuals run at
    'continuous_actions': False
}

score_dict = {
    "red": [],
    "blue": []
}

def merge_dicts(dict1, dict2):
    dict2.update(dict1)
    return dict2

def save_dict(path, dict):
    with open(path, 'w') as f:
        f.write(json.dumps(dict))

if __name__ == '__main__':
    
    # Menu
    print('1. Override a model')
    print('2. Continue training a model')
    print('3. Train a new model')
    print('4. Quit')
    choice = input('Enter a number: ')
    if choice == '1': # Override a model
        # Delete the model folder and create a new one
        model_name = input('Enter model name: ')
        model_path = f'models/{model_name}'
        if not os.path.exists(model_path):
            print('Model does not exist')
            sys.exit()
        shutil.rmtree(model_path)
        FOLDER = model_path
        os.makedirs(model_path)
        os.makedirs(f'{model_path}/training_vids')
        
        # Save params and env_config
        save_dict(FOLDER + '/params.json', params)
        save_dict(FOLDER + '/cf.json', env_config)

    elif choice == '2': # Continue training a model
        model_name = input('Enter model name: ')
        FOLDER = f'models/{model_name}'
        if not os.path.exists(FOLDER):
            print('Model does not exist')
            sys.exit()
            
        # Load params and env_config and scores
        with open(f'{FOLDER}/params.json', 'r') as f:
            params = json.load(f)
        with open(f'{FOLDER}/cf.json', 'r') as f:
            env_config = json.load(f)
        with open(f'{FOLDER}/scores.json', 'r') as f:
            score_dict = json.load(f)

        # Save params and env_config
        save_dict(FOLDER + '/params.json', params)
        save_dict(FOLDER + '/cf.json', env_config)
            
    elif choice == '3': # Train a new model
        # Create a new folder for the model
        for i in range(1, 100):
            if not os.path.exists(f'models/{i}'):
                FOLDER = f'models/{i}'
                os.makedirs(FOLDER)
                os.makedirs(f'{FOLDER}/training_vids')
                break
            
        # Save params and env_config
        save_dict(FOLDER + '/params.json', params)
        save_dict(FOLDER + '/cf.json', env_config)
        
    env = battle_env.parallel_env(**env_config)

    red_agent_list = env.possible_red
    blue_agent_list = env.possible_blue

    obs_len = env.observation_space(red_agent_list[0]).shape[0]
    critic_dims = obs_len * env.n_agents

    # Red team is the maddpg team
    red_team = maddpg.Team(red_agent_list, obs_len, env.n_actions, critic_dims, params['fc1_dims'], params['fc2_dims'], params['buffer_size'], params['batch_size'], params['gamma'], params['lr'], FOLDER)
    
    # Blue team is the instinct agent team
    blue_team = instinct.Team(blue_agent_list, red_agent_list, env)

    steps = 0

    print(f'\n{" Starting Training ":=^43}')
    start = datetime.datetime.now()

    start_game = params['curr_game']-1
    
    # Training loop
    for i in range(params['curr_game'], params['n_games']+1):
        params['curr_game'] = i
        
        # The continuous update of the training loop
        now = datetime.datetime.now()
        elapsed = now - start
        estimate = (elapsed.total_seconds() / (i-start_game) * (params['n_games']-i)) / 3600
        sys.stdout.write(f"\r{' Game {game} | %{percent:.1f} | {estimate:.1f} Hours Left '.format(game=i, percent=i/params['n_games']*100, estimate=estimate):=^43}") # Will overwrite the previous line
        
        observations = env.reset() # Reset the environment

        # Reset noise for exploration of maddpg
        explore_remaining = max(0, params['n_explores'] - i) / params['n_explores']
        params['curr_noise'] = params['init_noise'] + (params['init_noise'] - params['final_noise']) * explore_remaining
        params['curr_noise'] = round(params['curr_noise'], 2)
        red_team.scale_noise(params['curr_noise'])
        red_team.reset_noise()

        red_score = 0
        blue_score = 0
        red_obs = {}
        blue_obs = {}
        
        # Organize all the observations by team
        for agent in red_agent_list:
            red_obs[agent] = observations[agent]
        for agent in blue_agent_list:
            blue_obs[agent] = observations[agent]

        if i % params['render_interval'] == 0 and i > 0:
            env.show = True
            env.start_recording(f'{FOLDER}/training_vids/{i}.mp4') # Record the video of 1 game

        elif env.show == True:
            env.export_video() # Stop recording video
            env.show = False
            env.close()

        while not env.env_done:
            actions = merge_dicts(red_team.choose_actions(red_obs), blue_team.choose_actions(blue_obs)) # Put together actions from both teams

            observations_, rewards, dones, _ = env.step(actions)

            red_obs_ = {}
            red_actions = {}
            red_rewards = {}
            red_dones = {}
            blue_obs_ = {}
            blue_actions = {}
            blue_rewards = {}
            blue_dones = {}

            # Organize all every return by team
            for agent in red_agent_list:
                red_obs_[agent] = observations_[agent]
                red_actions[agent] = actions[agent]
                red_rewards[agent] = rewards[agent]
                red_score += rewards[agent]
                red_dones[agent] = dones[agent]

            for agent in blue_agent_list:
                blue_obs_[agent] = observations_[agent]
                blue_score += rewards[agent]

            # Store the transitions in the replay buffer
            red_team.memory.store_transition(red_obs, red_actions, red_rewards, red_obs_, red_dones)

            # Learn from the replay buffer
            if steps % params['learn_interval'] == 0 and steps > 0:
                red_team.learn()

            # Save the model and scores and params
            if steps % params['save_interval'] == 0 and steps > 0:
                red_team.save_models()
                save_dict(FOLDER + '/scores.json', score_dict)
                save_dict(FOLDER + '/params.json', params)

            red_obs = red_obs_
            blue_obs = blue_obs_

            # Append scores
            score_dict['red'].append(round(red_score, 3))
            score_dict['blue'].append(round(blue_score, 3))
            steps += 1

        # Print update
        if i % params['print_interval'] == 0:
            now = datetime.datetime.now()
            elapsed = now - start
            s = elapsed.total_seconds()
            hr, rem = divmod(s, 3600)
            min, sec = divmod(rem, 60)

            formatted_elapsed = f'{int(hr):02}:{int(min):02}:{int(sec):02}'
            formatted_time = now.strftime("%I:%M:%S %p")

            avg_red = round(np.mean(score_dict['red'][-params['print_interval']:]), 3)
            avg_blue = round(np.mean(score_dict['blue'][-params['print_interval']:]), 3)

            statement = (
                f"\n{'-'*43}\n"
                f"| {('Current Time: ' + formatted_time):<40}|\n"
                f"| {('Elapsed Time: ' + formatted_elapsed):<40}|\n"
                f"| {('Games: ' + str(i)):<40}|\n"
                f"| {('Timesteps: ' + str(steps)):<40}|\n"
                f"| {('Exploration Scale: ' + str(params['curr_noise'])):<40}|\n"
                f"| {('Red Avg Score: ' + str(avg_red)):<40}|\n"
                f"| {('Blue Avg Score: ' + str(avg_blue)):<40}|\n"
                f"{'-'*43}\n"
            )
            print(statement)