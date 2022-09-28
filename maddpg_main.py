import envs.battle_env as battle_env
from maddpg.team import Team
import numpy as np
import os

def merge_dicts(dict1, dict2):
    dict2.update(dict1)
    return dict2

def main():
    # Create a new folder for the model
    for i in range(1, 100):
        if not os.path.exists(f'models/maddpg_{i}'):
            FOLDER = f'models/maddpg_{i}'
            os.makedirs(FOLDER)
            os.makedirs(f'{FOLDER}/training_vids')
            break

    env_config = {
        'n_agents': 2, # Number of planes on each team
        'show': False, # Show visuals
        'hit_base_reward': 100, # Reward value for hitting enemy base
        'hit_plane_reward': 30, # Reward value for hitting enemy plane
        'miss_punishment': -1, # Punishment value for missing a shot
        'die_punishment': 0, # Punishment value for a plane dying
        'fps': 20, # Framerate that the visuals run at
        'continuous_input': True
    }

    env = battle_env.parallel_env(**env_config)
    red_agent_list = env.possible_red
    blue_agent_list = env.possible_blue
    obs_len = env.observation_space(red_agent_list[0]).shape[0]
    critic_dims = obs_len * env.n_agents
    red_team = Team(red_agent_list, obs_len, env.n_actions, critic_dims, chkpt_dir=FOLDER)
    blue_team = Team(blue_agent_list, obs_len, env.n_actions, critic_dims, chkpt_dir=FOLDER)

    PRINT_INTERVAL = 100
    SAVE_INTERVAL = 100
    LEARN_INTERVAL = 20
    SHOW_INTERVAL = 1000
    N_GAMES = 100000
    steps = 0
    red_scores = []
    blue_scores = []

    print("\n=====================\n| Starting Training |\n=====================\n")
    for i in range(N_GAMES):
        observations = env.reset()

        red_score = 0
        blue_score = 0
        red_obs = {}
        blue_obs = {}

        for agent in red_agent_list:
            red_obs[agent] = observations[agent]
        for agent in blue_agent_list:
            blue_obs[agent] = observations[agent]

        if i % SHOW_INTERVAL == 0 and i > 0:
            env.show = True
            env.start_recording(f'{FOLDER}/training_vids/{i}.mp4')

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
                blue_actions[agent] = actions[agent]
                blue_rewards[agent] = rewards[agent]
                blue_score += rewards[agent]
                blue_dones[agent] = dones[agent]

            red_team.memory.store_transition(red_obs, red_actions, red_rewards, red_obs_, red_dones)
            blue_team.memory.store_transition(blue_obs, blue_actions, blue_rewards, blue_obs_, blue_dones)

            if steps % LEARN_INTERVAL == 0 and steps > 0:
                red_team.learn()
                blue_team.learn()

            if steps % SAVE_INTERVAL == 0 and steps > 0:
                red_team.save_models()
                blue_team.save_models()

            red_obs = red_obs_
            blue_obs = blue_obs_

            red_scores.append(red_score)
            blue_scores.append(blue_score)
            steps += 1

        if i % PRINT_INTERVAL == 0 and i > 0:
            avg_red = np.mean(red_scores[-PRINT_INTERVAL:])
            avg_blue = np.mean(blue_scores[-PRINT_INTERVAL:])
            print(f"--------------------\nEpisode {i}\nAvg Red Score: {avg_red:.2f}\nAvg Blue Score: {avg_blue:.2f}\n--------------------\n")

if __name__ == '__main__':
    main()