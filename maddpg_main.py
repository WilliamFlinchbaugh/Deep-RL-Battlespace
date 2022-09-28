import envs.battle_env as battle_env
from maddpg.team import Team
import numpy as np

def merge_dicts(dict1, dict2):
    dict2.update(dict1)
    return dict2

def main():
    env_config = {
        'n_agents': 2, # Number of planes on each team
        'show': False, # Show visuals
        'hit_base_reward': 100, # Reward value for hitting enemy base
        'hit_plane_reward': 20, # Reward value for hitting enemy plane
        'miss_punishment': 0, # Punishment value for missing a shot
        'die_punishment': 0, # Punishment value for a plane dying
        'fps': 120, # Framerate that the visuals run at
        'continuous_input': True
    }

    env = battle_env.parallel_env(**env_config)
    red_agent_list = env.possible_red
    blue_agent_list = env.possible_blue
    obs_len = env.observation_space(red_agent_list[0]).shape[0]
    critic_dims = obs_len * env.n_agents
    red_team = Team(red_agent_list, obs_len, env.n_actions, critic_dims)
    blue_team = Team(blue_agent_list, obs_len, env.n_actions, critic_dims)

    PRINT_INTERVAL = 1
    SAVE_INTERVAL = 20
    LEARN_INTERVAL = 50
    N_GAMES = 10000
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
            avg_red = np.mean(red_scores[-100:])
            avg_blue = np.mean(blue_scores[-100:])
            print(f"---Episode {i}---\nRed Score: {red_score}\nBlue Score: {blue_score}\n")
            # print(f"---Episode {i}---\nAvg Red Score: {avg_red:.2f}\nAvg Blue Score: {avg_blue:.2f}\n")

if __name__ == '__main__':
    main()