import envs.battle_env as battle_env
import instinct.team as instinct

def merge_dicts(dict1, dict2):
    dict2.update(dict1)
    return dict2

def main():
    cf = {
        'n_agents': 2, # Number of planes on each team
        'show': True, # Show visuals
        'hit_base_reward': 1, # Reward value for hitting enemy base
        'hit_plane_reward': 1, # Reward value for hitting enemy plane
        'miss_punishment': 0, # Punishment value for missing a shot
        'die_punishment': 0, # Punishment value for a plane dying
        'fps': 20, # Framerate that the visuals run at
        'continuous_actions': True
    }

    env = battle_env.parallel_env(**cf)
    red_agent_list = env.possible_red
    blue_agent_list = env.possible_blue
    red_team = instinct.Team(red_agent_list, blue_agent_list, env)
    blue_team = instinct.Team(blue_agent_list, red_agent_list, env)

    # env.start_recording('test.mp4')
    for _ in range(5):
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
            observations_, _, _, _ = env.step(actions)
            for agent in red_agent_list:
                red_obs_[agent] = observations_[agent]
            for agent in blue_agent_list:
                blue_obs_[agent] = observations_[agent]
            red_obs = red_obs_
            blue_obs = blue_obs_

    # env.export_video()

if __name__ == '__main__':
    main()