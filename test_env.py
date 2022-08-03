import battle_env

cf = {
    'n_agents': 2, # Number of planes on each team
    'show': True, # Show visuals
    'hit_base_reward': 1, # Reward value for hitting enemy base
    'hit_plane_reward': 1, # Reward value for hitting enemy plane
    'miss_punishment': 0, # Punishment value for missing a shot
    'die_punishment': 0, # Punishment value for a plane dying
    'fps': 50 # Framerate that the visuals run at
}

env = battle_env.parallel_env(**cf)

env.start_recording('test.mp4')
for _ in range(5):
    observations = env.reset()
    actions = {}
    while not env.env_done:
        for agent in env.agents:
            actions[agent] = env.action_space(agent).sample()
        observations, rewards, dones, infos = env.step(actions)

env.export_video()
