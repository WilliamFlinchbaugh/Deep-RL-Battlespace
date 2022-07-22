import battle_v1

cf = {
    'n_agents': 3, # Number of planes on each team
    'show': False, # Show visuals
    'hit_base_reward': 10, # Reward value for hitting enemy base
    'hit_plane_reward': 2, # Reward value for hitting enemy plane
    'miss_punishment': 0, # Punishment value for missing a shot
    'die_punishment': -5, # Punishment value for a plane dying
    'fps': 15 # Framerate that the visuals run at
}

env = battle_v1.env(**cf)

# Random choice evaluation
env.reset()
for agent in env.agent_iter():
    if env.dones[agent]:
        env.step(None)
        continue
    observation, reward, done, info = env.last()
    print(len(observation))
    action = env.action_space(agent).sample()
    env.step(action)
env.close()
