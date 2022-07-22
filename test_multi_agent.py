import battle_v1

# n_agents=1, show=False, hit_base_reward=10, hit_plane_reward=2, miss_punishment=0, lose_punishment=-3, die_punishment=-3, fps=20
cf = {
    'n_agents': 3, # Number of planes on each team
    'show': True, # Show visuals
    'hit_base_reward': 10, # Reward value for hitting enemy base
    'hit_plane_reward': 2, # Reward value for hitting enemy plane
    'miss_punishment': 0, # Punishment value for missing a shot
    'lose_punishment': -3, # Punishment value for losing the game
    'die_punishment': -3, # Punishment value for a plane dying
    'fps': 20 # Framerate that the visuals run at
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
