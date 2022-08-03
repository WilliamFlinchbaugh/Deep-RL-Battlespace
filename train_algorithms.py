import train_dqn
import train_dueling_ddqn
import train_ppo

# Implemented algorithms: dqn, dueling_ddqn
ALGORITHM = 'ppo'

algorithms = {
    'dqn':train_dqn,
    'dueling_ddqn':train_dueling_ddqn,
    'ppo':train_ppo
}

algorithm = algorithms[ALGORITHM]

cf = {
    'n_agents': 2, # Number of planes on each team
    'show': False, # Show visuals
    'hit_base_reward': 1000, # Reward value for hitting enemy base
    'hit_plane_reward': 50, # Reward value for hitting enemy plane
    'miss_punishment': -2, # Punishment value for missing a shot
    'die_punishment': 0, # Punishment value for a plane dying
    'fps': 20 # Framerate that the visuals run at
}

algorithm.train(config=cf, n_games=30000)