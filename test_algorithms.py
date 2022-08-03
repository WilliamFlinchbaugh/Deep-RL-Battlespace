import battle_env
from algorithms import eval_dqn, eval_dueling_ddqn, eval_ppo

# Implemented algorithms: dqn, dueling_ddqn, ppo
ALGORITHM = 'dueling_ddqn'

algorithms = {
    'dqn':eval_dqn,
    'dueling_ddqn':eval_dueling_ddqn,
    'ppo':eval_ppo
}

algorithm = algorithms[ALGORITHM]

cf = {
    'n_agents': 2, # Number of planes on each team
    'show': True, # Show visuals
    'hit_base_reward': 1000, # Reward value for hitting enemy base
    'hit_plane_reward': 50, # Reward value for hitting enemy plane
    'miss_punishment': -2, # Punishment value for missing a shot
    'die_punishment': 0, # Punishment value for a plane dying
    'fps': 20 # Framerate that the visuals run at
}

env = battle_env.parallel_env(**cf)

algorithm.train(env=env, eval_games=10)