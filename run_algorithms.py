import battle_env
from algorithms import run_dueling_ddqn, run_ppo, run_dqn

# Implemented algorithms: dqn, dueling_ddqn, ppo
ALGORITHM = 'ppo'

algorithms = {
    'dqn':run_dqn,
    'dueling_ddqn':run_dueling_ddqn,
    'ppo':run_ppo
}

algorithm = algorithms[ALGORITHM]

cf = {
    'n_agents': 2, # Number of planes on each team
    'show': True, # Show visuals
    'hit_base_reward': 100, # Reward value for hitting enemy base
    'hit_plane_reward': 10, # Reward value for hitting enemy plane
    'miss_punishment': -2, # Punishment value for missing a shot
    'die_punishment': 0, # Punishment value for a plane dying
    'fps': 20 # Framerate that the visuals run at
}

env = battle_env.parallel_env(**cf)

algorithm.train(env=env, n_games=20000)
algorithm.evaluate(env=env, model_path='models/ppo_5', eval_games=10)