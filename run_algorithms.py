import envs.battle_env as battle_env
from algorithms import run_dueling_ddqn, run_ppo

def main():
    # Implemented algorithms: dueling_ddqn, ppo (both are completely decentralized)
    ALGORITHM = 'ppo'

    algorithms = {
        'dueling_ddqn':run_dueling_ddqn,
        'ppo':run_ppo
    }

    algorithm = algorithms[ALGORITHM]

    env_config = {
        'n_agents': 2, # Number of planes on each team
        'show': False, # Show visuals
        'hit_base_reward': 100, # Reward value for hitting enemy base
        'hit_plane_reward': 13, # Reward value for hitting enemy plane
        'miss_punishment': -3, # Punishment value for missing a shot
        'die_punishment': 0, # Punishment value for a plane dying
        'fps': 20 # Framerate that the visuals run at
    }

    env = battle_env.parallel_env(**env_config)

    algorithm.train(env=env, env_config=env_config, n_games=75000)
    algorithm.evaluate(env=env, model_path='models/ppo_2', eval_games=10)

if __name__ == '__main__':
    main()