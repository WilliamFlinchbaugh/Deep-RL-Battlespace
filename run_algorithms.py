import battle_env
from algorithms import run_dueling_ddqn, run_ppo, run_dqn

def main():
    # Implemented algorithms: dqn, dueling_ddqn, ppo
    ALGORITHM = 'ppo'

    algorithms = {
        'dqn':run_dqn,
        'dueling_ddqn':run_dueling_ddqn,
        'ppo':run_ppo
    }

    algorithm = algorithms[ALGORITHM]

    env_config = {
        'n_agents': 2, # Number of planes on each team
        'show': False, # Show visuals
        'hit_base_reward': 100, # Reward value for hitting enemy base
        'hit_plane_reward': 10, # Reward value for hitting enemy plane
        'miss_punishment': -2, # Punishment value for missing a shot
        'die_punishment': 0, # Punishment value for a plane dying
        'fps': 20 # Framerate that the visuals run at
    }

    env = battle_env.parallel_env(**env_config)

    algorithm.train(env=env, env_config=env_config, n_games=30000)
    algorithm.evaluate(env=env, model_path='models/ppo_1', eval_games=10)

if __name__ == '__main__':
    main()