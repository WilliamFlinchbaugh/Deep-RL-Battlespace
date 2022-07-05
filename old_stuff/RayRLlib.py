import gym
import envs
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
ray.init()

env_name = 'BattlespaceEnv-v0'

lr = 1e-4
gamma = 0.995
gae_lambda =  0.95
clip_param =  0.2

def env_creator(env_name):
    if env_name == 'BattlespaceEnv-v0':
        from envs.battlespace_env_dir.battlespace_env import BattlespaceEnv as env
    else:
        raise NotImplementedError
    return env

config = {
    'env': env_creator(env_name)
}

stop = {
    'timesteps_total': 10000
}

results = tune.run(
    'PPO', # Specify the algorithm to train
    config=config,
    stop=stop
)

