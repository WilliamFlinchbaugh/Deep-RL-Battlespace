"""Simple example of setting up a multi-agent policy mapping.
Control the number of agents and policies via --num-agents and --num-policies.
This works with hundreds of agents and policies, but note that initializing
many TF policies will take some time.
Also, TF evals might slow down with large numbers of policies. To debug TF
execution, set the TF_TIMELINE_DIR environment variable.
"""

import os
import ray
from ray import tune
import battle_v1
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

n_agents = 3

def env_creator(args):
    env = battle_v1.env(**args)
    return env

register_env("BattleEnvironment", lambda config: PettingZooEnv(env_creator(config)))

config = {
    "env": "BattleEnvironment",
    "env_config": {
        'n_agents': n_agents, # Number of planes on each team
        'show': True, # Show visuals
        'hit_base_reward': 10, # Reward value for hitting enemy base
        'hit_plane_reward': 2, # Reward value for hitting enemy plane
        'miss_punishment': 0, # Punishment value for missing a shot
        'lose_punishment': -3, # Punishment value for losing the game
        'die_punishment': -3, # Punishment value for a plane dying
        'fps': 60 # Framerate that the visuals run at
    },
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    "num_workers": 1,
    "framework": "tf2",
}

ray.init(num_cpus=16)

stop={"timesteps_total": 1000000}

results = tune.run("PPO", stop=stop, config=config, verbose=1)

ray.shutdown()