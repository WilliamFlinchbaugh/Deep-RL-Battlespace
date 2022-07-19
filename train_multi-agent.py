"""Simple example of setting up a multi-agent policy mapping.
Control the number of agents and policies via --num-agents and --num-policies.
This works with hundreds of agents and policies, but note that initializing
many TF policies will take some time.
Also, TF evals might slow down with large numbers of policies. To debug TF
execution, set the TF_TIMELINE_DIR environment variable.
"""

import argparse
import os
import random

import ray
from ray import tune
import battle_v1
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.agents.ppo import PPOTFPolicy


if __name__ == "__main__":
    n_agents = 2

    def env_creator(args):
        return PettingZooEnv(battle_v1.env(**args))

    env = env_creator({})
    register_env("battle", env_creator)

    ray.init(num_cpus=16)

    config = {
        'n_agents': n_agents, # Number of planes on each team
        'show': False, # Show visuals
        'hit_base_reward': 10, # Reward value for hitting enemy base
        'hit_plane_reward': 2, # Reward value for hitting enemy plane
        'miss_punishment': 0, # Punishment value for missing a shot
        'lose_punishment': -3, # Punishment value for losing the game
        'die_punishment': -3, # Punishment value for a plane dying
        'fps': 10 # Framerate that the visuals run at
    }

    single_dummy_env = env_creator(config)
    obs_space = single_dummy_env.observation_space
    act_space = single_dummy_env.action_space

    policies = {
        "red_policy": (
            PPOTFPolicy,
            obs_space,
            act_space,
            {},
        ),
        "blue_policy": (
            PPOTFPolicy,
            obs_space,
            act_space,
            {},
        ),
    }

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if int(agent_id[-1]) < n_agents:
            return "red_policy"
        return "blue_policy"

    config = {
        "env": "battle",
        "env_config": {
            'n_agents': n_agents, # Number of planes on each team
            'show': False, # Show visuals
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
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "framework": "tf2",
    }

    stop={"timesteps_total": 1000000}

    results = tune.run("PPO", stop=stop, config=config, verbose=1)

    ray.shutdown()