import battle_v1
import supersuit as ss
from stable_baselines3 import PPO

cf = {
    'n_agents': 2, # Number of planes on each team
    'show': True, # Show visuals
    'hit_base_reward': 1, # Reward value for hitting enemy base
    'hit_plane_reward': 1, # Reward value for hitting enemy plane
    'miss_punishment': 0, # Punishment value for missing a shot
    'die_punishment': 0, # Punishment value for a plane dying
    'fps': 15, # Framerate that the visuals run at
    'force_discrete_action': False
}

env = battle_v1.parallel_env(**cf)
env = ss.black_death_v3(env)
model = PPO.load("models/6/final_model")

for _ in range(5):
    observations = env.reset()
    actions = {}
    while not env.env.env_done:
        for agent in env.agents:
            actions[agent], _state = model.predict(observations[agent])
        observations, rewards, dones, infos = env.step(actions)
