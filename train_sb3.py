import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import battle_v1
import os
from pprint import pprint
import timeit

start=timeit.default_timer()

for i in range(1, 100):
    if not os.path.exists(f'models/{i}'):
        FOLDER =  f'models/{i}'
        os.makedirs(FOLDER)
        break

# ---------- CALLBACK ----------
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, '{}'.format(self.n_calls))
            self.model.save(model_path)
        return True

CHECKPOINT_DIR = f'{FOLDER}/checkpoints'
save_freq = 100000

cf = {
    'n_agents': 2, # Number of planes on each team
    'show': False, # Show visuals
    'hit_base_reward': 2, # Reward value for hitting enemy base
    'hit_plane_reward': 1, # Reward value for hitting enemy plane
    'miss_punishment': 0, # Punishment value for missing a shot
    'die_punishment': 0, # Punishment value for a plane dying
    'fps': 15 # Framerate that the visuals run at
}

timesteps = 5000000
saved_timesteps = timesteps // save_freq * save_freq
file = open(f"{FOLDER}/results.txt", 'a')
print("Timesteps:", saved_timesteps, file=file)
pprint(cf, stream=file)

env = battle_v1.parallel_env(**cf)
env = ss.black_death_v3(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

callback = TrainAndLoggingCallback(check_freq=save_freq, save_path=CHECKPOINT_DIR)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="logs")
model.learn(total_timesteps=saved_timesteps, callback=callback)
model.save(f"{FOLDER}/final_model")
del model

stop = timeit.default_timer()
time = start - stop
print(f'Runtime: {time//3600}::{time%3600//60}::{time%3600%60}')
print(f'Runtime: {time//3600}::{time%3600//60}::{time%3600%60}', file=file)

# Turn on visuals and show 5 games
cf['show'] = True
env = battle_v1.parallel_env(**cf)
env = ss.black_death_v3(env)
model = PPO.load(f"{FOLDER}/final_model.zip")

for _ in range(5):
    observations = env.reset()
    actions = {}
    while not env.env.env_done:
        for agent in env.agents:
            actions[agent], _state = model.predict(observations[agent])
        observations, rewards, dones, infos = env.step(actions)