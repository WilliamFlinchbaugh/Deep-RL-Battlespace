from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from battle_environment import BattleEnvironment
import os

for i in range(100):
    if not os.path.exists(f'models/PPO_{i}'):
        FOLDER =  f'models/PPO_{i}'
        LOG_DIR = f'{FOLDER}/logs'
        os.makedirs(FOLDER)
        os.makedirs(LOG_DIR)
        break
# FOLDER = 'models/PPO_13'

# ---------- CONFIG ----------
cf = {
    'show_viz': False,
    'hit_base_reward': 100,
    'hit_plane_reward': 30,
    'miss_punishment': 0,
    'too_long_punishment': 0,
    'closer_to_base_reward': 0,
    'closer_to_plane_reward': 0,
    'lose_punishment': 0,
    'fps': 20
}

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

CHECKPOINT_DIR = f'{FOLDER}/train'
save_freq = 10000
timesteps = 1000000
saved_timesteps = timesteps // save_freq * save_freq

# ---------- TRAINING ----------
callback = TrainAndLoggingCallback(check_freq=save_freq, save_path=CHECKPOINT_DIR)
env = BattleEnvironment(show=cf['show_viz'], hit_base_reward=cf['hit_base_reward'], hit_plane_reward=cf['hit_plane_reward'], miss_punishment=cf['miss_punishment'], 
    too_long_punishment=cf['too_long_punishment'], closer_to_base_reward=cf['closer_to_base_reward'], closer_to_plane_reward=cf['closer_to_plane_reward'], lose_punishment=cf['lose_punishment'])
model = PPO('MlpPolicy', env, tensorboard_log=LOG_DIR, verbose=1)
model.learn(total_timesteps=timesteps, callback=callback)
del model

# ---------- EVALUATION WITHOUT VISUALS ----------
model = PPO.load(f'{CHECKPOINT_DIR}/{saved_timesteps}')
env = BattleEnvironment(show=cf['show_viz'], hit_base_reward=cf['hit_base_reward'], hit_plane_reward=cf['hit_plane_reward'], miss_punishment=cf['miss_punishment'], 
    too_long_punishment=cf['too_long_punishment'], closer_to_base_reward=cf['closer_to_base_reward'], closer_to_plane_reward=cf['closer_to_plane_reward'], lose_punishment=cf['lose_punishment'])
episodes = 1000
for episode in range(episodes): # Evaluates the model n times
    state = env.reset()
    score = 0
    while not env.done:
        env.render()
        action, _states = model.predict(state)
        n_state, reward, done, info = env.step(action)
        score += reward
    
    if (episode+1) % 10 == 0:
        print(f"episode {episode+1}")

print(env.wins())
with open(f'{FOLDER}/results.txt', 'w') as f:
    print(f"---CONSTANTS---\nhit base:{cf['hit_base_reward']}\nhit plane:{cf['hit_plane_reward']}\nmiss:{cf['miss_punishment']}\ntoo long:{cf['too_long_punishment']}\ncloser to base:{cf['closer_to_base_reward']}\ncloser to plane:{cf['closer_to_plane_reward']}\nlose:{cf['lose_punishment']}\n", file=f)
    print(f"---EVALUATION---\n{env.wins()}\n", file=f)

# ---------- EVALUATION WITH VISUALS ----------
model = PPO.load(f'{CHECKPOINT_DIR}/{saved_timesteps}')
cf['show_viz'] = True
env = BattleEnvironment(show=cf['show_viz'], hit_base_reward=cf['hit_base_reward'], hit_plane_reward=cf['hit_plane_reward'], miss_punishment=cf['miss_punishment'], 
    too_long_punishment=cf['too_long_punishment'], closer_to_base_reward=cf['closer_to_base_reward'], closer_to_plane_reward=cf['closer_to_plane_reward'], lose_punishment=cf['lose_punishment'], fps=cf['fps'])
episodes = 100
for episode in range(episodes): # Evaluates the model n times
    state = env.reset()
    score = 0
    while not env.done:
        env.render()
        action, _states = model.predict(state)
        n_state, reward, done, info = env.step(action)
        score += reward
    
    if (episode+1) % 10 == 0:
        print(f"episode {episode+1}")

print(env.wins())

with open(f'{FOLDER}/results.txt', 'w') as f:
    print(f"---VIZ EVALUATION---\n{env.wins()}\n", file=f)

