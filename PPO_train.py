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
# FOLDER = 'models/PPO_1'

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
timesteps = 500000
saved_timesteps = timesteps // save_freq * save_freq

# ---------- TRAINING ----------
callback = TrainAndLoggingCallback(check_freq=save_freq, save_path=CHECKPOINT_DIR)
env = BattleEnvironment(show=False, hit_base_reward=50, hit_plane_reward=20, miss_punishment=0, too_long_punishment=0, closer_to_base_reward=0, 
    closer_to_plane_reward=0, lose_punishment=-50)
config = [env.hit_base_reward, env.hit_plane_reward, env.miss_punishment, env.too_long_punishment, env.closer_to_base_reward, env.closer_to_plane_reward, env.lose_punishment]
model = PPO('MlpPolicy', env, tensorboard_log=LOG_DIR, verbose=1)
model.learn(total_timesteps=timesteps, callback=callback)
del model

# ---------- EVALUATION WITHOUT VISUALS ----------
model = PPO.load(f'{CHECKPOINT_DIR}/{saved_timesteps}')
env = BattleEnvironment(show=False, hit_base_reward=config[0], hit_plane_reward=config[1], miss_punishment=config[2], too_long_punishment=config[3], closer_to_base_reward=config[4], 
    closer_to_plane_reward=config[5], lose_punishment=config[6])
episodes = 1000
avg = 0
for episode in range(episodes): # Evaluates the model n times
    state = env.reset()
    score = 0
    while not env.done:
        env.render()
        action, _states = model.predict(state)
        # action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
        avg += reward

    if episode % 50 == 0:
        print('# Episodes:{} Avg Score:{}'.format(episode, avg/50))
        avg = 0

print(env.wins())
with open(f'{FOLDER}/results.txt', 'w') as f:
    print(f"---CONSTANTS---\nhit base:{config[0]}\nhit plane:{config[1]}\nmiss:{config[2]}\ntoo long:{config[3]}\ncloser to base:{config[4]}\ncloser to plane:{config[5]}\nlose:{config[6]}\n", file=f)
    print(f"---EVALUATION---\n{env.wins()}\n", file=f)

# ---------- EVALUATION WITH VISUALS ----------
model = PPO.load(f'{CHECKPOINT_DIR}/{saved_timesteps}')
env = BattleEnvironment(show=True, hit_base_reward=config[0], hit_plane_reward=config[1], miss_punishment=config[2], too_long_punishment=config[3], closer_to_base_reward=config[4], 
    closer_to_plane_reward=config[5], lose_punishment=config[6], fps=30)
episodes = 100
for episode in range(episodes): # Evaluates the model n times
    state = env.reset()
    score = 0
    while not env.done:
        env.render()
        action, _states = model.predict(state)
        # action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    if episode % 10 == 0:
        print('# Episodes:{} Avg Score:{}'.format(episode+1, avg/10))
        avg = 0
print(env.wins())
with open(f'{FOLDER}/results.txt', 'w') as f:
    print(f"---VIZ EVALUATION---\n{env.wins()}\n", file=f)

