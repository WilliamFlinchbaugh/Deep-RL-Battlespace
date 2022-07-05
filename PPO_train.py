from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from battle_environment import BattleEnvironment
import os
FOLDER = 'results/PPO_2'
LOG_DIR = f'{FOLDER}/logs/'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

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
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True

CHECKPOINT_DIR = f'{FOLDER}/train/'
save_freq = 50000
timesteps = 2000000
saved_timesteps = timesteps // save_freq * save_freq

# ---------- TRAINING ----------
callback = TrainAndLoggingCallback(check_freq=save_freq, save_path=CHECKPOINT_DIR)
env = BattleEnvironment(show=False, hit_base_reward=100, hit_plane_reward=100, miss_punishment=-5, too_long_punishment=0, closer_to_base_reward=0, 
    closer_to_plane_reward=0, lose_punishment=-50)
config = [env.hit_base_reward, env.hit_plane_reward, env.miss_punishment, env.too_long_punishment, env.closer_to_base_reward, env.closer_to_plane_reward, env.lose_punishment]
model = PPO('MlpPolicy', env, tensorboard_log=LOG_DIR, verbose=1)
model.learn(total_timesteps=timesteps, callback=callback)
del model

# ---------- EVALUATION ----------
model = PPO.load(f'{CHECKPOINT_DIR}best_model_{saved_timesteps}')
env = BattleEnvironment(show=True, hit_base_reward=config[0], hit_plane_reward=config[1], miss_punishment=config[2], too_long_punishment=config[3], closer_to_base_reward=config[4], 
    closer_to_plane_reward=config[5], lose_punishment=config[6], fps=120)
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
        print('# Episodes:{} Avg Score:{}'.format(episode+1, score/10))
print(env.wins())
with open(f'{FOLDER}/results.txt', 'w') as f:
    print(f"VALUES:\nhit base:{config[0]}\nhit plane:{config[1]}\nmiss:{config[2]}\ntoo long:{config[3]}\ncloser to base:{config[4]}\ncloser to plane:{config[5]}\nlose:{config[6]}")
    print(env.wins(), file=f)

# ---------- VISUALIZATION ----------
env = BattleEnvironment(show=True, hit_base_reward=config[0], hit_plane_reward=config[1], miss_punishment=config[2], too_long_punishment=config[3], closer_to_base_reward=config[4], 
    closer_to_plane_reward=config[5], lose_punishment=config[6], fps=20)
episodes = 5
for episode in range(episodes): # Evaluates the model n times
    state = env.reset()
    score = 0
    while not env.done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode+1, score))
print(env.wins())
