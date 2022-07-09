from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from battle_environment import BattleEnvironment
import os
import pprint
import matplotlib.pyplot as plt

for i in range(1, 100):
    if not os.path.exists(f'models/PPO_{i}'):
        FOLDER =  f'models/PPO_{i}'
        LOG_DIR = f'{FOLDER}/logs'
        os.makedirs(FOLDER)
        os.makedirs(LOG_DIR)
        break
# FOLDER = 'models/PPO_13'

# ---------- CALLBACK ----------
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, env, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.num_wins = 0
        self.total_games = 0
        self.games = 0
        self.env = env
        self.total_wins = 0
        self.fig, self.ax = plt.subplots()
        self.ax.set(xlabel='timesteps', ylabel='percentage of agent wins')
        self.ax.grid()
        self.x = []
        self.y = []

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, '{}'.format(self.n_calls))
            self.model.save(model_path)
        if self.n_calls % 10000 == 0 and self.env.team['red']['wins'] > 0 and self.n_calls > 0:
            self.num_wins = self.env.team['red']['wins'] - self.total_wins
            self.games = self.env.total_games - self.total_games
            self.total_wins = self.env.team['red']['wins']
            self.total_games = self.env.total_games
            print(f"\n\n\n\n--------------------\ntimesteps:{self.n_calls}\nwin percentage:{round(self.num_wins/self.games * 100, 2)}\n--------------------\n\n\n\n")
            self.x.append(self.n_calls)
            self.y.append(self.num_wins/self.games * 100)
        return True

CHECKPOINT_DIR_R1 = f'{FOLDER}/train_r1'
CHECKPOINT_DIR_R2 = f'{FOLDER}/train_r2'
save_freq = 10000


# ---------- CONFIG 1st round ----------
cf = {
    'show_viz': False,
    'hit_base_reward': 100,
    'hit_plane_reward': 30,
    'miss_punishment': 0,
    'too_long_punishment': 0,
    'closer_to_base_reward': 0,
    'closer_to_plane_reward': 0,
    'lose_punishment': 0,
    'fps': 60
}

timesteps = 100000
saved_timesteps = timesteps // save_freq * save_freq

with open(f'{FOLDER}/results.txt', 'w') as f:
    print("Round 1:\n", file=f)
    print("Timesteps:", saved_timesteps, file=f)
    pprint.pprint(cf, stream=f)


# ---------- TRAINING 1st round ----------
env = BattleEnvironment(show=cf['show_viz'], hit_base_reward=cf['hit_base_reward'], hit_plane_reward=cf['hit_plane_reward'], miss_punishment=cf['miss_punishment'], 
    too_long_punishment=cf['too_long_punishment'], closer_to_base_reward=cf['closer_to_base_reward'], closer_to_plane_reward=cf['closer_to_plane_reward'], lose_punishment=cf['lose_punishment'])
callback_r1 = TrainAndLoggingCallback(check_freq=save_freq, save_path=CHECKPOINT_DIR_R1, env=env)
model = PPO('MlpPolicy', env, tensorboard_log=LOG_DIR, verbose=1)
model.learn(total_timesteps=timesteps, callback=callback_r1)
callback_r1.ax.plot(callback_r1.x, callback_r1.y)
callback_r1.fig.savefig(f"{CHECKPOINT_DIR_R1}/percent_win_r1.png")
plt.show()


# ---------- CONFIG 2nd round ----------
cf = {
    'show_viz': False,
    'hit_base_reward': 100,
    'hit_plane_reward': 100,
    'miss_punishment': -1,
    'too_long_punishment': 0,
    'closer_to_base_reward': 0,
    'closer_to_plane_reward': 0,
    'lose_punishment': 0,
    'fps': 20
}

timesteps = 100000
saved_timesteps = timesteps // save_freq * save_freq

with open(f'{FOLDER}/results.txt', 'w') as f:
    print("Round 2:\n", file=f)
    print("Timesteps:", saved_timesteps, file=f)
    pprint.pprint(cf, stream=f)


# ---------- TRAINING 2nd round ----------
env = BattleEnvironment(show=cf['show_viz'], hit_base_reward=cf['hit_base_reward'], hit_plane_reward=cf['hit_plane_reward'], miss_punishment=cf['miss_punishment'], 
    too_long_punishment=cf['too_long_punishment'], closer_to_base_reward=cf['closer_to_base_reward'], closer_to_plane_reward=cf['closer_to_plane_reward'], lose_punishment=cf['lose_punishment'])
callback_r2 = TrainAndLoggingCallback(check_freq=save_freq, save_path=CHECKPOINT_DIR_R2, env=env)
model.set_env(env)
model.learn(total_timesteps=timesteps, callback=callback_r2)
callback_r2.ax.plot(callback_r2.x, callback_r2.y)
callback_r2.fig.savefig(f"{CHECKPOINT_DIR_R1}/percent_win_r2.png")
plt.show()

# ---------- EVALUATION WITHOUT VISUALS ----------
env = BattleEnvironment(show=cf['show_viz'], hit_base_reward=cf['hit_base_reward'], hit_plane_reward=cf['hit_plane_reward'], miss_punishment=cf['miss_punishment'], 
    too_long_punishment=cf['too_long_punishment'], closer_to_base_reward=cf['closer_to_base_reward'], closer_to_plane_reward=cf['closer_to_plane_reward'], lose_punishment=cf['lose_punishment'])
model.set_env(env)
episodes = 1000
avg = 0
for episode in range(episodes): # Evaluates the model n times
    state = env.reset()
    score = 0
    while not env.done:
        env.render()
        action, _states = model.predict(state)
        n_state, reward, done, info = env.step(action)
        score += reward
        avg += reward
    if (episode+1) % 10 == 0:
        print(f"episode:{episode+1}, avg reward:{avg/10}")
        avg = 0
print(env.wins())
with open(f'{FOLDER}/results.txt', 'w') as f:
    print(f"---EVALUATION---\n{env.wins()}\n", file=f)


# ---------- EVALUATION WITH VISUALS ----------
cf['show_viz'] = True
env = BattleEnvironment(show=cf['show_viz'], hit_base_reward=cf['hit_base_reward'], hit_plane_reward=cf['hit_plane_reward'], miss_punishment=cf['miss_punishment'], 
    too_long_punishment=cf['too_long_punishment'], closer_to_base_reward=cf['closer_to_base_reward'], closer_to_plane_reward=cf['closer_to_plane_reward'], lose_punishment=cf['lose_punishment'], fps=cf['fps'])
model.set_env(env)
episodes = 50
for episode in range(episodes): # Evaluates the model n times
    state = env.reset()
    score = 0
    while not env.done:
        env.render()
        action, _states = model.predict(state)
        n_state, reward, done, info = env.step(action)
        score += reward
print(env.wins())

