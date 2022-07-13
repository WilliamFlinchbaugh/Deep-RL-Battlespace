import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from battle_environment import BattleEnvironment
import os
import pprint
import matplotlib.pyplot as plt

for i in range(1, 100):
    if not os.path.exists(f'models/PPO_{i}'):
        FOLDER =  f'models/PPO_{i}'
        LOG_DIR = f'{FOLDER}/logs'
        OPT_DIR = f'{FOLDER}/opt/'
        os.makedirs(FOLDER)
        os.makedirs(LOG_DIR)
        os.makedirs(OPT_DIR)
        break
    
# ---------- CONFIG ----------
cf = {
    'hit_base_reward': 2,
    'hit_plane_reward': 1,
    'miss_punishment': 0,
    'too_long_punishment': 0,
    'lose_punishment': -3
}

# ---------- HYPERPARAM TUNING ----------
def optimize_ppo(trial): 
    return {
        'n_steps':trial.suggest_int('n_steps', 2048, 8192),
        'gamma':trial.suggest_loguniform('gamma', 0.8, 0.9999),
        'learning_rate':trial.suggest_loguniform('learning_rate', 1e-6, 1e-3),
        'clip_range':trial.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda':trial.suggest_uniform('gae_lambda', 0.8, 0.99)
    }

def optimize_agent(trial):
    try:
        model_params = optimize_ppo(trial)
        env = BattleEnvironment(show=False, hit_base_reward=cf['hit_base_reward'], hit_plane_reward=cf['hit_plane_reward'], miss_punishment=cf['miss_punishment'], 
            too_long_punishment=cf['too_long_punishment'], lose_punishment=cf['lose_punishment'])
        model = PPO('MlpPolicy', env, tensorboard_log=LOG_DIR, verbose=0, **model_params)
        model.learn(total_timesteps=200000)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100)
        SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}'.format(trial.number))
        model.save(SAVE_PATH)
        return mean_reward
    except Exception as e:
        return -1000

study = optuna.create_study(direction='maximize')
study.optimize(optimize_agent, n_trials=20)

model_params = study.best_params
model_params['n_steps'] = (model_params['n_steps'] // 64) * 64 # set to a factor of 64
best_trial = study.best_trial.number

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
        self.ax.set(xlabel='# of games played', ylabel='Percentage of agent wins')
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
            self.x.append(self.env.total_games)
            self.y.append(self.num_wins/self.games * 100)
        return True

CHECKPOINT_DIR = f'{FOLDER}/train'
save_freq = 10000

timesteps = 3000000
saved_timesteps = timesteps // save_freq * save_freq
file = open(f"{FOLDER}/results.txt", 'a')
print("Timesteps:", saved_timesteps, file=file)
print("Config:", file=file)
pprint.pprint(cf, stream=file)
print("Parameters:", file=file)
pprint.pprint(model_params, stream=file)

# ---------- LOAD AND TRAIN MODEL -----------
callback = TrainAndLoggingCallback(check_freq=save_freq, save_path=CHECKPOINT_DIR)
env = BattleEnvironment(show=False, hit_base_reward=cf['hit_base_reward'], hit_plane_reward=cf['hit_plane_reward'], miss_punishment=cf['miss_punishment'], 
    too_long_punishment=cf['too_long_punishment'], lose_punishment=cf['lose_punishment'])
model = PPO('MlpPolicy', env, tensorboard_log=LOG_DIR, verbose=1, **model_params)
model.load(os.path.join(OPT_DIR, f'trial_{best_trial}.zip'))
model.learn(total_timesteps=timesteps, callback=callback)
model.save(f"{FOLDER}/final_model")
del model

callback.ax.plot(callback.x, callback.y)
callback.fig.savefig(f"{FOLDER}/percent_win.png")
plt.show()

# Load trained agent and evaluate 1000 games
eval_env = BattleEnvironment(show=False, hit_base_reward=cf['hit_base_reward'], hit_plane_reward=cf['hit_plane_reward'], miss_punishment=cf['miss_punishment'], 
    too_long_punishment=cf['too_long_punishment'], lose_punishment=cf['lose_punishment'])
model = PPO.load(f"{FOLDER}/final_model")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1000, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
print(eval_env.wins())
print(f"---EVALUATION---\n{eval_env.wins()}\n", file=file)

# Evaluate with visuals (10 games)
eval_env = BattleEnvironment(show=True, hit_base_reward=cf['hit_base_reward'], hit_plane_reward=cf['hit_plane_reward'], miss_punishment=cf['miss_punishment'], 
    too_long_punishment=cf['too_long_punishment'], lose_punishment=cf['lose_punishment'], fps=30)
model = PPO.load(f"{FOLDER}/final_model")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
print(eval_env.wins())
