import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from battle_environment import BattleEnvironment
import os

FOLDER = 'results/PPO_1'
LOG_DIR = f'{FOLDER}/logs/'
OPT_DIR = f'{FOLDER}/opt/'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if not os.path.exists(OPT_DIR):
    os.makedirs(OPT_DIR)

# ---------- HYPERPARAM TUNING ----------
def optimize_ppo(trial): 
    return {
        'n_steps':trial.suggest_int('n_steps', 2048, 8192),
        'gamma':trial.suggest_loguniform('gamma', 0.8, 0.9999),
        'learning_rate':trial.suggest_loguniform('learning_rate', 1e-5, 1e-4),
        'clip_range':trial.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda':trial.suggest_uniform('gae_lambda', 0.8, 0.99)
    }

def optimize_agent(trial):
    try:
        model_params = optimize_ppo(trial)
        env = BattleEnvironment()
        model = PPO('MlpPolicy', env, tensorboard_log=LOG_DIR, verbose=0, **model_params)
        model.learn(total_timesteps=200000)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
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
timesteps = 1000000
saved_timesteps = timesteps // save_freq * save_freq

# ---------- LOAD AND TRAIN MODEL -----------
callback = TrainAndLoggingCallback(check_freq=save_freq, save_path=CHECKPOINT_DIR)
env = BattleEnvironment()
model = PPO('MlpPolicy', env, tensorboard_log=LOG_DIR, verbose=1, **model_params)
model.load(os.path.join(OPT_DIR, f'trial_{best_trial}.zip'))
model.learn(total_timesteps=timesteps, callback=callback)
del model

# ---------- EVALUATION ----------
model = PPO.load(f'{CHECKPOINT_DIR}best_model_{saved_timesteps}')
env = BattleEnvironment(show=True, fps=120)
episodes = 100
for episode in range(episodes): # Evaluates the model n times
    state = env.reset()
    score = 0
    while not env.done:
        env.render()
        action, _states = model.predict(state)
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode+1, score))
with open(f'{FOLDER}/results.txt', 'w') as f:
    print(env.wins(), file=f)
