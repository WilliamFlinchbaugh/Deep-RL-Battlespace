from battle_environment import BattleEnvironment
from stable_baselines3 import PPO

cf = {
    'show_viz': False,
    'hit_base_reward': 100,
    'hit_plane_reward': 30,
    'miss_punishment': 0,
    'too_long_punishment': 0,
    'closer_to_base_reward': 0,
    'closer_to_plane_reward': 0,
    'lose_punishment': 0,
    'fps': 40
}

# ---------- NO VISUALS
env = BattleEnvironment(show=cf['show_viz'], hit_base_reward=cf['hit_base_reward'], hit_plane_reward=cf['hit_plane_reward'], miss_punishment=cf['miss_punishment'], 
    too_long_punishment=cf['too_long_punishment'], closer_to_base_reward=cf['closer_to_base_reward'], closer_to_plane_reward=cf['closer_to_plane_reward'], lose_punishment=cf['lose_punishment'])
model = PPO.load(f'models/PPO_13/train/640000.zip')

episodes = 1000
for episode in range(episodes): # Evaluates the model n times
    state = env.reset()
    score = 0
    while not env.done:
        env.render()
        action, _states = model.predict(state)
        # action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward

    if (episode+1) % 10 == 0:
        print(f"episode {episode +1}")

print(env.wins())

# ---------- VISUALS ----------
cf['show_viz'] = True
env = BattleEnvironment(show=cf['show_viz'], hit_base_reward=cf['hit_base_reward'], hit_plane_reward=cf['hit_plane_reward'], miss_punishment=cf['miss_punishment'], 
    too_long_punishment=cf['too_long_punishment'], closer_to_base_reward=cf['closer_to_base_reward'], closer_to_plane_reward=cf['closer_to_plane_reward'], lose_punishment=cf['lose_punishment'], fps=cf['fps'])
model = PPO.load(f'models/PPO_13/train/640000.zip', env=env)

episodes = 10
for episode in range(episodes): # Evaluates the model n times
    state = env.reset()
    score = 0
    while not env.done:
        env.render()
        action, _states = model.predict(state)
        # action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward

    if (episode+1) % 10 == 0:
        print(f"episode {episode +1}")

print(env.wins())