from battle_environment import BattleEnvironment

env = BattleEnvironment(show=True, hit_base_reward=10, hit_plane_reward=10, miss_punishment=-5, too_long_punishment=0, closer_to_base_reward=0, 
    closer_to_plane_reward=0, lose_punishment=0, fps=30)
episodes = 10
for episode in range(episodes): # Evaluates the model n times
    state = env.reset()
    score = 0
    while not env.done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    if (episode+1) % 50 == 0:
        print('# Episodes:{} Avg Score:{}'.format(episode+1, score/10))
print(env.wins())
