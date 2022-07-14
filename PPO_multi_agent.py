from pettingzoo.test import api_test
import battle_v1
import random

cf = {
    'n_agents': 1,
    'show': True,
    'hit_base_reward': 10,
    'hit_plane_reward': 2,
    'miss_punishment': 0,
    'too_long_punishment': 0,
    'lose_punishment': -3,
    'fps': 20
}

env = battle_v1.env(cf)
# api_test(env, num_cycles=1000, verbose_progress=False)

env.reset()
for agent in env.agent_iter():
    obs, reward, done, info = env.last()
    action = random.randint(0, 3)
    env.step(action)