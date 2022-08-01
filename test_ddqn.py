from dueling_ddqn import Agent, DuelingDeepQNetwork, ReplayBuffer
import torch as T
import battle_v1

cf = {
    'n_agents': 2, # Number of planes on each team
    'show': False, # Show visuals
    'hit_base_reward': 10, # Reward value for hitting enemy base
    'hit_plane_reward': 1, # Reward value for hitting enemy plane
    'miss_punishment': 0, # Punishment value for missing a shot
    'die_punishment': 0, # Punishment value for a plane dying
    'fps': 60 # Framerate that the visuals run at
}

# What device to use
use_gpu = True

if T.cuda.is_available():
    print("\nGPU available")
    if use_gpu:
        print("Using GPU")
        device = 'cuda:0'
else:
    print("\nUsing CPU")
    device = 'cpu'

GAMMA = 0.99
LEARNING_RATE = 0.001
EPS_MIN = 0.05
EPS_DEC = 8e-7
BUFFER_SIZE = 100000
BATCH_SIZE = 32

env = battle_v1.parallel_env(**cf)
n_actions = env.n_actions

agents = {}
for agent_id in env.possible_agents:
    agents[agent_id] = Agent(GAMMA, 1.0, LEARNING_RATE, n_actions, [env.obs_size], BUFFER_SIZE, BATCH_SIZE, eps_min=EPS_MIN, eps_dec=EPS_DEC)
    agents[agent_id].q_eval.load_checkpoint(f"battle_eval_{agent_id}")

for i in range(10):
    obs = env.reset()

    while not env.env_done:
        alive_agents = env.agents
        actions = {}
        for agent in alive_agents:
            actions[agent] = agents[agent].choose_action(obs[agent])
        obs_, rewards, dones, info = env.step(actions)
        obs = obs_

env.close()

