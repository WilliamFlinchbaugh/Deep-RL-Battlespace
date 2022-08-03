import battle_env
from algorithms import dqn, dueling_ddqn, ppo

# Implemented algorithms: dqn, dueling_ddqn, ppo
ALGORITHM = 'dueling_ddqn'

algorithms = {
    'dqn':dqn,
    'dueling_ddqn':dueling_ddqn,
    'ppo':ppo
}

algorithm = algorithms[ALGORITHM]

cf = {
    'n_agents': 2, # Number of planes on each team
    'show': False, # Show visuals
    'hit_base_reward': 1000, # Reward value for hitting enemy base
    'hit_plane_reward': 50, # Reward value for hitting enemy plane
    'miss_punishment': -2, # Punishment value for missing a shot
    'die_punishment': 0, # Punishment value for a plane dying
    'fps': 20 # Framerate that the visuals run at
}

env = battle_env.parallel_env(**cf)
n_actions = env.n_actions

agents = {}
for agent_id in env.possible_agents:
    agents[agent_id] = Agent(GAMMA, 0, LEARNING_RATE, n_actions, [env.obs_size], BUFFER_SIZE, BATCH_SIZE, agent_id, eps_min=EPS_MIN, eps_dec=EPS_DEC, chkpt_dir="models/dueling_ddqn_2")
    agents[agent_id].load_models()

for i in range(20):
    obs = env.reset()

    while not env.env_done:
        alive_agents = env.agents
        actions = {}
        for agent in alive_agents:
            actions[agent] = agents[agent].choose_action(obs[agent])
        obs_, rewards, dones, info = env.step(actions)
        obs = obs_

env.close()