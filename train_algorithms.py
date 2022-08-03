import battle_env
import os
import datetime
from pytz import timezone
from algorithms import dqn, dueling_ddqn

ALGORITHM = 'dqn'
GAMMA = 0.99
LEARNING_RATE = 0.001
EPS_MIN = 0.05
EPS_DEC = 2e-7
BUFFER_SIZE = 100000
BATCH_SIZE = 32

# Create a new folder for the model
for i in range(1, 100):
    if not os.path.exists(f'models/{ALGORITHM}_{i}'):
        FOLDER = f'models/{ALGORITHM}_{i}'
        os.makedirs(FOLDER)
        break

algorithms = {
    'dqn':dqn,
    'dueling_ddqn':dueling_ddqn
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
    agents[agent_id] = algorithm.Agent(GAMMA, 1.0, LEARNING_RATE, n_actions, [env.obs_size], 
                BUFFER_SIZE, BATCH_SIZE, agent_id, eps_min=EPS_MIN, eps_dec=EPS_DEC, chkpt_dir=FOLDER)

n_games = 30000
timesteps_cntr = 0
wins = {
    'red': 0,
    'blue': 0,
    'tie': 0
}

print("\n=====================\n| Starting Training |\n=====================\n")
start = datetime.datetime.now()

for i in range(n_games):
    obs = env.reset()

    while not env.env_done:
        timesteps_cntr += 1
        alive_agents = env.agents
        actions = {}
        for agent in alive_agents:
            actions[agent] = agents[agent].choose_action(obs[agent])
        obs_, rewards, dones, info = env.step(actions)
        for agent in alive_agents:
            agents[agent].store_transition(obs[agent], actions[agent],
                            rewards[agent], obs_[agent], dones[agent])
            agents[agent].learn()
        obs = obs_

    # Add outcome to wins
    wins[env.winner] += 1

    if env.total_games % 100 == 0 and env.total_games > 0:
        now = datetime.datetime.now()
        elapsed = now - start # Elapsed time in seconds
        central = timezone("US/Central")
        time_now = now.astimezone(central)
        s = elapsed.total_seconds()
        hours, remainder = divmod(s, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_elapsed = '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))

        # Print out progress
        print(f'\n=========================\n\
| Current Time: {now.strftime("%I:%M %p")}\n\
| Elapsed Time: {formatted_elapsed}\n\
| Games: {env.total_games}\n\
| Epsilon: {round(agents[env.possible_agents[0]].epsilon, 3)}\n\
| Timesteps: {timesteps_cntr}\n\
| Red Wins: {wins["red"]}\n\
| Blue Wins: {wins["blue"]}\n\
| Ties: {wins["tie"]}\n\
==========================\n')

        wins = {'red': 0, 'blue': 0, 'tie': 0} # Reset the win history

        # Visualize 1 game every 1000 trained games
        if env.total_games % 1000 == 0:
            env.show = True

        # Save models
        print("\n=================\n| Saving Models |\n=================\n")
        for agent in agents.values():
            agent.save_models()

    elif env.show:
        env.show = False
        env.close()

env.close()