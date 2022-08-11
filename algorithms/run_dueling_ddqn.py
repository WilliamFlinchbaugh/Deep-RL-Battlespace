import os
import datetime
from algorithms.dueling_ddqn import Agent
from utils import plot_data
from pprint import pprint

GAMMA = 0.99
LEARNING_RATE = 0.001
EPS_MIN = 0.05
EPS_DEC = 5e-7
BUFFER_SIZE = 100000
BATCH_SIZE = 32

def train(env, env_config, n_games=10000, gamma=GAMMA, learning_rate=LEARNING_RATE, eps_min=EPS_MIN, eps_dec=EPS_DEC, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE):
    # Create a new folder for the model
    for i in range(1, 100):
        if not os.path.exists(f'models/dueling_ddqn_{i}'):
            FOLDER = f'models/dueling_ddqn_{i}'
            os.makedirs(FOLDER)
            os.makedirs(f'{FOLDER}/training_vids')
            break

    # Save the configuration of the model
    hyperparams = {'gamma': gamma, 'learning_rate': learning_rate, 'eps_min': eps_min, 'eps_dec': eps_dec, 'buffer_size': buffer_size, 'batch_size': batch_size}
    f = open(f"{FOLDER}/config.txt", 'a')
    f.write("ALGORITHM: DUELING DDQN\n\nENV CONFIG:\n")
    pprint(env_config, stream=f)
    f.write("\nHYPERPARAMETERS:\n")
    pprint(hyperparams, stream=f)
    f.close()
    n_actions = env.n_actions

    agents = {}
    for agent_id in env.possible_agents:
        agents[agent_id] = Agent(GAMMA, 1.0, LEARNING_RATE, n_actions, [env.obs_size], 
                    BUFFER_SIZE, BATCH_SIZE, agent_id, eps_min=EPS_MIN, eps_dec=EPS_DEC, chkpt_dir=FOLDER)

    timesteps_cntr = 0
    wins = {
        'red': 0,
        'blue': 0,
        'tie': 0
    }
    rewards_history = {agent_id: [] for agent_id in env.possible_agents}

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

            for agent in env.possible_agents:
                rewards_history[agent].append(rewards[agent])

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
                env.start_recording(f'{FOLDER}/training_vids/{i+1}.mp4')

            # Save models
            print("\n=================\n| Saving Models |\n=================\n")
            for agent in agents.values():
                agent.save_models()

        elif env.show:
            env.export_video()
            env.show = False
            env.close()

    plot_data(rewards_history, FOLDER + '/mean_rew.svg')
    env.close()

def evaluate(env, model_path, eval_games=10):
    env.show = True
    n_actions = env.n_actions

    agents = {}
    for agent_id in env.possible_agents:
        agents[agent_id] = Agent(GAMMA, 1.0, LEARNING_RATE, n_actions, [env.obs_size], 
                    BUFFER_SIZE, BATCH_SIZE, agent_id, eps_min=EPS_MIN, eps_dec=EPS_DEC, chkpt_dir=model_path)
        agents[agent_id].load_models()

    env.start_recording(f'{model_path}/eval_vid.mp4')
    for i in range(eval_games):
        obs = env.reset()

        while not env.env_done:
            alive_agents = env.agents
            actions = {}
            for agent in alive_agents:
                actions[agent] = agents[agent].choose_action(obs[agent])
            obs_, rewards, dones, info = env.step(actions)
            obs = obs_

    env.export_video()
    env.close()