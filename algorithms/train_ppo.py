import os
import datetime
from algorithms.ppo import Agent

GAMMA = 0.99
ALPHA = 0.0003
GAE_LAMBDA = 0.95
POLICY_CLIP = 0.2
BATCH_SIZE = 64
N_EPOCHS = 10

def train(env, n_games=10000):
    # Create a new folder for the model
    for i in range(1, 100):
        if not os.path.exists(f'models/ppo_{i}'):
            FOLDER = f'models/ppo_{i}'
            os.makedirs(FOLDER)
            break

    n_actions = env.n_actions

    agents = {}
    for agent_id in env.possible_agents:
        agents[agent_id] = Agent(n_actions, [env.obs_size], GAMMA, ALPHA, GAE_LAMBDA,
                    POLICY_CLIP, BATCH_SIZE, N_EPOCHS, chkpt_dir=FOLDER, name=agent_id)

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
            prob = {}
            val = {}

            for agent in alive_agents:
                actions[agent], prob[agent], val[agent] = agents[agent].choose_action(obs[agent])

            obs_, rewards, dones, info = env.step(actions)

            for agent in alive_agents:
                agents[agent].remember(obs[agent], actions[agent], prob[agent], val[agent], 
                                rewards[agent], dones[agent])

                if i % 20 == 0:
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