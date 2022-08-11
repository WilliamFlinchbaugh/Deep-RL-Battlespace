import os
import datetime
import envs.battle_env as battle_env
import algorithms.WIP_custom_model as model
from utils.utils import plot_data
from pprint import pprint

GAMMA = 0.99
ALPHA = 0.0003
GAE_LAMBDA = 0.95 
POLICY_CLIP = 0.2
BATCH_SIZE = 64
N_EPOCHS = 10

def train(env, env_config, n_games=10000, gamma=GAMMA, alpha=ALPHA, gae_lambda=GAE_LAMBDA, policy_clip=POLICY_CLIP, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS):
    # Create a new folder for the model
    for i in range(1, 100):
        if not os.path.exists(f'models/ppo_shared_{i}'):
            FOLDER = f'models/ppo_shared_{i}'
            os.makedirs(FOLDER)
            os.makedirs(f'{FOLDER}/training_vids')
            break

    # Save the configuration of the model
    hyperparams = {'gamma': gamma, 'alpha': alpha, 'gae_lambda': gae_lambda, 'policy_clip': policy_clip, 'batch_size': batch_size, 'n_epochs': n_epochs}
    f = open(f"{FOLDER}/config.txt", 'a')
    f.write("ALGORITHM: PPO SHARED\n\nENV CONFIG:\n")
    pprint(env_config, stream=f)
    f.write("\nHYPERPARAMETERS:\n")
    pprint(hyperparams, stream=f)
    f.close()

    n_actions = env.n_actions

    # Instantiate teams
    teams = {
        'blue': model.Team(env.n_agents, env.possible_blue, n_actions, [env.obs_size], gamma, alpha, gae_lambda, policy_clip, batch_size, n_epochs, chkpt_dir=f"{FOLDER}/blue"),
        'red': model.Team(env.n_agents, env.possible_red, n_actions, [env.obs_size], gamma, alpha, gae_lambda, policy_clip, batch_size, n_epochs, chkpt_dir=f"{FOLDER}/red"),
    }
    timesteps_cntr = 0
    wins = {
        'red': 0,
        'blue': 0,
        'tie': 0
    }
    score_history = {agent_id: [] for agent_id in env.possible_agents}


    print("\n=====================\n| Starting Training |\n=====================\n")
    start = datetime.datetime.now()

    for i in range(n_games):
        obs = env.reset()
        scores = {agent_id: 0 for agent_id in env.possible_agents}
        recording = False

        while not env.env_done:
            timesteps_cntr += 1
            actions = {}
            prob = {}
            val = {}

            for id in env.agents:
                agent = teams[env.team_map[id]].agents[id] # Get the agent from the team
                actions[id], prob[id], val[id] = agent.choose_action(obs[id])

            obs_, rewards, dones, info = env.step(actions)
 
            for id in env.possible_agents:
                scores[id] += rewards[id]

            for id in env.agents:
                agent = teams[env.team_map[id]].agents[id] # Get the agent from the team
                agent.remember(obs[id], actions[id], prob[id], val[id], 
                                rewards[id], dones[id])

                if i % 20 == 0 and not recording: # Learn every 20 games
                    agent = teams[env.team_map[id]].agents[id] # Get the agent from the team
                    agent.learn()

            obs = obs_

        # Add outcome to wins
        wins[env.winner] += 1

        # Append scores
        for id in env.possible_agents:
            score_history[id].append(scores[id])

        # Print out progress and save models
        if env.total_games % 50 == 0 and env.total_games > 0:
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

            # Save models
            print("\n=================\n| Saving Models |\n=================\n")
            for team in teams.values():
                team.save_models()

            # Visualize 1 game every 1000 trained games
            if env.total_games % 1000 == 0:
                env.show = True
                env.start_recording(f'{FOLDER}/training_vids/{i+1}.mp4')
                recording = True

        elif env.show:
            env.export_video()
            env.show = False
            env.close()
            recording = False

    plot_data(score_history, FOLDER + '/mean_rew.svg')
    env.close()

def evaluate(env, model_path, eval_games=10, gamma=GAMMA, alpha=ALPHA, gae_lambda=GAE_LAMBDA, policy_clip=POLICY_CLIP, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS):
    env.show = True
    n_actions = env.n_actions

    teams = {
        'blue': model.Team(env.n_agents, env.possible_blue, n_actions, [env.obs_size], gamma, alpha, gae_lambda, policy_clip, batch_size, n_epochs, chkpt_dir=f"{model_path}/blue"),
        'red': model.Team(env.n_agents, env.possible_red, n_actions, [env.obs_size], gamma, alpha, gae_lambda, policy_clip, batch_size, n_epochs, chkpt_dir=f"{model_path}/red"),
    }

    env.start_recording(f'{model_path}/eval_vid.mp4')
    for _ in range(eval_games):
        obs = env.reset()

        while not env.env_done:
            actions = {}
            for id in env.agents:
                agent = teams[env.team_map[id]].agents[id] # Get the agent from the team
                actions[id], _, _ = agent.choose_action(obs[agent])

            obs_, _, _, _ = env.step(actions)
            obs = obs_

    env.export_video()
    env.close()

if __name__ == '__main__':

    env_config = {
        'n_agents': 2, # Number of planes on each team
        'show': False, # Show visuals
        'hit_base_reward': 100, # Reward value for hitting enemy base
        'hit_plane_reward': 13, # Reward value for hitting enemy plane
        'miss_punishment': -3, # Punishment value for missing a shot
        'die_punishment': 0, # Punishment value for a plane dying
        'fps': 20 # Framerate that the visuals run at
    }

    env = battle_env.parallel_env(**env_config)

    train(env=env, env_config=env_config, n_games=75000)
    # evaluate(env=env, model_path='models/ppo_shared_2', eval_games=10)