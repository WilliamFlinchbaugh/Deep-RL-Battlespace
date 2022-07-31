import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import battle_v1
import timeit

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device(device)
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.05, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.Q_eval = DeepQNetwork(lr, n_actions=n_actions,
                    input_dims=input_dims, fc1_dims=256, fc2_dims=256)
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array([observation])).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if not self.ready():
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(
                self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(
                self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(
                self.terminal_memory[batch]).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma*T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def ready(self):
        return self.mem_cntr >= self.batch_size

cf = {
    'n_agents': 2, # Number of planes on each team
    'show': False, # Show visuals
    'hit_base_reward': 5, # Reward value for hitting enemy base
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
LEARNING_RATE = .01
EPS_START = 1.0
EPS_END = 0.05
EPS_DEC = 8e-7
BATCH_SIZE = 64

env = battle_v1.parallel_env(**cf)
n_actions = env.n_actions

agents = {}
for agent_id in env.possible_agents:
    agents[agent_id] = Agent(GAMMA, EPS_START, LEARNING_RATE, [env.obs_size], 
                BATCH_SIZE, n_actions, eps_end=EPS_END, eps_dec=EPS_DEC)

n_games = 100000
timesteps_cntr = 0
wins = {
    'red': 0,
    'blue': 0,
    'tie': 0
}

print("\n\n=====================\n| STARTING TRAINING |\n=====================\n")
start = timeit.default_timer() # Get the starting time

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
        now = timeit.default_timer()
        time = now - start # Elapsed time in seconds

        # Print out progress
        print(f'\n=========================\n\
| Elapsed Time: {int(time//3600)}::{int(time%3600//60)}::{int(time%3600%60)}\n\
| Games: {env.total_games}\n\
| Epsilon: {agents[env.possible_agents[0]].epsilon}\n\
| Timesteps: {timesteps_cntr}\n\
| Red Wins: {wins["red"]}\n\
| Blue Wins: {wins["blue"]}\n\
| Ties: {wins["tie"]}\n\
==========================\n')

        wins = {'red': 0, 'blue': 0, 'tie': 0} # Reset the win history
        env.show = True # Evaluate one game
    elif env.show:
        env.show = False
        env.close()

env.close()

# Save the models
for agent_id, agent in agents.items():
    path = f"models/{agent_id}.pt"
    T.save(agent.Q_eval.state_dict(), path)
