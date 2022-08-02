import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import battle_v1
import time

class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                    dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                        dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DuelingDeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = nn.Linear(*input_dims, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        flat1 = F.relu(self.fc1(state))
        V = self.V(flat1)
        A = self.A(flat1)

        return V, A

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, agent_id, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, chkpt_dir='tmp'):
        self.agent_id = agent_id
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims)

        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=f'battle_eval_{agent_id}',
                                   chkpt_dir=self.chkpt_dir)

        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=f'battle_next_{agent_id}',
                                   chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation,dtype=T.float).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s,
                        (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_,
                        (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1,keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

cf = {
    'n_agents': 2, # Number of planes on each team
    'show': False, # Show visuals
    'hit_base_reward': 1000, # Reward value for hitting enemy base
    'hit_plane_reward': 30, # Reward value for hitting enemy plane
    'miss_punishment': -1, # Punishment value for missing a shot
    'die_punishment': 0, # Punishment value for a plane dying
    'fps': 30 # Framerate that the visuals run at
}

if __name__ == '__main__':

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
    EPS_DEC = 2e-7
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32

    env = battle_v1.parallel_env(**cf)
    n_actions = env.n_actions

    agents = {}
    for agent_id in env.possible_agents:
        agents[agent_id] = Agent(GAMMA, 1.0, LEARNING_RATE, n_actions, [env.obs_size], 
                    BUFFER_SIZE, BATCH_SIZE, agent_id, eps_min=EPS_MIN, eps_dec=EPS_DEC)

    n_games = 20000
    timesteps_cntr = 0
    wins = {
        'red': 0,
        'blue': 0,
        'tie': 0
    }

    print("\n=====================\n| Starting Training |\n=====================\n")
    start = time.time() # Get the starting time

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
            now = time.time()
            elapsed = now - start # Elapsed time in seconds

            # Print out progress
            print(f'\n=========================\n\
| Current Time: {time.strftime("%H:%M:%S", time.gmtime(now))}\
| Elapsed Time: {time.strftime("%H:%M:%S", time.gmtime(elapsed))}\n\
| Games: {env.total_games}\n\
| Epsilon: {round(agents[env.possible_agents[0]].epsilon, 3)}\n\
| Timesteps: {timesteps_cntr}\n\
| Red Wins: {wins["red"]}\n\
| Blue Wins: {wins["blue"]}\n\
| Ties: {wins["tie"]}\n\
==========================\n')

            wins = {'red': 0, 'blue': 0, 'tie': 0} # Reset the win history'

            # Visualize 1 game every 1000 trained games
            if env.total_games % 1000 == 0:
                env.show = True

            # Save models
            print("\n=====================\n| Saving Models |\n=====================\n")
            for agent in agents.values():
                agent.save_models()

        elif env.show:
            env.show = False
            env.close()

    env.close()

