import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class ReplayBuffer:
    def __init__(self, mem_size, batch_size, agent_list, obs_shape):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.batch_size = batch_size
        self.agent_list = agent_list
        self.n_agents = len(agent_list)

        self.states_mem = np.zeros((self.mem_size, self.n_agents, *obs_shape), dtype=np.float32)
        self.actions_mem = np.zeros((self.mem_size, self.n_agents), dtype=np.float32)
        self.rewards_mem = np.zeros((self.mem_size, self.n_agents), dtype=np.float32)
        self.new_states_mem = np.zeros((self.mem_size, self.n_agents, *obs_shape), dtype=np.float32)
        self.dones_mem = np.zeros((self.mem_size, self.n_agents), dtype=np.bool8)

    def store_transition(self, states, actions, rewards, states_, dones):
        index = self.mem_cntr % self.mem_size
        for idx, agent in enumerate(self.agent_list):
            self.states_mem[index][idx] = states[agent]
            self.actions_mem[index][idx] = actions[agent]
            self.rewards_mem[index][idx] = rewards[agent]
            self.new_states_mem[index][idx] = states_[agent]
            self.dones_mem[index][idx] = dones[agent]
        self.mem_cntr += 1

    def sample(self):
        max_mem = min(self.mem_cntr, self.mem_size) # makes sure we only sample the filled data
        batch = np.random.choice(max_mem, self.batch_size)
        states = self.states_mem[batch]
        actions = self.actions_mem[batch]
        rewards = self.rewards_mem[batch]
        states_ = self.new_states_mem[batch]
        dones = self.dones_mem[batch]

        return states, actions, rewards, states_, dones

    def is_ready(self):
        return self.mem_cntr >= self.batch_size
        
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, n_agents, lr, chkpt_dir='tmp/ddpg', name='critic'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.n_agents = n_agents

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)

        self.bn1 = nn.LayerNorm(fc1_dims)
        self.bn2 = nn.LayerNorm(fc2_dims)

        self.action_value = nn.Linear(self.n_actions, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(lr, self.parameters())
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, lr, chkpt_dir='tmp/ddpg', name='actor'):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(lr, self.parameters())
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

class Agent():
    def __init__(self, input_dims, n_actions, fc1_dims, fc2_dims, actor_lr, gamma=0.99, buffer_size=1000000, batch_size=64):
        self.actor = ActorNetwork()
        self.target_actor = ActorNetwork()

class Team():
    def __init__(self, mem_size, batch_size, agent_list, input_dims, n_actions, fc1_dims, fc2_dims, actor_lr, critic_lr, tau):
        self.replay_buffer = ReplayBuffer(mem_size, batch_size, agent_list)
        self.critic = CriticNetwork(input_dims, fc1_dims, fc2_dims, n_actions, len(agent_list), critic_lr)
        self.target_critic = CriticNetwork(input_dims, fc1_dims, fc2_dims, n_actions, len(agent_list), critic_lr)
        self.input_dims = input_dims
        self.n_actions = n_actions

        self.agents = []
        for agent in agent_list:
            self.agents.append(Agent(input_dims, n_actions, fc1_dims, fc2_dims, actor_lr))

    def learn(self):
        if not self.memory.is_ready():
            return

        states, actions, rewards, states_, dones = self.replay_buffer.sample()

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        dones = T.tensor(dones, dtype=T.bool).to(self.actor.device)
        
        target_actions = 0 # FIXME target_actions = target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[dones] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        # self.actor.optimizer.zero_grad()
        # actor_loss = -self.critic.forward(states, self.actor.forward(states))
        # actor_loss = T.mean(actor_loss)
        # actor_loss.backward()
        # self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        critic_params = self.critic.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        critic_state_dict = dict(critic_params)
        target_critic_state_dict = dict(target_critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_state_dict[name].clone()
        self.target_critic.load_state_dict(critic_state_dict)
        

