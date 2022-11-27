import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CriticNetwork(nn.Module):
    def __init__(self, obs_len, n_actions, n_agents, fc1_dims=64, fc2_dims=64, lr=0.001, chkpt_dir='tmp/maddpg', name='critic'):
        super(CriticNetwork, self).__init__()
        self.bn1 = nn.BatchNorm1d(obs_len*n_agents+n_actions*n_agents)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.fill_(0)
        
        self.fc1 = nn.Linear(obs_len*n_agents+n_actions*n_agents, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)
        self.checkpoint_file = os.path.join(chkpt_dir, name)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, obs, actions):
        x = T.cat([obs, actions], dim=1)
        x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, obs_len, n_actions, fc1_dims=64, fc2_dims=64, lr=0.001, chkpt_dir='tmp/maddpg', name='actor'):
        super(ActorNetwork, self).__init__()
        self.bn1 = nn.BatchNorm1d(obs_len)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.fill_(0)
        
        self.fc1 = nn.Linear(obs_len, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        self.pi.weight.data.uniform_(-3e-3, 3e-3)
        
        self.checkpoint_file = os.path.join(chkpt_dir, name)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, obs):
        x = self.bn1(obs)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        pi = F.tanh(self.pi(x))
        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))