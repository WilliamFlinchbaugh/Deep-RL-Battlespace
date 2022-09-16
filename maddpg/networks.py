import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CriticNetwork(nn.Module):
    def __init__(self, input_size, n_actions, n_agents, fc1_dims=64, fc2_dims=64, lr=0.01, chkpt_dir='tmp/maddpg', name='critic'):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size*n_agents+n_actions*n_agents, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)
        self.checkpoint_file = os.path.join(chkpt_dir, name)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observations, actions):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, input_size, n_actions, fc1_dims=64, fc2_dims=64, lr=0.01, chkpt_dir='tmp/maddpg', name='actor'):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        self.checkpoint_file = os.path.join(chkpt_dir, name)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        pi = T.softmax(self.pi(x), dim=1)
        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))