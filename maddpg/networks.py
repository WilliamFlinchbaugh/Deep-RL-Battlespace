import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# From https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/DDPG/pytorch/lunar-lander/ddpg_torch.py
class CriticNetwork(nn.Module):
    def __init__(self, obs_len, n_actions, n_agents, fc1_dims=64, fc2_dims=64, lr=0.001, chkpt_dir='tmp/maddpg', name='critic'):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(obs_len*n_agents+n_actions*n_agents, fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(fc1_dims)
        
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(fc2_dims)
                
        self.q = nn.Linear(fc2_dims, 1)
        f3 = 0.003
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)
        
        self.checkpoint_file = os.path.join(chkpt_dir, name)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, obs, actions):
        x = T.cat([obs, actions], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.q(x)
        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, obs_len, n_actions, fc1_dims=64, fc2_dims=64, lr=0.001, chkpt_dir='tmp/maddpg', name='actor'):
        super(ActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(obs_len, fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)  
        self.bn1 = nn.LayerNorm(fc1_dims)
        
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)  
        self.bn2 = nn.LayerNorm(fc2_dims)
        
        self.pi = nn.Linear(fc2_dims, n_actions)
        f3 = 0.003
        T.nn.init.uniform_(self.pi.weight.data, -f3, f3)
        T.nn.init.uniform_(self.pi.bias.data, -f3, f3)
        self.pi.weight.data.uniform_(-3e-3, 3e-3)
        
        self.checkpoint_file = os.path.join(chkpt_dir, name)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, obs):
        x = F.relu(self.bn1(self.fc1(obs)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = T.tanh(self.pi(x))
        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))