import battle_v1
import numpy as np
import supersuit as ss
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, n_actions, agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.agents = agents
        self.n_agents = len(agents)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.actor_dims = actor_dims

        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, self.n_agents))
        self.terminal_memory = np.zeros((self.mem_size, self.n_agents), dtype=bool)

        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = {agent: np.zeros((self.mem_size, self.actor_dims)) for agent in self.agents}
        self.actor_new_state_memory = {agent: np.zeros((self.mem_size, self.actor_dims)) for agent in self.agents}
        self.actor_action_memory = {agent: np.zeros((self.mem_size, self.n_actions)) for agent in self.agents}

    def store_transition(self, raw_obs, state, action, reward, raw_obs_, state_, done):
        if self.mem_cntr % self.mem_size == 0 and self.mem_cntr > 0:
            self.init_actor_memory()

        index = self.mem_cntr % self.mem_size

        for agent in self.agents:
            self.actor_state_memory[agent][index] = raw_obs[agent]
            self.actor_new_state_memory[agent][index] = raw_obs_[agent]
            self.actor_action_memory[agent][index] = action[agent]

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = {}
        actor_new_states = {}
        actions = {}

        for agent in self.agents:
            actor_states[agent] = self.actor_state_memory[agent][batch]
            actor_new_states[agent] = self.actor_new_state_memory[agent][batch]
            actions[agent] = self.actor_action_memory[agent][batch]

        return actor_states, states, actions, rewards, actor_new_states, states_, terminal

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True
        return False

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_agents, n_actions, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims+n_agents*n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:@' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = F.relu.fc1(T.cat([state, action], dim=1))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__Init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:@' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = T.softmax(self.pi(x), dim=1)
        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_id, chkpt_dir, alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.99, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, agent_id+'_actor', chkpt_dir)
        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions, agent_id+'_critic', chkpt_dir)
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, agent_id+'_target_actor', chkpt_dir)
        self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions, agent_id+'_target_critic', chkpt_dir)

        self.update_network_parameters(tau=1)
        
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_state_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)
                
        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_state_dict[name].clone()
        self.target_critic.load_state_dict(critic_state_dict)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        actions = self.actor.forward(state)
        noise = T.rand(self.n_actions).to(self.actor.device)
        action = actions + noise

        return action.detach().cpu().numpy()[0]

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()


class MADDPG:
    def __init__(self, actor_dims, critic_dims, agents, n_actions, scenario='simple', alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.99, tau=0.01, chkpt_dir='checkpoints/'):
        self.n_agents = len(agents)
        self.agents = {agent: Agent(actor_dims[agent], critic_dims, n_actions, self.n_agents, agent, chkpt_dir, alpha, beta) for agent in agents}

        self.n_actions = n_actions
        chkpt_dir += scenario

    def save_checkpoint(self):
        print(' === Saving Checkpoint ===')
        for agent in self.agents.values():
            agent.save_models()

    def load_checkpoint(self):
        print(' === Loading Checkpoint ===')
        for agent in self.agents.values():
            agent.load_models()

    def choose_action(self, )


cf = {
    'n_agents': 2, # Number of planes on each team
    'show': True, # Show visuals
    'hit_base_reward': 1, # Reward value for hitting enemy base
    'hit_plane_reward': 1, # Reward value for hitting enemy plane
    'miss_punishment': 0, # Punishment value for missing a shot
    'die_punishment': 0, # Punishment value for a plane dying
    'fps': 15 # Framerate that the visuals run at
}

env = battle_v1.parallel_env(**cf)
env = ss.black_death_v3(env)

