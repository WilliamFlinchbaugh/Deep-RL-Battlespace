from maddpg.networks import ActorNetwork, CriticNetwork
import torch as T
import numpy as np
from utils.noise import OUNoise

class NetworkedAgent:
    def __init__(self, agent_list, n_actions, obs_len, name, n_agents, fc1_dims, fc2_dims, gamma, lr, chkpt_dir):
        self.agent_list = agent_list
        self.n_actions = n_actions
        self.obs_len = obs_len
        self.name = name
        self.tau = 0.01
        self.gamma = gamma
        self.timestep = 0
        self.noise = OUNoise(self.n_actions)

        self.actor = ActorNetwork(obs_len, n_actions, fc1_dims, fc2_dims, lr, chkpt_dir, f'actor_{name}')
        self.target_actor = ActorNetwork(obs_len, n_actions, fc1_dims, fc2_dims, lr, chkpt_dir, f'target_actor_{name}')
        self.critic = CriticNetwork(obs_len, n_actions, n_agents, fc1_dims, fc2_dims, lr, chkpt_dir, f'critic_{name}')
        self.target_critic = CriticNetwork(obs_len, n_actions, n_agents, fc1_dims, fc2_dims, lr, chkpt_dir, f'target_critic_{name}')

        self.update_network_parameters(tau=1)
    
    def choose_action(self, observation):
        self.actor.eval()
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.actor.device)
        actions = self.actor(state)
        actions += T.tensor(self.noise.noise(), dtype=T.float).to(self.actor.device)
        actions = actions.clamp(-1, 1)
        self.actor.train()
        return actions.detach().cpu().numpy()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_state_dict = dict(target_critic_params)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_state_dict[name].clone()

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_state_dict[name].clone()
        
        self.target_actor.load_state_dict(actor_state_dict)
        self.target_critic.load_state_dict(critic_state_dict)
        
    def reset_noise(self):
        self.noise.reset()
        
    def scale_noise(self, scale):
        self.noise.scale = scale

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