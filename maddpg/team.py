from maddpg.buffer import ReplayBuffer
from maddpg.networks import CriticNetwork
from maddpg.agent import NetworkedAgent
import torch as T
import torch.nn.functional as F

class Team:
    def __init__(self, agent_list, obs_size, n_actions, critic_dims, fc1_dims, fc2_dims, mem_size, batch_size, gamma, lr, chkpt_dir):
        self.batch_size = batch_size
        self.agent_list = agent_list # List of the string names of all agents in this team
        self.agents = {}
        for idx, agent in enumerate(agent_list):
            self.agents[agent] = NetworkedAgent(agent_list, n_actions, obs_size, agent, len(agent_list), fc1_dims, fc2_dims, gamma, lr, chkpt_dir) # Create an agent for each agent in the team
        self.memory = ReplayBuffer(mem_size, batch_size, agent_list, obs_size, critic_dims, n_actions)

    def choose_actions(self, observations): # Call each agent's choose_action method
        actions = {}
        for agent_id, agent in self.agents.items():
            actions[agent_id] = agent.choose_action(observations[agent_id])
        return actions

    def learn(self):
        if not self.memory.is_ready():
            return

        T.autograd.set_detect_anomaly(True)

        device = self.agents[self.agent_list[0]].actor.device

        actor_states, states, actions, rewards, actor_new_states, states_, dones = self.memory.sample()

        # Turn all into tensors
        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = [] # Actions returned from the target actor when sent new_states
        all_agents_new_mu_actions = [] # Actions returned from the actor when sent new_states
        old_agents_actions = [] # Actions returned from the actor when sent states

        # Gather all actions from all agents
        for idx, agent_id in enumerate(self.agent_list):
            agent = self.agents[agent_id]
            new_states = T.tensor(actor_new_states[idx], dtype=T.float).to(device)
            new_pi = agent.target_actor.forward(new_states)
            all_agents_new_actions.append(new_pi)

            mu_states = T.tensor(actor_states[idx], dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)

            old_agents_actions.append(actions[idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

        # Zero all actor gradients
        for idx, agent_id in enumerate(self.agent_list):
            self.agents[agent_id].actor.optimizer.zero_grad()

        # Update the critic and actor for each agent
        for idx, agent_id in enumerate(self.agent_list):
            agent = self.agents[agent_id]

            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            critic_value_[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:,idx] + agent.gamma*critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            actor_loss.backward(retain_graph=True)

        # Step the actor optimizers and update the target networks
        for idx, agent_id in enumerate(self.agent_list):
            self.agents[agent_id].actor.optimizer.step()
            self.agents[agent_id].update_network_parameters()
    
    # Deal with noise
    def scale_noise(self, scale):
        for agent in self.agents.values():
            agent.scale_noise(scale)
            
    def reset_noise(self):
        for agent in self.agents.values():
            agent.reset_noise()

    # Save and load all agent models
    def save_models(self):
        for agent in self.agents.values():
            agent.save_models()

    def load_models(self):
        for agent in self.agents.values():
            agent.load_models()