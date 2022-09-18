from buffer import ReplayBuffer
from networks import CriticNetwork
from agent import NetworkedAgent

class Team:
    def __init__(self, agent_list, obs_spaces, n_actions, mem_size=1000000, batch_size=128, gamma=0.95):
        self.agent_list = agent_list
        self.agents = {}
        for idx, agent in enumerate(agent_list):
            self.agents[agent] = NetworkedAgent(agent_list, n_actions, obs_spaces[idx].shape[0], agent)
        self.memory = ReplayBuffer(mem_size, batch_size, agent_list, obs_spaces)

    def choose_actions(self, observations):
        actions = {}
        for agent_id, agent in self.agents.items():
            actions[agent_id] = agent.choose_action(observations[agent_id])
        return actions

    def learn(self):
        if not self.memory.is_ready():
            return

        device = self.agents[self.agent_list[0]].actor.device

        states, critic_states, actions, rewards, new_states, critic_new_states, dones = self.memory.sample()
        
        for idx, agent in self.agents.items():
            _critic_states = T.tensor(critic_states, dtype=T.float).to(self.actor.device)
            _critic_new_states = T.tensor(critic_new_states, dtype=T.float).to(self.actor.device)
            _states = T.tensor(states[idx], dtype=T.float).to(self.actor.device)
            _actions = T.tensor(actions[idx], dtype=T.float).to(self.actor.device)
            _rewards = T.tensor(rewards[idx], dtype=T.float).to(self.actor.device)
            _new_states = T.tensor(new_states[idx], dtype=T.float).to(self.actor.device)
            _dones = T.tensor(dones[idx], dtype=T.bool).to(self.actor.device)

            target_actions = agent.target_actor.forward(_critic_new_states)
            critic_value_ = agent.target_critic.forward(_critic_new_states, target_actions)
            critic_value = agent.critic.forward(_critic_states, _actions)

            critic_value_[dones] = 0.0
            critic_value_ = critic_value_.view(-1)

            target = rewards + self.gamma*critic_value_
            target = target.view(self.batch_size, 1)

            self.critic.optimizer.zero_grad()
            critic_loss = F.mse_loss(target, critic_value)
            critic_loss.backward()
            self.critic.optimizer.step()
            
            self.actor.optimizer.zero_grad()
            actor_loss = -self.critic.forward(states, self.actor.forward(states))
            actor_loss = T.mean(actor_loss)
            actor_loss.backward()
            self.actor.optimizer.step()

            self.update_network_parameters()
        


    def save_models(self):
        for agent in self.agents.values():
            agent.save_models()

    def load_modles(self):
        for agent in self.agents.values():
            agent.load_models()