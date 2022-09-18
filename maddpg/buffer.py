import numpy as np

class ReplayBuffer:
    def __init__(self, mem_size, batch_size, agent_list, obs_size, critic_dims):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.batch_size = batch_size
        self.agent_list = agent_list
        self.n_agents = len(agent_list)

        self.critic_states = np.zeros((self.mem_size, critic_dims))
        self.critic_new_states = np.zeros((self.mem_size, critic_dims))
        self.states_mem = {agent: np.zeros((self.mem_size, obs_size)) for agent in agent_list}
        self.actions_mem = {agent: np.zeros(self.mem_size) for agent in agent_list}
        self.rewards_mem = {agent: np.zeros(self.mem_size) for agent in agent_list}
        self.new_states_mem = {agent: np.zeros((self.mem_size, obs_size)) for agent in agent_list}
        self.dones_mem = {agent: np.zeros(self.mem_size, dtype=np.bool) for agent in agent_list}

    def store_transition(self, states, actions, rewards, states_, dones):
        index = self.mem_cntr % self.mem_size
        critic_states = np.array([])
        critic_new_states = np.array([])
        for agent in self.agent_list:
            self.states_mem[agent][index] = states[agent]
            self.actions_mem[agent][index] = actions[agent]
            self.rewards_mem[agent][index] = rewards[agent]
            self.new_states_mem[agent][index] = states_[agent]
            self.dones_mem[agent][index] = dones[agent]

            for obs in states[agent]:
                critic_states = np.concatenate([critic_states, obs])
            for obs in states_[agent]:
                critic_new_states = np.concatenate([critic_states, obs])

        self.critic_states[index] = critic_states
        self.critic_new_states[index] = critic_new_states

        self.mem_cntr += 1

    def sample(self):
        max_mem = min(self.mem_cntr, self.mem_size) # makes sure we only sample the filled data
        batch = np.random.choice(max_mem, self.batch_size)
        critic_states = self.critic_states[batch]
        critic_states_ = self.critic_new_states[batch]
        states = {agent: self.states_mem[agent][batch] for agent in self.agent_list}
        actions = {agent: self.actions_mem[agent][batch] for agent in self.agent_list}
        rewards = {agent: self.rewards_mem[agent][batch] for agent in self.agent_list}
        states_ = {agent: self.new_states_mem[agent][batch] for agent in self.agent_list}
        dones = {agent: self.dones_mem[agent][batch] for agent in self.agent_list}

        return states, critic_states, actions, rewards, states_, critic_states_, dones

    def is_ready(self):
        return self.mem_cntr >= self.batch_size