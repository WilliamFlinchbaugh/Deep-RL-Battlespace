import numpy as np

class ReplayBuffer:
    def __init__(self, mem_size, batch_size, agent_list, obs_size):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.batch_size = batch_size
        self.agent_list = agent_list
        self.n_agents = len(agent_list)

        self.states_mem = np.zeros((self.mem_size, self.n_agents, obs_size), dtype=np.float32)
        self.actions_mem = np.zeros((self.mem_size, self.n_agents), dtype=np.float32)
        self.rewards_mem = np.zeros((self.mem_size, self.n_agents), dtype=np.float32)
        self.new_states_mem = np.zeros((self.mem_size, self.n_agents, obs_size), dtype=np.float32)
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