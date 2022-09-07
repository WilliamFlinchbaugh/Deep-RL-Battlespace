import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_mem = np.zeros((self.mem_size, *input_shape))
        self.weight_mem = np.zeros((self.mem_size, n_actions))
        self.action_mem = np.zeros((self.mem_size, n_actions))
        self.shaped_rew_mem = np.zeros(self.mem_size)
        self.new_state_mem = np.zeros((self.mem_size, *input_shape))
        self.network_mem = np.zeros(self.mem_size)

    def store_transition(self, state, weight, action, shaped_rew, state_, network):
        index = self.mem_cntr % self.mem_size
        self.state_mem[index] = state
        self.weight_mem[index] = weight
        self.action_mem[index] = action
        self.shaped_rew_mem[index] = shaped_rew
        self.new_state_mem[index] = state_
        self.network_mem[index] = network
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_mem[batch]
        weights = self.weight_mem[batch]
        actions = self.action_mem[batch]
        shaped_rew = self.shaped_rew_mem[batch]
        states_ = self.new_state_mem[batch]
        networks = self.network_mem[batch]

        return states, weights, actions, shaped_rew, states_, networks

