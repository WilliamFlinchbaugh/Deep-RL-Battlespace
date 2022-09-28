import numpy as np

class ReplayBuffer:
    def __init__(self, mem_size, batch_size, agent_list, obs_size, critic_dims, n_actions):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.batch_size = batch_size
        self.agent_list = agent_list
        self.n_agents = len(agent_list)

        self.state_mem = np.zeros((self.mem_size, critic_dims))
        self.new_state_mem = np.zeros((self.mem_size, critic_dims))
        self.rew_mem = np.zeros((self.mem_size, self.n_agents))
        self.done_mem = np.zeros((self.mem_size, self.n_agents), dtype=bool)

        self.actor_states = []
        self.actor_new_states = []
        self.action_mem = []

        for agent in agent_list:
            self.actor_states.append(np.zeros((self.mem_size, obs_size)))
            self.actor_new_states.append(np.zeros((self.mem_size, obs_size)))
            self.action_mem.append(np.zeros((self.mem_size, n_actions)))

    def store_transition(self, states, actions, rewards, states_, dones):
        index = self.mem_cntr % self.mem_size
        # print(f"states: {states}\nactions: {actions}\nrewards: {rewards}\nstates_: {states_}\ndones: {dones}")
        _states = np.array([])
        _new_states = np.array([])
        _rewards = np.array([])
        _dones = np.array([], dtype=bool)
        for idx, agent in enumerate(self.agent_list):
            _states = np.concatenate([_states, states[agent]])
            _new_states = np.concatenate([_new_states, states_[agent]])
            _rewards = np.append(_rewards, rewards[agent])
            _dones = np.append(_dones, dones[agent])

            self.actor_states[idx][index] = states[agent]
            self.actor_new_states[idx][index] = states_[agent]
            self.action_mem[idx][index] = actions[agent]

        self.state_mem[index] = _states
        self.new_state_mem[index] = _new_states
        self.rew_mem[index] = _rewards
        self.done_mem[index] = _dones

        self.mem_cntr += 1

    def sample(self):
        max_mem = min(self.mem_cntr, self.mem_size) # makes sure we only sample the filled data
        batch = np.random.choice(max_mem, self.batch_size)

        states = self.state_mem[batch]
        states_ = self.new_state_mem[batch]
        rewards = self.rew_mem[batch]
        dones = self.done_mem[batch]

        actor_states = []
        actor_new_states = []
        actions = []

        for i in range(self.n_agents):
            actor_states.append(self.actor_states[i][batch])
            actor_new_states.append(self.actor_new_states[i][batch])
            actions.append(self.action_mem[i][batch])

        return np.array(actor_states), states, np.array(actions), rewards, np.array(actor_new_states), states_, dones

    def is_ready(self):
        return self.mem_cntr >= self.batch_size