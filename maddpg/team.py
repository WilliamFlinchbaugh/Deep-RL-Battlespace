from buffer import ReplayBuffer
from networks import CriticNetwork
from agent import NetworkedAgent

class Team:
    def __init__(self, agent_list, n_actions, obs_spaces, action_spaces, mem_size=1000000, batch_size=128, gamma=0.95):
        self.agent_list = agent_list
        self.agents = {}
        for idx, agent in enumerate(agent_list):
            self.agents[agent] = NetworkedAgent(agent_list, action_spaces[idx], obs_spaces[idx], agent)
        self.memory = ReplayBuffer(mem_size, batch_size, agent_list, obs_spaces)

    def choose_actions(self, observations):
        actions = {}
        for agent_id, agent in self.agents.items():
            actions[agent_id] = agent.choose_action(observations[agent_id])
        return actions

    def learn(self):
        if not self.memory.is_ready():
            return
        
        


    def save_models(self):
        for agent in self.agents.values():
            agent.save_models()

    def load_modles(self):
        for agent in self.agents.values():
            agent.load_models()