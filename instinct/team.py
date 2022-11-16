from instinct.agent import InstinctAgent

class Team:
    def __init__(self, agent_list, enemy_list, env):
        self.agent_list = agent_list
        self.enemy_list = enemy_list
        self.agents = {}
        for idx, agent in enumerate(agent_list):
            self.agents[agent] = InstinctAgent(agent_list, enemy_list, env)
    
    def choose_actions(self, observations):
        actions = {}
        for agent_id, agent in self.agents.items():
            actions[agent_id] = agent.choose_action(observations[agent_id])
        return actions