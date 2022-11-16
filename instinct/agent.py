import math

class InstinctAgent:
    def __init__(self, agent_list, enemy_list, env):
        self.agent_list = agent_list
        self.enemy_list = enemy_list
        self.env = env

    def choose_action(self, observation):
        enemy_info = {}
        enemy_info['base_dist'] = (observation[0] + 1) / 2 * (math.sqrt(math.pow(self.env.width, 2) + math.pow(self.env.height, 2)))
        enemy_info['base_angle'] = observation[1] * 360
        index = 2
        for enemy in self.enemy_list:
            enemy_info[f"{enemy}_alive"] = observation[index]
            enemy_info[f"{enemy}_dist"] = (observation[index+1] + 1) / 2 * (math.sqrt(math.pow(self.env.width, 2) + math.pow(self.env.height, 2)))
            enemy_info[f"{enemy}_angle"] = observation[index+2] * 360
            index += 3
        
        enemy_scores = []
        enemy_scores.append(enemy_info[f"base_dist"] * abs(enemy_info[f"base_angle"]) * 8/9)
        for enemy in self.enemy_list:
            if enemy_info[f"{enemy}_alive"] == 1:
                enemy_scores.append(enemy_info[f"{enemy}_dist"] * abs(enemy_info[f"{enemy}_angle"]))
            else:
                enemy_scores.append(1000000) # if dead, set to a very high value

        if min(enemy_scores) == enemy_scores[0]:
            target = "base"
        else:
            target = self.enemy_list[enemy_scores.index(min(enemy_scores)) - 1]

        actions = [0, 0, 0]
        if enemy_info[f"{target}_dist"] < self.env.shot_dist and abs(enemy_info[f"{target}_angle"]) < 30:
            actions[2] = 1 # shoot!
        actions[0] = enemy_info[f"{target}_dist"] / math.sqrt(math.pow(self.env.width, 2) + math.pow(self.env.height, 2)) * 2 - 1
        if enemy_info[f"{target}_angle"] > 0:
            actions[1] = max(-enemy_info[f"{target}_angle"] / self.env.max_turn, -1)
        else:
            actions[1] = min(-enemy_info[f"{target}_angle"] / self.env.max_turn, 1)

        return actions
