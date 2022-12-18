import math
import numpy as np

class InstinctAgent:
    def __init__(self, agent_list, enemy_list, env):
        self.agent_list = agent_list
        self.enemy_list = enemy_list
        self.env = env

    def choose_action(self, observation):
        # Collect the observations and calculate the actual data
        enemy_info = {}

        # Base info
        enemy_info['base_dist'] = (observation[0] + 1) / 2 * (math.sqrt(math.pow(self.env.width, 2) + math.pow(self.env.height, 2)))
        enemy_info['base_angle'] = observation[1] * 360

        # Enemy info
        index = 2
        for enemy in self.enemy_list:
            enemy_info[f"{enemy}_alive"] = observation[index]
            enemy_info[f"{enemy}_dist"] = (observation[index+1] + 1) / 2 * (math.sqrt(math.pow(self.env.width, 2) + math.pow(self.env.height, 2)))
            enemy_info[f"{enemy}_angle"] = observation[index+2] * 360
            index += 3
        
        # Calculate the "scores" for each enemy to find target
        enemy_scores = []
        enemy_scores.append(enemy_info[f"base_dist"] * abs(enemy_info[f"base_angle"]))
        for enemy in self.enemy_list:
            if enemy_info[f"{enemy}_alive"] == 1:
                enemy_scores.append(enemy_info[f"{enemy}_dist"] * abs(enemy_info[f"{enemy}_angle"]))
            else:
                enemy_scores.append(1000000) # if dead, set to a very high value

        # Choose the target
        if min(enemy_scores) == enemy_scores[0]:
            target = "base"
        else:
            target = self.enemy_list[enemy_scores.index(min(enemy_scores)) - 1]

        if self.env.continuous_actions: # If continuous actions
            actions = [0, 0, 0]
            if enemy_info[f"{target}_dist"] < self.env.shot_dist / 3 * 2 and abs(enemy_info[f"{target}_angle"]) < 20: # If within 2/3 of shot distance and within 20 degrees
                actions[2] = 1 if np.random.rand() > 0.6 else -1 # shoot if it passes the 60/40
            actions[0] = enemy_info[f"{target}_dist"] / math.sqrt(math.pow(self.env.width, 2) + math.pow(self.env.height, 2)) * 2 - 1 # Calculate speed based on distance to target
            if enemy_info[f"{target}_angle"] > 0: # If aiming to the right of target, turn left
                actions[1] = max(-enemy_info[f"{target}_angle"] / self.env.max_turn, -1)
            else: # If aiming to the left of target, turn right
                actions[1] = min(-enemy_info[f"{target}_angle"] / self.env.max_turn, 1)
            
            noise = np.random.uniform(-0.15, 0.15, size=3) # Add a small bit of noise to the actions
            actions = np.clip(actions + noise, -1, 1)

            return actions

        else: # If discrete actions
            if enemy_info[f"{target}_dist"] < self.env.shot_dist / 2 and abs(enemy_info[f"{target}_angle"]) < 20:
                return 1
            if enemy_info[f"{target}_angle"] > 0:
                return 3
            else:
                return 2
