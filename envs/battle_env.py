from envs.sprites import Plane, Base, Bullet, Explosion
import envs.sprites as sprites
import pygame
from pygame.locals import *
import numpy as np
import math
from gym import spaces
from gym.utils import EzPickle
import os
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec
import vidmaker
import cv2
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Needed or else it sometimes causes issues on windows machines

def env(**kwargs):
    """
    Wraps the raw environment in useful PZ wrappers
    Returns the wrapped environment
    """    
    env = raw_env(**kwargs)
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(**kwargs):
    """
    Wraps the parallel environment in the parallel_to_aec wrapper
    Returns the AEC environment
    """    
    env = parallel_env(**kwargs)
    env = parallel_to_aec(env)
    return env

# ---------- HELPER FUNCTIONS -----------
def rel_angle(p0, a0, p1):
    """
    Calculates the relative angle of position1 based on the angle0 and position0
    Returns the calculated value in degrees
    ** Don't ask me how the math works I spent hours stealing code to get it to work **
    """    
    dx = p0[0] - p1[0]
    dy = p0[1] - p1[1]
    rads = math.atan2(dy,dx)
    rads %= 2*math.pi
    degs = math.degrees(rads)
    rel_angle = 180 + a0 - (360 - degs)
    if rel_angle < -180: rel_angle += 360
    if rel_angle > 180: rel_angle -= 360
    return rel_angle

def dist(p1,p0):
    """
    Returns the distance between two points
    """    
    return math.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)

# ----------- BATTLE ENVIRONMENT -----------
class parallel_env(ParallelEnv, EzPickle):
    """PettingZoo Environment taking parallel actions

    Args:
        EzPickle: Needed to fix pickling issue when wrapping the environment
    """

    metadata = {
        "render_modes": ["human"],
        "name": "battle_env_v1"
    }

    def __init__(self, n_agents=1, show=False, hit_base_reward=100, hit_plane_reward=10, miss_punishment=-1, die_punishment=-5, lose_punishment=-20, fps=20, continuous_actions=False):
        """Initializes values, observation spaces, action spaces, etc.

        Args:
            n_agents (int, optional): The number of agents per team. Defaults to 1.
            show (bool, optional): Whether or not to show the pygame visuals. Defaults to False.
            hit_base_reward (int, optional): Reward value for hitting enemy base. Defaults to 10.
            hit_plane_reward (int, optional): Reward value for hitting enemy plane. Defaults to 2.
            miss_punishment (int, optional): Punishment value for missing a bullet. Defaults to 0.
            die_punishment (int, optional): Punishment value for plane dying. Defaults to -3.
            fps (int, optional): Framerate for pygame visualization to run at. Defaults to 20.
        """
        EzPickle.__init__(self)
        self.n_agents = n_agents # n agents per team

        pygame.init() # Initialize pygame module

        # Set the hitpoints for the planes and bases
        self.base_hp = 5 * self.n_agents
        self.plane_hp = 4

        # Initializing the team dictionaries
        self.team = {}
        self.team['red'] = {}
        self.team['blue'] = {}
        self.team['red']['base'] = Base('red', self.base_hp)
        self.team['blue']['base'] = Base('blue', self.base_hp)
        self.team['red']['planes'] = {}
        self.team['blue']['planes'] = {}
        self.team['red']['wins'] = 0
        self.team['blue']['wins'] = 0

        # Possible_agents is every possible agent, agents is agents that are currently alive
        self.possible_agents = [f"plane{r}" for r in range(self.n_agents * 2)]
        self.possible_red = self.possible_agents[:self.n_agents]
        self.possible_blue = self.possible_agents[self.n_agents:]
        self.agents = self.possible_agents[:]

        # Creates the planes and makes a map so that they are easier to find
        self.team_map = {}
        for x in self.possible_red:
            self.team_map[x] = 'red'
            self.team['red']['planes'][x] = Plane('red', self.plane_hp, x)
        for x in self.possible_blue:
            self.team_map[x] = 'blue'
            self.team['blue']['planes'][x] = Plane('blue', self.plane_hp, x)

        """
        Observation space contains the following:
        - Distance to enemy base
        - Angle to enemy base
        - If each enemy plane is alive
        - Distance to each enemy plane
        - Angle to each enemy plane

        The observations are normalized between [-1, 1]]
        The observation space for each agent is (3 * n_agents + 2)
        """

        self.obs_size = 3 * n_agents + 2 # Length of observation for each agent
        high = np.ones(self.obs_size, dtype=np.float32)
        obs_space = spaces.Box(high, -high) # Observation space for each agent
        self.observation_spaces = {agent: obs_space for agent in self.possible_agents} # Dictionary containing an observation space for each agent
        
        """
        Action space contains the following:
        - 0: Forward
        - 1: Shoot
        - 2: Turn Left
        - 3: Turn Right
        """
        self.continuous_actions = continuous_actions
        if (self.continuous_actions):
            """
            Action is a vectors of [speed [min speed, max speed], angle [-max turn, max turn], shoot? [-1, 1] if > 0 shoot else dont
            They are all normalized into [-1, 1]
            """
            self.n_actions = 3
            self.max_turn = 35
            self.max_speed = 275
            self.min_speed = 200
            high = np.ones(self.n_actions, dtype=np.float32)
            action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        else:
            self.n_actions = 4
            self.step_turn = 15 # degrees to turn per step
            self.speed = 215 # mph
            action_space = spaces.Discrete(self.n_actions)

        self.action_spaces = {agent: action_space for agent in self.possible_agents} # Dictionary containing an action space for each agent
        
        # ---------- Initialize values ----------
        self.continuous_actions = continuous_actions
        self.width = sprites.DISP_WIDTH
        self.height = sprites.DISP_HEIGHT
        self.max_time = 10 + (self.n_agents * 2)
        self.total_games = 0
        self.ties = 0
        self.bullets = []
        self.explosions = []
        self.bullet_speed = 450 # mph
        self.shot_dist = 500 # Distance a bullet can travel before it disappears
        self.total_time = 0 # in hours
        self.time_step = 0.1 # hours per time step
        self.show = show # show the pygame animation
        self.hit_base_reward = hit_base_reward
        self.hit_plane_reward = hit_plane_reward
        self.miss_punishment = miss_punishment
        self.die_punishment = die_punishment
        self.lose_punishment = lose_punishment
        self.fps = fps
        self.recording = False

    def observation_space(self, agent):
        """Gets the observation space for a given agent

        Args:
            agent (string): Agent ID (as stored in self.possible_agents)
        """
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """Gets the action space for a given agent

        Args:
            agent (string): Agent ID (as stored in self.possible_agents)
        """
        return self.action_spaces[agent]
    
    def observe(self, agent):
        """Returns an observation for a given agent

        Args:
            agent (string): Agent ID (as stored in self.possible_agents)

        Returns:
            observation (np.array): Contains the values of an observation for the agent
        """

        agent_team = self.team_map[agent]

        # Sets default values that will be changed if the planes are alive
        obs = -np.ones(self.obs_size, dtype=np.float32)

        if not (agent in self.agents and agent in self.team[agent_team]['planes']): # If this plane (the one observing) is dead
            return obs

        # Gathers values for observing
        agent_plane = self.team[agent_team]['planes'][agent]
        ocolor = 'blue' if agent_team == 'red' else 'red'
        oplane_ids = self.possible_blue if ocolor == 'blue' else self.possible_red
        agent_pos = agent_plane.get_pos()
        agent_dir = agent_plane.get_direction()
        obase = self.team[ocolor]['base']
        obase_pos = obase.get_pos()

        # Values for enemy base
        obs[0] = dist(agent_pos, obase_pos) / (math.sqrt(math.pow(self.width, 2) + math.pow(self.height, 2))) * 2 - 1
        obs[1] = rel_angle(agent_pos, agent_dir, obase_pos) / 360
        index = 2

        # Values for each enemy plane
        for id in oplane_ids:
            if id in self.agents: # if that plane is alive
                plane = self.team[self.team_map[id]]['planes'][id]
                plane_pos = plane.get_pos()
                obs[index] = 1 # Plane is alive
                obs[index + 1] = dist(agent_pos, plane_pos) / (math.sqrt(math.pow(self.width, 2) + math.pow(self.height, 2))) * 2 - 1 # Distance to plane
                obs[index + 2] = rel_angle(agent_pos, agent_dir, plane_pos) / 360 # Angle to plane
            index += 3

        return np.array(obs, dtype=np.float32) # Cast the dict to np.array 

    def reset(self, seed=None, return_info=False, options=None):
        """Reset all of the values so that the game can be restarted

        Returns:
            observations (dict): Initial observations of each agent
        """
        
        # Reset the winner
        self.winner = 'none'

        # Reset the bases
        self.team['red']['base'].reset()
        self.team['blue']['base'].reset()

        # Delete all of the planes
        self.team['red']['planes'].clear()
        self.team['blue']['planes'].clear()

        # Re-populate the planes in the team dicts
        for x in self.possible_red:
            self.team['red']['planes'][x] = Plane('red', self.plane_hp, x)
        for x in self.possible_blue:
            self.team['blue']['planes'][x] = Plane('blue', self.plane_hp, x)

        self.display = pygame.Surface((self.width, self.height)) # Reset the pygame display
        self.total_time = 0 # Reset the time
        self.bullets = [] # Clear all of the bullets
        self.rendering = False # Not currently rendering (used to initiate display)
        self.agents = self.possible_agents[:] # Resetting all agents alive
        self.dones = {agent: False for agent in self.possible_agents} # No agents are currently done
        self.env_done = False # Environment is not done

        observations = {agent: self.observe(agent) for agent in self.possible_agents} # Get observations for each agent
        return observations

    def step(self, actions):
        """Takes steps for each agent

        Args:
            actions (dict): Dictionary containing the actions of each agent

        Returns:
            observations (dict): Dictionary containing the observation of each agent
            rewards (dict): Dictionary containing the rewards of each agent
            dones (dict): Dictionary indicating which agents are done and should be skipped over
            infos (dict): Used for extra info (not utilized)
        """

        # clip the actions to the action space
        if self.continuous_actions:
            for key, value in actions.items():
                actions[key] = np.clip(value, self.action_space(key).low, self.action_space(key).high)

        # Initialize all rewards to 0
        rewards = {agent: 0 for agent in self.possible_agents}

        # If env is done just return some empty info
        if self.env_done:
            observations = {agent: self.observe(agent) for agent in self.possible_agents} # Get observation for each agent
            infos = {agent: {} for agent in self.possible_agents} # Empty info for each agent
            return observations, rewards, self.dones, infos 

        # If passing no actions or no agents alive, then we have a tie because all agents are dead
        if len(actions) == 0 or len(self.agents) == 0:
            self.tie()
            observations = {agent: self.observe(agent) for agent in self.possible_agents} # Get observation for each agent
            infos = {agent: {} for agent in self.possible_agents} # Empty info for each agent
            return observations, rewards, self.dones, infos 

        # Increment time
        self.total_time += self.time_step

        # Check for tie
        if self.total_time >= self.max_time: # If over the max time
            self.tie()
            observations = {agent: self.observe(agent) for agent in self.possible_agents} # Get observation for each agent
            infos = {agent: {} for agent in self.possible_agents} # Empty info for each agent
            return observations, rewards, self.dones, infos

        for agent_id in self.agents: # Carry out actions for each agent that is alive
            action = actions[agent_id] # Grab action for this agent 
            if type(action) == np.ndarray and not self.continuous_actions: # If the action is a numpy array and the action space is discrete
                action = np.argmax(action)
            self.process_action(action, agent_id) # Perform the action

        # Move every bullet and check for hits
        for bullet in self.bullets[:]:
            # Move bullet and gather outcome
            outcome = bullet.update(self.width, self.height, self.time_step)

            # Kill bullet if miss
            if outcome == 'miss':
                rewards[bullet.agent_id] += self.miss_punishment # Issue punishment for missing
                self.bullets.remove(bullet) # Kill the bullet

            # Kill bullet and provide reward if hits base
            elif isinstance(outcome, Base):
                outcome.hit() # Damage the base 
                rewards[bullet.agent_id] += self.hit_base_reward # Issue the reward for hitting
                self.bullets.remove(bullet) # Kill the bullet
            
            # Kill bullet and provide reward if hits plane
            elif isinstance(outcome, Plane):
                outcome.hit() # Damage the plane
                rewards[bullet.agent_id] += self.hit_plane_reward # Issue reward for hitting plane
                self.bullets.remove(bullet) # Kill the bullet

                # Plane is dead
                if not outcome.alive:
                    if self.show:
                        self.explosions.append(Explosion(outcome.get_pos())) # Create an explosion
                    self.agents.remove(outcome.id) # Remove the agent from self.agents
                    self.team[outcome.team]['planes'].pop(outcome.id) # Remove the plane from its team
                    rewards[outcome.id] += self.die_punishment # Issue punishment for dying
                    self.dones[outcome.id] = True # Set that agent's done to True
        
        # Check if red won game
        if not self.team['blue']['base'].alive:
            for plane in self.possible_red:
                rewards[plane] += self.lose_punishment
            self.win('red')

        # Check if blue won game
        if not self.team['red']['base'].alive:
            for plane in self.possible_blue:
                rewards[plane] += self.lose_punishment
            self.win('blue')

        # Render the environment
        if self.show:
            self.render()
        
        observations = {agent: self.observe(agent) for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}
        
        return observations, rewards, self.dones, infos
    
    def process_action(self, action, agent_id):
        """Processes an action for a single agent

        Args:
            action (int): The action that an agent is taking
            agent_id (string): Agent ID as given in self.possible_agents
        """
        if agent_id not in self.team['red']['planes'] and agent_id not in self.team['blue']['planes']: # if the agent is dead
            return

        # Get some info about the agent
        team = 'red' if agent_id in self.team['red']['planes'] else 'blue'
        agent = self.team[team]['planes'][agent_id]
        oteam = 'blue' if team == 'red' else 'red'
        agent_pos = agent.get_pos()
        agent_dir = agent.get_direction()
        if not self.continuous_actions:
            # --------------- FORWARD ---------------
            if action == 0: 
                agent.forward(self.speed, self.time_step) # Move the plane forward

            # --------------- SHOOT ---------------
            elif action == 1:
                self.bullets.append(Bullet(agent_pos[0], agent_pos[1], agent_dir, self.bullet_speed, team, self.team[oteam], agent_id, self.shot_dist)) # Shoot a bullet
                agent.forward(self.speed, self.time_step) # Move the plane forward
            
            # --------------- TURN LEFT ---------------
            elif action == 2:
                agent.rotate(self.step_turn) # Rotate the plane
                agent.forward(self.speed, self.time_step) # Move the plane forward

            # ---------------- TURN RIGHT ----------------
            elif action == 3:
                agent.rotate(-self.step_turn) # Rotate the plane
                agent.forward(self.speed, self.time_step) # Move the plane forward
        else:
            speed = ((action[0] + 1)/2) * (self.max_speed - self.min_speed) + self.min_speed # Calculate speed from input
            agent.forward(speed, self.time_step) # Move forward
            turn_angle = action[1] * self.max_turn # Calculate angle to turn from input
            agent.rotate(turn_angle) # Rotate
            if action[2] > 0: # Check if shoot
                self.bullets.append(Bullet(agent_pos[0], agent_pos[1], agent_dir, self.bullet_speed, team, self.team[oteam], agent_id, self.shot_dist)) # Shoot a bullet
            
    def winner_screen(self):
        """
        Display the winner of the game when the game is over
        """
        if self.show: # Makes sure that we are visualizing
            font = pygame.font.Font(pygame.font.get_default_font(), 32)
            if self.winner != 'none' and self.winner != 'tie':
                text = font.render(f"THE WINNER IS {self.winner.upper()}", True, sprites.BLACK)
                textRect = text.get_rect()
                textRect.center = (self.width//2, self.height//2)
            else:
                text = font.render(f"THE GAME IS A TIE", True, sprites.BLACK)
                textRect = text.get_rect()
                textRect.center = (self.width//2, self.height//2)
            self.display.blit(text, textRect)
            pygame.display.update()

            if self.fps <= 60:
                pygame.time.wait(500) # Pause the last frame
                if self.recording:
                    for _ in range(15): # Pause the last frame of video for like a second
                        self.video.update(cv2.cvtColor(pygame.surfarray.pixels3d(self.display).swapaxes(0, 1), cv2.COLOR_BGR2RGB))

    def wins(self):
        """Gives a nice output of the wins for each team and the winrate of the red team

        Returns:
            string: Formatted string of the # of wins, ties, and winrate
        """
        return "Wins by red: {}\nWins by blue: {}\nTied games: {}\nWin rate: {}".format(self.team['red']['wins'], self.team['blue']['wins'], self.ties, self.team['red']['wins']/self.total_games)

    def close(self):
        """
        Closes the pygame display
        """
        pygame.display.quit()
    
    def make_discrete(self, actions_dict):
        discrete_actions_dict = {}
        for agent, action in actions_dict.items():
            discrete_actions_dict[agent] = np.argmax(action)
        return discrete_actions_dict

    def tie(self):
        """
        Ends the game in a tie
        Sets all the agents to done and adds to the tie total
        """
        self.winner = 'tie'
        self.total_games += 1 
        self.ties += 1
        self.env_done = True
        self.dones = {agent: True for agent in self.possible_agents}
        if self.show:
            self.render()

    def win(self, winner):
        """
        Ends the game in a win for winner
        Sets all the agents to done and adds to the win total for the winner

        Args:
            winner (string): Represents the team that won the game
        """
        self.winner = winner
        self.total_games += 1 
        self.team[winner]['wins'] += 1
        self.env_done = True
        self.dones = {agent: True for agent in self.possible_agents}
        if self.show:
            self.render()

    def render(self, mode="human"):
        """
        Renders the pygame visuals
        """

        # Just to ensure it won't render if self.show is False
        if not self.show: 
            self.rendering = False
            return

        # We need to initialize everything if not yet rendering
        if not self.rendering: 
            self.rendering = True
            pygame.display.init()
            pygame.font.init()
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Battlespace Simulator")
            self.clock = pygame.time.Clock()
            if self.fps <= 60:
                pygame.time.wait(500)

        # Check if we should quit
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE: # If pressed escape
                    self.close()
            elif event.type == QUIT: # If trying to close the window
                self.close()

        # Fill background
        self.display.fill(sprites.WHITE)

        # Draw bullets
        for bullet in self.bullets:
            bullet.draw(self.display)

        # Draw explosions
        for explosion in self.explosions:
            explosion.draw(self.display)
                
        # Draw bases
        self.team['red']['base'].draw(self.display)
        self.team['blue']['base'].draw(self.display)

        # Draw planes
        for plane in self.team['red']['planes'].values():
            if plane.alive:
                plane.update()
                plane.draw(self.display)
        for plane in self.team['blue']['planes'].values():
            if plane.alive:
                plane.update()
                plane.draw(self.display)

        # Calls winner screen if done
        if self.winner != 'none':
            self.winner_screen()

        # Update the display, update the video if recording, and tick the clock with the framerate
        if self.recording:
            self.video.update(cv2.cvtColor(pygame.surfarray.pixels3d(self.display).swapaxes(0, 1), cv2.COLOR_BGR2RGB))
        pygame.display.update()
        self.clock.tick(self.fps)

    def start_recording(self, path):
        """
        Starts recording a video of the rendering

        Args:
            path (string): The path to save the video to
        """
        self.recording = True
        self.video = vidmaker.Video(path, fps=self.fps, resolution=(self.width, self.height))
    
    def export_video(self):
        """
        Exports the video if recording
        """
        if self.recording:
            self.recording = False
            self.video.export()
        else:
            print('Not recording!')