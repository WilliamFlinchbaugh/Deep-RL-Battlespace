import random
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
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Needed or else it sometimes causes issues on windows machines

# Constants for pygame
WHITE = (255, 255, 255)
RED = (138, 24, 26)
BLUE = (0, 93, 135)
BLACK = (0, 0, 0)
DISP_WIDTH = 1200
DISP_HEIGHT = 800

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

def blitRotate(image, pos, originPos, angle):
    """
    Takes in a pygame image, position, the position of the origin, and an angle
    Rotates the image from the center and resizes the rect to fit the image
    Returns the rotated image and rect
    """    
    # offset from pivot to center
    image_rect = image.get_rect(topleft = (pos[0] - originPos[0], pos[1] - originPos[1]))
    offset_center_to_pivot = pygame.math.Vector2(pos) - image_rect.center
    
    # roatated offset from pivot to center
    rotated_offset = offset_center_to_pivot.rotate(-angle)

    # rotated image center
    rotated_image_center = (pos[0] - rotated_offset.x, pos[1] - rotated_offset.y)

    # get a rotated image
    rotated_image = pygame.transform.rotate(image, angle)
    rotated_image_rect = rotated_image.get_rect(center = rotated_image_center)

    return rotated_image, rotated_image_rect

def calc_new_xy(old_xy, speed, time, angle):
    """
    Takes a point, speed, timestep, and angle to calculate the new position
    Returns the new point
    """    
    new_x = old_xy[0] + (speed*time*math.cos(-math.radians(angle)))
    new_y = old_xy[1] + (speed*time*math.sin(-math.radians(angle)))
    return (new_x, new_y)

# ---------- PLANE CLASS ----------
class Plane(pygame.sprite.Sprite):
    """
    Pygame sprite of a plane
    """    
    def __init__(self, team, hp, id):
        """ Initializes the values and pygame image/rect

        Args:
            team (string): Represents the color of the team that the plane is on; 'red' or 'blue'
            hp (int): # of healthpoints for the plane (# of shots that can be taken)
            id (string): The id used in env.agents and env.possible_agents
        """
        pygame.sprite.Sprite.__init__(self)
        self.id = id
        self.team = team
        self.color = RED if self.team == 'red' else BLUE
        self.image = pygame.image.load(f"assets/{team}_plane.png")
        self.w, self.h = self.image.get_size()
        self.xmin = self.w / 2
        self.xmax = DISP_WIDTH - (self.w / 2)
        self.ymin = self.h / 2
        self.ymax = DISP_HEIGHT - (self.h / 2)
        self.direction = 0
        self.rect = self.image.get_rect()
        self.max_hp = hp
        self.hp = self.max_hp
        self.alive = True
        self.reset()

    def reset(self):
        """
        Sets to a random position on the left or right side depending on team
        Resets all other values
        """
        self.hp = self.max_hp
        self.alive = True
        if self.team == 'red':
            x = self.xmax/3 * random.random()
            y = self.ymax * random.random()
            self.rect.center = (x, y)
            self.direction = 180 * random.random() + 270
            if self.direction >= 360: self.direction -= 360
        else:
            x = self.xmax/3 * random.random() + (2 * self.xmax) / 3
            y = self.ymax * random.random()
            self.rect.center = (x, y)
            self.direction = 90 * random.random() + 180
        
        # Keep player on the screen
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > DISP_WIDTH:
            self.rect.right = DISP_WIDTH
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= DISP_HEIGHT:
            self.rect.bottom = DISP_HEIGHT
        
    def rotate(self, angle):
        """Rotates the plane by adding to self.direction

        Args:
            angle (float/int): # of degrees that the plane should turn
        """
        self.direction += angle
        while self.direction > 360:
            self.direction -= 360
        while self.direction < 0:
            self.direction += 360

        # Keep player on the screen
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > DISP_WIDTH:
            self.rect.right = DISP_WIDTH
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= DISP_HEIGHT:
            self.rect.bottom = DISP_HEIGHT

    def set_direction(self, direction):
        """Sets self.direction

        Args:
            direction (float): # of degrees that represents the plane's direction
        """
        self.direction = direction

    def forward(self, speed, time):
        """Moves the plane forward based on the direction, speed, and timestep

        Args:
            speed (int): The speed that the plane moves at (in MPH)
            time (float): Timestep for the plane (in hrs)
        """
        oldpos = self.rect.center
        self.rect.center = calc_new_xy(oldpos, speed, time, self.direction)

        # Keep player on the screen
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > DISP_WIDTH:
            self.rect.right = DISP_WIDTH
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= DISP_HEIGHT:
            self.rect.bottom = DISP_HEIGHT

    def hit(self):
        """
        Process a shot on the plane by decrementing hp

        Returns:
            hp (int): The HP after taking a hit
        """
        self.hp -= 1
        if self.hp <= 0:
            self.alive = False
        return self.hp

    def draw(self, surface):
        """Draws the plane to the display surface

        Args:
            surface (pygame.Surface): The display to draw the plane to
        """
        image, rect = blitRotate(self.image, self.rect.center, (self.w/2, self.h/2), self.direction)
        surface.blit(image, rect)
        if self.hp > 0:
            rect = pygame.Rect(0, 0, self.hp * 10, 10)
            border_rect = pygame.Rect(0, 0, self.hp * 10 + 2, 12)
            rect.center = (self.rect.centerx, self.rect.centery - 35)
            border_rect.center = rect.center
            pygame.draw.rect(surface, BLACK, border_rect, border_radius = 3)
            pygame.draw.rect(surface, self.color, rect, border_radius = 3)

    def update(self):
        """
        Makes sure that the plane stays on the screen
        """
        # Keep player on the screen
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > DISP_WIDTH:
            self.rect.right = DISP_WIDTH
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= DISP_HEIGHT:
            self.rect.bottom = DISP_HEIGHT

    def get_pos(self):
        """Gives the current position of the plane as a point (x, y)

        Returns:
            tuple (x, y): current x-y position (center)
        """
        image, rect = blitRotate(self.image, self.rect.center, (self.w/2, self.h/2), self.direction)
        return (rect.centerx, rect.centery)
    
    def get_direction(self):
        """Gives the current direction that the plane is facing

        Returns:
            float: Current direction in degrees
        """
        return self.direction

# ---------- BASE CLASS ----------
class Base(pygame.sprite.Sprite):
    """
    Pygame sprite of a base
    """
    def __init__(self, team, hp):
        """Initiates values for the base

        Args:
            team (string): Represents the team of the base, 'red' or 'blue'
            hp (int): The # of hitpoints that the base should have
        """
        pygame.sprite.Sprite.__init__(self)
        self.team = team
        self.color = RED if self.team == 'red' else BLUE
        self.image = pygame.image.load(f"assets/{team}_base.png")
        self.w, self.h = self.image.get_size()
        self.xmin = self.w / 2
        self.xmax = DISP_WIDTH - (self.w / 2)
        self.ymin = self.h / 2
        self.ymax = DISP_HEIGHT - (self.h / 2)
        self.rect = self.image.get_rect()
        self.max_hp = hp
        self.hp = self.max_hp
        self.alive = True
        self.reset()
        
    def reset(self):
        """
        Spawns base in random location on left or right of screen based on team
        Resets other values
        """
        self.alive = True
        self.hp = self.max_hp
        if self.team == 'red':
            x = self.xmax/3 * random.random()
            y = self.ymax * random.random()
            self.rect.center = (x, y)
        else:
            x = self.xmax/3 * random.random() + (2 * self.xmax) / 3
            y = self.ymax * random.random()
            self.rect.center = (x, y)

    def hit(self):
        """Decrements the base's health

        Returns:
            int: HP after taking hit
        """
        self.hp -= 1
        if self.hp <= 0:
            self.alive = False
        return self.hp

    def draw(self, surface):
        """Draws the base to the display surface

        Args:
            surface (pygame.Surface): Pygame surface to draw to
        """
        surface.blit(self.image, self.rect)
        rect = pygame.Rect(0, 0, self.hp * 10, 10)
        if self.hp > 0:
            border_rect = pygame.Rect(0, 0, self.hp * 10 + 2, 12)
            rect.center = (self.rect.centerx, self.rect.centery - 40)
            border_rect.center = rect.center
            pygame.draw.rect(surface, BLACK, border_rect, border_radius = 3)
            pygame.draw.rect(surface, self.color, rect, border_radius = 3)
            
    def get_pos(self):
        """Gets position of the base

        Returns:
            tuple (x, y): Point of the base (center)
        """
        return self.rect.center

# ---------- BULLET CLASS ----------
class Bullet(pygame.sprite.Sprite):
    """
    Pygame sprite of a bullet
    """
    def __init__(self, x, y, angle, speed, fcolor, oteam, agent_id):
        """Initiates values

        Args:
            x (int): x coordinate to spawn bullet
            y (int): y coordinate to spawn bullet
            angle (float/int): Angle that the bullet is heading
            speed (int): Speed that the bullet moves at
            fcolor (string): String representing the team that the bullet was shot from, 'red' or 'blue
            oteam (dict): dictionary of the enemy team containing ['base'] and ['planes']
        """
        pygame.sprite.Sprite.__init__(self)
        self.off_screen = False
        self.image = pygame.Surface((6, 3), pygame.SRCALPHA)
        self.fcolor = fcolor
        self.color = RED if self.fcolor == 'red' else BLUE
        self.oteam = oteam
        self.agent_id = agent_id
        self.image.fill(self.color)
        self.rect = self.image.get_rect(center=(x, y))
        self.w, self.h = self.image.get_size()
        self.direction = angle + (random.random() * 8 - 4)
        self.pos = (x, y)
        self.speed = speed
        self.dist_travelled = 0
        self.max_dist = 600

    # Checks the status of the bullet (hit or miss or neither)
    def update(self, screen_width, screen_height, time):
        """Moves the bullet and checks for collisions

        Args:
            screen_width (int): Width of the display
            screen_height (int): Height of the display
            time (float): Timestep to calculate the distance to move

        Returns:
            outcome (string or Plane or Base): 'none', 'miss', Base object, or Plane object based on what the bullet collides with
        """
        oldpos = self.rect.center
        self.rect.center = calc_new_xy(oldpos, self.speed, time, self.direction)
        self.dist_travelled += self.speed * time
        # Miss if travelled max dist
        if self.dist_travelled >= self.max_dist:
            return 'miss'
        
        # Miss if goes off screen
        elif self.rect.centerx > screen_width or self.rect.centerx < 0 or self.rect.centery > screen_height or self.rect.centery < 0:
            return 'miss'

        # Hit if collides with enemy base
        if self.rect.colliderect(self.oteam['base']):
            return self.oteam['base']

        # Hit if collides with any enemy plane
        for plane in self.oteam['planes'].values():
            if self.rect.colliderect(plane.rect):
                return plane
        return 'none'

    def draw(self, surface):
        """Draws the bullet to the display surface

        Args:
            surface (pygame.Surface): Surface to draw bullet to
        """
        image, rect = blitRotate(self.image, self.rect.center, (self.w/2, self.h/2), self.direction)
        surface.blit(image, rect)
        
    def get_pos(self):
        """Gets position of the bullet

        Returns:
            tuple (x, y): Center of bullet rect
        """
        return self.rect.center
    
    def get_direction(self):
        """Gets direction of the bullet

        Returns:
            direction (float): Direction bullet is heading in
        """
        return self.direction

# ---------- EXPLOSION CLASS ----------
class Explosion(pygame.sprite.Sprite):
    """
    Pygame sprite of an explosion
    Animation of an explosion when a plane dies
    """
    def __init__(self, center):
        """Initializes values

        Args:
            center (tuple): (x, y) point at which to spawn the explosion animation
        """
        pygame.sprite.Sprite.__init__(self)

        # Load in the animation photos
        self.explosion_anim = []
        for i in range(9):
            img = pygame.image.load(f"assets/explode{i}.png")
            img.set_colorkey(BLACK)
            img_sm = pygame.transform.scale(img, (64, 64))
            self.explosion_anim.append(img_sm)

        self.frame = 0
        self.image = self.explosion_anim[self.frame]
        self.rect = self.image.get_rect()
        self.rect.center = center

    def draw(self, surface):
        """Draws the explosion to the display surface
        Increments the frame to show the animation

        Args:
            surface (pygame.Surface): Surface to display animation to
        """
        if not self.frame == len(self.explosion_anim):
            center = self.rect.center
            self.image = self.explosion_anim[self.frame]
            self.rect = self.image.get_rect()
            self.rect.center = center
            surface.blit(self.image, self.rect)
            self.frame += 1
            return
        self.kill()

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

    def __init__(self, n_agents=1, show=False, hit_base_reward=10, hit_plane_reward=2, miss_punishment=0, die_punishment=-3, fps=20, **kwargs):
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
        EzPickle.__init__(self, n_agents, show, hit_base_reward, hit_plane_reward, miss_punishment, die_punishment, fps, **kwargs)
        self.n_agents = n_agents # n agents per team

        pygame.init()

        self.base_hp = 4 * self.n_agents
        self.plane_hp = 2

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

        self.agent_name_mapping = dict(zip(self.agents, list(range(self.n_agents))))

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
        - 1: Forward
        - 2: Shoot
        - 3: Turn Left
        - 4: Turn Right

        MADDPG uses continuous actions, so we use force_discrete_action to configure that
        The actions will be a np.array of values and the max value will be taken as the action
        """

        self.n_actions = 4
        action_space = spaces.Discrete(self.n_actions)
        self.action_spaces = {agent: action_space for agent in self.possible_agents} # Dictionary containing an action space for each agent
        
        # ---------- Initialize values ----------
        self.width = DISP_WIDTH
        self.height = DISP_HEIGHT
        self.max_time = 10 + (self.n_agents * 2)
        self.total_games = 0
        self.ties = 0
        self.bullets = []
        self.explosions = []
        self.speed = 225 # mph
        self.bullet_speed = 450 # mph
        self.total_time = 0 # in hours
        self.time_step = 0.1 # hours per time step
        self.step_turn = 20 # degrees to turn per step
        self.show = show # show the pygame animation
        self.hit_base_reward = hit_base_reward
        self.hit_plane_reward = hit_plane_reward
        self.miss_punishment = miss_punishment
        self.die_punishment = die_punishment
        self.fps = fps

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
                obs[index] = 1
                obs[index + 1] = dist(agent_pos, plane_pos) / (math.sqrt(math.pow(self.width, 2) + math.pow(self.height, 2))) * 2 - 1
                obs[index + 2] = rel_angle(agent_pos, agent_dir, plane_pos) / 360
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

        self.display = pygame.Surface((DISP_WIDTH, DISP_HEIGHT)) # Reset the pygame display
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

        # Set rewards and cumulative rewards to 0
        rewards = {agent: 0 for agent in self.possible_agents}

        # If passing no actions
        if not actions:
            self.winner = "tie"
            self.env_done = True # Set environment to done
            self.agents = [] # Kill all agents
            observations = {agent: self.observe(agent) for agent in self.possible_agents} # Get observation for each agent
            infos = {agent: {} for agent in self.possible_agents} # Empty info for each agent
            self.dones = {agent: True for agent in self.possible_agents}
            return observations, rewards, self.dones, infos 

        # Increase time and check for a tie
        self.total_time += self.time_step
        if self.total_time >= self.max_time: # If over the max time
            self.dones = {agent: True for agent in self.possible_agents} # Set all agents to done
            self.winner = 'tie'
            self.total_games += 1 
            self.ties += 1
            if self.show:
                self.render()
            observations = {agent: self.observe(agent) for agent in self.possible_agents} # Get observation for each agent
            infos = {agent: {} for agent in self.possible_agents} # Empty info for each agent
            self.agents = [] # Kill all agents
            self.env_done = True # Set environment to done
            self.dones = {agent: True for agent in self.possible_agents}
            return observations, rewards, self.dones, infos

        for agent_id in self.agents: # Carry out actions for each agent that is alive
            action = actions[agent_id] # Grab action for this agent 
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
            self.winner = 'red'
            self.team['red']['wins'] += 1
            self.total_games += 1
            self.dones = {agent: True for agent in self.possible_agents}
            self.env_done = True

        if not self.team['red']['base'].alive:
            self.winner = 'blue'
            self.team['blue']['wins'] += 1
            self.total_games += 1
            self.dones = {agent: True for agent in self.possible_agents}
            self.env_done = True

        # Render the environment
        if self.show:
            self.render()
        
        observations = {agent: self.observe(agent) for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}

        if self.env_done: # The game is over
            self.agents = [] # Kill all agents 
            self.dones = {agent: True for agent in self.possible_agents} # Set all agents to done
        
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

        # --------------- FORWARD ---------------
        if action == 0: 
            agent.forward(self.speed, self.time_step) # Move the plane forward

         # --------------- SHOOT ---------------
        elif action == 1:
            self.bullets.append(Bullet(agent_pos[0], agent_pos[1], agent_dir, self.bullet_speed, team, self.team[oteam])) # Shoot a bullet
            agent.forward(self.speed, self.time_step) # Move the plane forward
        
        # --------------- TURN LEFT ---------------
        elif action == 2:
            agent.rotate(self.step_turn) # Rotate the plane
            agent.forward(self.speed, self.time_step) # Move the plane forward

        # ---------------- TURN RIGHT ----------------
        elif action == 3:
            agent.rotate(-self.step_turn) # Rotate the plane
            agent.forward(self.speed, self.time_step) # Move the plane forward

    def winner_screen(self):
        """
        Display the winner of the game when the game is over
        """
        if self.show: # Makes sure that we are visualizing
            font = pygame.font.Font('freesansbold.ttf', 32)
            if self.winner != 'none' and self.winner != 'tie':
                text = font.render(f"THE WINNER IS {self.winner.upper()}", True, BLACK)
                textRect = text.get_rect()
                textRect.center = (DISP_WIDTH//2, DISP_HEIGHT//2)
            else:
                text = font.render(f"THE GAME IS A TIE", True, BLACK)
                textRect = text.get_rect()
                textRect.center = (DISP_WIDTH//2, DISP_HEIGHT//2)
            self.display.blit(text, textRect)

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

    def render(self, mode="human"):
        """
        Renders the pygame visuals
        """
        # Just to ensure it won't render if self.show == False
        if not self.show: return

        if not self.rendering: # We need to initialize everything if not yet rendering
            self.rendering = True
            pygame.display.init()
            pygame.font.init()
            self.display = pygame.display.set_mode((DISP_WIDTH, DISP_HEIGHT))
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
        self.display.fill(WHITE)

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
            pygame.display.update()
            if self.fps <= 60:
                pygame.time.wait(500)

        # Update the display and tick the clock with the framerate
        pygame.display.update()
        self.clock.tick(self.fps)