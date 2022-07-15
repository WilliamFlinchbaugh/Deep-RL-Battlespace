import random
import pygame
from pygame.locals import *
import numpy as np
import numpy.linalg as LA
import math
from collections import defaultdict
import gym
from gym import spaces
import os
import sys
import functools
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
os.environ['KMP_DUPLICATE_LIB_OK']='True'

WHITE = (255, 255, 255)
RED = (138, 24, 26)
BLUE = (0, 93, 135)
BLACK = (0, 0, 0)
DISP_WIDTH = 1000
DISP_HEIGHT = 1000

def env(**kwargs):
    env = raw_env(**kwargs)
    # This wrapper is only for environments which print results to the terminal
    env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

# ---------- HELPER FUNCTIONS -----------
def rel_angle(p0, a0, p1):
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
    return math.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)

def blitRotate(image, pos, originPos, angle):

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
    new_x = old_xy[0] + (speed*time*math.cos(-math.radians(angle)))
    new_y = old_xy[1] + (speed*time*math.sin(-math.radians(angle)))
    return (new_x, new_y)

# ---------- PLANE CLASS ----------
class Plane:
    def __init__(self, team, hp):
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
        self.direction = direction

    def forward(self, speed, time):
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
        self.hp -= 1
        if self.hp <= 0:
            self.alive = False
        return self.hp

    def draw(self, surface):
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
        image, rect = blitRotate(self.image, self.rect.center, (self.w/2, self.h/2), self.direction)
        return (rect.centerx, rect.centery)
    
    def get_direction(self):
        return self.direction

# ---------- BASE CLASS ----------
class Base:
    def __init__(self, team, hp):
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
        self.hp -= 1
        if self.hp <= 0:
            self.alive = False
        return self.hp

    def draw(self, surface):
        surface.blit(self.image, self.rect)
        rect = pygame.Rect(0, 0, self.hp * 10, 10)
        if self.hp > 0:
            border_rect = pygame.Rect(0, 0, self.hp * 10 + 2, 12)
            rect.center = (self.rect.centerx, self.rect.centery - 40)
            border_rect.center = rect.center
            pygame.draw.rect(surface, BLACK, border_rect, border_radius = 3)
            pygame.draw.rect(surface, self.color, rect, border_radius = 3)
            
    def get_pos(self):
        return self.rect.center

# ---------- BULLET CLASS ----------
class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y, angle, speed, fcolor, oteam):
        pygame.sprite.Sprite.__init__(self)
        self.off_screen = False
        self.image = pygame.Surface((6, 3), pygame.SRCALPHA)
        self.fcolor = fcolor
        self.color = RED if self.fcolor == 'red' else BLUE
        self.oteam = oteam
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
        image, rect = blitRotate(self.image, self.rect.center, (self.w/2, self.h/2), self.direction)
        surface.blit(image, rect)
        
    def get_pos(self):
        return self.rect.center
    
    def get_direction(self):
        return self.direction

# ----------- BATTLE ENVIRONMENT -----------
class raw_env(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "battle_env_v1",
        "is_parallelizable": False,
        "render_fps": 30,
    }
    def __init__(self, n_agents=1, show=False, hit_base_reward=10, hit_plane_reward=2, miss_punishment=0, lose_punishment=-3, die_punishment=-3, fps=20):
        super(raw_env, self).__init__()
        self.n_agents = n_agents

        base_hp = 5 * self.n_agents
        plane_hp = 3
        self.team = {}
        self.team['red'] = {}
        self.team['blue'] = {}
        self.team['red']['base'] = Base('red', base_hp)
        self.team['blue']['base'] = Base('blue', base_hp)
        self.team['red']['planes'] = {}
        self.team['blue']['planes'] = {}
        self.team['red']['wins'] = 0
        self.team['blue']['wins'] = 0

        self.agents = [f"plane_{r}" for r in range(self.n_agents * 2)]
        self.team_map = {}
        for i in range(len(self.agents)):
            if i < self.n_agents:
                self.team['red']['planes'][self.agents[i]] = Plane('red', plane_hp)
                self.team_map[f'plane_{i}'] = 'red'
            else:
                self.team['blue']['planes'][self.agents[i]] = Plane('blue', plane_hp)
                self.team_map[f'plane_{i}'] = 'blue'

        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.n_agents))))
        self._agent_selector = agent_selector(self.agents)

        obs = {} # Observation per agent, consists of dist + angle of each enemy, and base

        obs['base_dist'] = spaces.Box(-1, 1, dtype=np.float32, shape=(1,))
        obs['base_angle'] = spaces.Box(-1, 1, dtype=np.float32, shape=(1,))
        for i in range(n_agents):
            obs[f'plane_{i}_alive'] = spaces.Box(-1, 1, dtype=np.int16, shape=(1,))
            obs[f'plane_{i}_dist'] = spaces.Box(-1, 1, dtype=np.float32, shape=(1,))
            obs[f'plane_{i}_angle'] = spaces.Box(-1, 1, dtype=np.float32, shape=(1,))

        mins = np.array([x.low[0] for x in obs.values()])
        maxs = np.array([x.high[0] for x in obs.values()])

        obs_space = spaces.Box(mins, maxs, dtype=np.float32)

         # Forward, Shoot, Turn right, Turn left
        self.observation_spaces = {agent: obs_space for agent in self.possible_agents}
        self.action_spaces = {agent: spaces.Discrete(4) for agent in self.possible_agents}
        
        # ---------- Initialize values ----------
        self.width = DISP_WIDTH
        self.height = DISP_HEIGHT
        self.max_time = 8 + (self.n_agents * 2)
        self.total_games = 0
        self.ties = 0
        self.bullets = []
        self.speed = 200 # mph
        self.bullet_speed = 400 # mph
        self.total_time = 0 # in hours
        self.time_step = 0.1 # hours per time step
        self.step_turn = 20 # degrees to turn per step
        self.show = show # show the pygame animation
        self.hit_base_reward = hit_base_reward
        self.hit_plane_reward = hit_plane_reward
        self.miss_punishment = miss_punishment
        self.lose_punishment = lose_punishment
        self.die_punishment = die_punishment
        self.fps = fps

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def observe(self, agent):
        agent_team = self.team_map[agent]
        agent_plane = self.team[agent_team]['planes'][agent]
        oteam = 'blue' if agent_team == 'red' else 'red'
        agent_pos = agent_plane.get_pos()
        agent_dir = agent_plane.get_direction()
        oplanes = self.team[oteam]['planes']
        obase = self.team[oteam]['base']
        obase_pos = obase.get_pos()

        dct = {}

        dct['base_dist'] = dist(agent_pos, obase_pos) / (math.sqrt(math.pow(self.width, 2) + math.pow(self.height, 2))) * 2 - 1
        dct['base_angle'] = rel_angle(agent_pos, agent_dir, obase_pos) / 360
        for key, plane in oplanes.items():
            plane_pos = plane.get_pos()
            dct[f'{key}_alive'] = 1 if plane.alive else 0
            dct[f'{key}_dist'] = dist(agent_pos, plane_pos) / (math.sqrt(math.pow(self.width, 2) + math.pow(self.height, 2))) * 2 - 1
            dct[f'{key}_angle'] = rel_angle(agent_pos, agent_dir, plane_pos) / 360

        return np.array([x for x in dct.values()], dtype=np.float32)

    def reset(self, seed=None, return_info=False, options=None):
        self.winner = 'none'

        self.team['red']['base'].reset()
        self.team['blue']['base'].reset()

        for plane in self.team['red']['planes'].values():
            plane.reset()
        for plane in self.team['blue']['planes'].values():
            plane.reset()

        self.total_time = 0
        self.bullets = []

        if self.show:
            pygame.init()
            pygame.font.init()
            self.clock = pygame.time.Clock()
            self.display = pygame.display.set_mode((DISP_WIDTH, DISP_HEIGHT))
            pygame.display.set_caption("Battlespace Simulator")
            pygame.time.wait(1000)

        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))

    def step(self, action):
        # Checks if agent is already done
        if self.dones[self.agent_selection]: 
            return self._was_done_step(action)

        self.rewards = {agent: 0 for agent in self.agents}
        agent_id = self.agent_selection

        self._cumulative_rewards[agent_id] = 0

        # Take action
        self.process_action(action, agent_id)

        # If it's the last agent, we need to move bullets and check for wins/ties
        all_agents_updated = self._agent_selector.is_last()
        if all_agents_updated:
            # Move every bullet and check for hits
            for bullet in self.bullets[:]:
                # Move bullet and gather outcome
                outcome = bullet.update(self.width, self.height, self.time_step)

                # Kill bullet if miss
                if outcome == 'miss':
                    self.rewards[agent_id] += self.miss_punishment
                    self.bullets.remove(bullet)

                # Kill bullet and provide reward if hits base
                elif isinstance(outcome, Base):
                    outcome.hit()
                    self.rewards[agent_id] += self.hit_base_reward

                    # Won the game (base is dead)
                    if not outcome.alive:
                        self.winner = bullet.fcolor
                        self.team[bullet.fcolor]['wins'] += 1

                        # Give lose_punishment to enemy team planes
                        for key, plane in self.team_map:
                            if plane == outcome.team:
                                self.rewards[key] += self.lose_punishment

                        self.total_games += 1
                        self.dones = {agent: True for agent in self.agents}

                        if self.show:
                            self.render()
                            print(f"{self.winner} wins")

                    # Didn't win, just hit the base
                    else:
                        self.bullets.remove(bullet)
                
                # Kill bullet and provide reward if hits plane
                elif isinstance(outcome, Plane):
                    outcome.hit()
                    self.rewards[agent_id] += self.hit_plane_reward
                    self.bullets.remove(bullet)

                    # Plane is dead
                    if not outcome.alive:
                        self.rewards[agent_id] += self.die_punishment
                        self.dones[agent_id] = True

            # Increase time and check for a tie
            self.total_time += self.time_step
            if self.total_time >= self.max_time:
                self.dones = {agent: True for agent in self.agents}
                self.winner = 'tie'
                self.total_games += 1
                self.ties += 1

            # Render the environment
            if self.show:
                self.render()
        
        # Select next agent
        self.agent_selection = self._agent_selector.next()

        # Adds rewards
        self._accumulate_rewards()
    
    def process_action(self, action, agent_id):
        agent = self.team['red']['planes'][agent_id] if agent_id in self.team['red']['planes'] else self.team['blue']['planes'][agent_id]
        team = 'red' if agent_id in self.team['red']['planes'] else 'blue'
        oteam = 'blue' if team == 'red' else 'red'
        agent_pos = agent.get_pos()
        agent_dir = agent.get_direction()

        # --------------- FORWARD ---------------
        if action == 0: 
            agent.forward(self.speed, self.time_step)

         # --------------- SHOOT ---------------
        elif action == 1:
            self.bullets.append(Bullet(agent_pos[0], agent_pos[1], agent_dir, self.bullet_speed, team, self.team[oteam]))
            agent.forward(self.speed, self.time_step)
        
        # --------------- TURN LEFT ---------------
        elif action == 2:
            agent.rotate(self.step_turn)
            agent.forward(self.speed, self.time_step)

        # ---------------- TURN RIGHT ----------------
        elif action == 3:
            agent.rotate(-self.step_turn)
            agent.forward(self.speed, self.time_step)

    def winner_screen(self):
        if self.show:
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
        return "Wins by red: {}\nWins by blue: {}\nTied games: {}\nWin rate: {}".format(self.team['red']['wins'], self.team['blue']['wins'], self.ties, self.team['red']['wins']/self.total_games)

    def close(self):
        pygame.quit()
        sys.exit()

    def render(self, mode="human"):
        # Just to ensure it won't render if self.show == False
        if self.show: 
            # Check if we should quit
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.close()
                elif event.type == QUIT:
                    self.close()
            
            font = pygame.font.Font('freesansbold.ttf', 12)

            # Fill background
            self.display.fill(WHITE)

            # Draw bullets
            for bullet in self.bullets:
                bullet.draw(self.display)
                    
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
                pygame.time.wait(1500)
                self.close()

            pygame.display.update()
            self.clock.tick(self.fps)