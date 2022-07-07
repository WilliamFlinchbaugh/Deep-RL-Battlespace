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
from stable_baselines3.common import env_checker
os.environ['KMP_DUPLICATE_LIB_OK']='True'

WHITE = (255, 255, 255)
RED = (138, 24, 26)
BLUE = (0, 93, 135)
BLACK = (0, 0, 0)
DISP_WIDTH = 1000
DISP_HEIGHT = 1000

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
    def __init__(self, team):
        self.team = team
        self.color = RED if self.team == 'red' else BLUE
        self.image = pygame.image.load(f"assets/{team}_plane.png")
        self.w, self.h = self.image.get_size()
        self.direction = 0
        self.rect = self.image.get_rect()
        self.hp = 10
        self.reset()

    def reset(self):
        self.hp = 6
        if self.team == 'red':
            x = (DISP_WIDTH - self.w)/4 * random.random() + self.w/2
            y = (DISP_HEIGHT - self.h)/2 * random.random() + self.h/2
            self.rect.center = (x, y)
            self.direction = 180 * random.random() + 270
            if self.direction >= 360: self.direction -= 360
        else:
            x = (DISP_WIDTH - self.w)/4 * random.random() + (DISP_WIDTH - self.w/2)/4*3 + self.w/2
            y = (DISP_HEIGHT - self.h)/2 * random.random() + self.h/2
            self.rect.center = (x, y)
            self.direction = 90 * random.random() + 180
        
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
        return self.hp

    def draw(self, surface):
        image, rect = blitRotate(self.image, self.rect.center, (self.w/2, self.h/2), self.direction)
        surface.blit(image, rect)
        rect = pygame.Rect(0, 0, self.hp * 5, 10)
        border_rect = pygame.Rect(0, 0, self.hp * 5 + 2, 12)
        rect.center = (self.rect.centerx, self.rect.centery - 20)
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
    def __init__(self, team):
        self.team = team
        self.color = RED if self.team == 'red' else BLUE
        self.image = pygame.image.load(f"assets/{team}_base.png")
        self.w, self.h = self.image.get_size()
        self.rect = self.image.get_rect()
        self.reset()
        self.hp = 3
        
    def reset(self):
        self.hp = 3
        if self.team == 'red':
            x = (DISP_WIDTH - self.w)/4 * random.random() + self.w/2
            y = (DISP_HEIGHT - self.h) * random.random() + self.h/2
            self.rect.center = (x, y)
        else:
            x = (DISP_WIDTH - self.w)/4 * random.random() + (DISP_WIDTH - self.w/2)/4*3 + self.w/2
            y = (DISP_HEIGHT - self.h) * random.random() + self.h/2
            self.rect.center = (x, y)

    def hit(self):
        self.hp -= 1
        return self.hp

    def draw(self, surface):
        surface.blit(self.image, self.rect)
        rect = pygame.Rect(0, 0, self.hp * 10, 10)
        border_rect = pygame.Rect(0, 0, self.hp * 10 + 2, 12)
        rect.center = (self.rect.centerx, self.rect.centery - 40)
        border_rect.center = rect.center
        pygame.draw.rect(surface, BLACK, border_rect, border_radius = 3)
        pygame.draw.rect(surface, self.color, rect, border_radius = 3)
            
    def get_pos(self):
        return self.rect.center

# ---------- BULLET CLASS ----------
class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y, angle, speed, fcolor, oplanes, obase):
        pygame.sprite.Sprite.__init__(self)
        self.off_screen = False
        self.image = pygame.Surface((6, 3), pygame.SRCALPHA)
        self.fcolor = fcolor
        self.color = RED if self.fcolor == 'red' else BLUE
        self.oplanes = oplanes
        self.obase = obase
        self.image.fill(self.color)
        self.rect = self.image.get_rect(center=(x, y))
        self.w, self.h = self.image.get_size()
        self.direction = angle + (random.random() * 8 - 4)
        self.pos = (x, y)
        self.speed = speed
        self.dist_travelled = 0
        self.max_dist = 600
        self.hit_plane = -1

    def update(self, screen_width, screen_height, time):
        oldpos = self.rect.center
        self.rect.center = calc_new_xy(oldpos, self.speed, time, self.direction)
        self.dist_travelled += self.speed * time
        if self.dist_travelled >= self.max_dist:
            return 'miss'
        elif self.rect.centerx > screen_width or self.rect.centerx < 0 or self.rect.centery > screen_height or self.rect.centery < 0:
            return 'miss'
        for i in range(len(self.oplanes)):
            if self.rect.colliderect(self.oplanes[i].rect):
                self.plane_hit = i
                return 'plane'
        if self.rect.colliderect(self.obase.rect):
            return 'base'
        return 'none'

    def draw(self, surface):
        image, rect = blitRotate(self.image, self.rect.center, (self.w/2, self.h/2), self.direction)
        surface.blit(image, rect)
        
    def get_pos(self):
        return self.rect.center
    
    def get_direction(self):
        return self.direction

# ----------- BATTLE ENVIRONMENT -----------
class BattleEnvironment(gym.Env):
    def __init__(self, show=False, hit_base_reward=100, hit_plane_reward=100, miss_punishment=-5, too_long_punishment=0, closer_to_base_reward=0, 
        closer_to_plane_reward=0, lose_punishment=-50, fps=60):
        super(BattleEnvironment, self).__init__()

        self.width = DISP_WIDTH
        self.height = DISP_HEIGHT
        self.max_time = 10
        self.action_space = spaces.Discrete(4) # Forward, Shoot, Turn right/Turn to plane, Turn left/Turn to base
        self.max_bullets = 20

        # ---------- Observation Space ----------
        obs_space = {
            'fplane_x': spaces.Box(-1, 1, dtype=np.float32, shape=(1,)),
            'fplane_y': spaces.Box(-1, 1, dtype=np.float32, shape=(1,)),
            'fplane_hp': spaces.Box(-1, 1, dtype=np.float32, shape=(1,)),
            'fplane_direction': spaces.Box(-1, 1, dtype=np.float32, shape=(1,)),
            'rel_angle_oplane': spaces.Box(-1, 1, dtype=np.float32, shape=(1,)),
            'rel_angle_obase': spaces.Box(-1, 1, dtype=np.float32, shape=(1,)),
            'fbase_x': spaces.Box(-1, 1, dtype=np.float32, shape=(1,)),
            'fbase_y': spaces.Box(-1, 1, dtype=np.float32, shape=(1,)),
            'fbase_hp': spaces.Box(-1, 1, dtype=np.float32, shape=(1,)),
            'oplane_x': spaces.Box(-1, 1, dtype=np.float32, shape=(1,)),
            'oplane_y': spaces.Box(-1, 1, dtype=np.float32, shape=(1,)),
            'oplane_hp': spaces.Box(-1, 1, dtype=np.float32, shape=(1,)),
            'oplane_direction': spaces.Box(-1, 1, dtype=np.float32, shape=(1,)),
            'rel_angle_fplane': spaces.Box(-1, 1, dtype=np.float32, shape=(1,)),
            'rel_angle_fbase': spaces.Box(-1, 1, dtype=np.float32, shape=(1,)),
            'obase_x': spaces.Box(-1, 1, dtype=np.float32, shape=(1,)),
            'obase_y': spaces.Box(-1, 1, dtype=np.float32, shape=(1,)),
            'obase_hp': spaces.Box(-1, 1, dtype=np.float32, shape=(1,)),
            'time': spaces.Box(-1, 1, dtype=np.float32, shape=(1,))
        }

        for i in range(self.max_bullets):
            obs_space[f'bullet_{i}_shot'] = spaces.Box(-1, 1, dtype=np.int16, shape=(1,))
            obs_space[f'bullet_{i}_team'] = spaces.Box(-1, 1, dtype=np.int16, shape=(1,))
            obs_space[f'bullet_{i}_x'] = spaces.Box(-1, 1, dtype=np.float32, shape=(1,))
            obs_space[f'bullet_{i}_y'] = spaces.Box(-1, 1, dtype=np.float32, shape=(1,))
            obs_space[f'bullet_{i}_direction'] = spaces.Box(-1, 1, dtype=np.float32, shape=(1,))
            obs_space[f'bullet_{i}_dist_to_plane'] = spaces.Box(-1, 1, dtype=np.float32, shape=(1,))
            obs_space[f'bullet_{i}_fplane_angle'] = spaces.Box(-1, 1, dtype=np.float32, shape=(1,))
            obs_space[f'bullet_{i}_fbase_angle'] = spaces.Box(-1, 1, dtype=np.float32, shape=(1,))
            

        mins = np.array([x.low[0] for x in obs_space.values()])
        maxs = np.array([x.high[0] for x in obs_space.values()])

        self.observation_space = spaces.Box(mins, maxs, dtype=np.float32)
        
        # ---------- Initialize values ----------
        self.team = {}
        self.team['red'] = {}
        self.team['blue'] = {}
        self.team['red']['base'] = Base('red')
        self.team['blue']['base'] = Base('blue')
        self.team['red']['planes'] = []
        self.team['red']['planes'].append(Plane('red'))
        self.team['blue']['planes'] = []
        self.team['blue']['planes'].append(Plane('blue'))
        self.team['red']['wins'] = 0
        self.team['blue']['wins'] = 0
        self.ties = 0
        self.bullets = []
        self.speed = 200 # mph
        self.bullet_speed = 400 # mph
        self.total_time = 0 # in hours
        self.time_step = 0.1 # hours per time step
        self.show = show # show the pygame animation
        self.step_turn = 20 # degrees to turn per step
        self.hit_base_reward = hit_base_reward
        self.hit_plane_reward = hit_plane_reward
        self.miss_punishment = miss_punishment
        self.too_long_punishment = too_long_punishment
        self.closer_to_base_reward = closer_to_base_reward
        self.closer_to_plane_reward = closer_to_plane_reward
        self.lose_punishment = lose_punishment
        self.fps = fps
    
    def _get_observation(self):
        fplane = self.team['red']['planes'][0]
        fbase = self.team['red']['base']
        oplane = self.team['blue']['planes'][0]
        obase = self.team['red']['base']

        fplane_pos = fplane.get_pos()
        fplane_direction = fplane.get_direction()
        fbase_pos = fbase.get_pos()
        oplane_pos = oplane.get_pos()
        oplane_direction = oplane.get_direction()
        obase_pos = obase.get_pos()

        rel_angle_oplane = rel_angle(fplane_pos, fplane_direction, oplane_pos)
        rel_angle_obase = rel_angle(fplane_pos, fplane_direction, obase_pos)
        rel_angle_fplane = rel_angle(oplane_pos, oplane_direction, fplane_pos)
        rel_angle_fbase = rel_angle(oplane_pos, oplane_direction, fbase_pos)

        dct = {
            'fplane_x': (fplane_pos[0] / self.width) * 2 - 1,
            'fplane_y': (fplane_pos[1] / self.height) * 2 - 1,
            'fplane_hp': (fplane.hp / 10) * 2 - 1,
            'fplane_direction': (fplane_direction / 360) * 2 - 1,
            'rel_angle_oplane': rel_angle_oplane / 360,
            'rel_angle_obase': rel_angle_obase / 360,
            'fbase_x': (fbase_pos[0] / self.width) * 2 - 1,
            'fbase_y': (fbase_pos[1] / self.height) * 2 - 1,
            'fbase_hp': (fbase.hp / 10) * 2 - 1,
            'oplane_x': (oplane_pos[0] / self.width) * 2 - 1,
            'oplane_y': (oplane_pos[1] / self.width) * 2 - 1,
            'oplane_hp': (oplane.hp / 10) * 2 - 1,
            'oplane_direction': (oplane_direction / 360) * 2 - 1,
            'rel_angle_fplane': rel_angle_fplane / 360,
            'rel_angle_fbase': rel_angle_fbase / 360,
            'obase_x': (obase_pos[0] / self.width) * 2 - 1,
            'obase_y': (obase_pos[1] / self.width) * 2 - 1,
            'obase_hp': (obase.hp / 10) * 2 - 1,
            'time': self.total_time / self.max_time
        }
        
        for i in range(self.max_bullets):
            if i < len(self.bullets):
                bullet = self.bullets[i]
                dct[f'bullet_{i}_shot'] = 1
                dct[f'bullet_{i}_team'] = 1 if bullet.fcolor == 'red' else 0
                dct[f'bullet_{i}_x'] = (bullet.get_pos()[0] / self.width) * 2 - 1
                dct[f'bullet_{i}_y'] = (bullet.get_pos()[1] / self.height) * 2 - 1
                dct[f'bullet_{i}_direction'] = (bullet.get_direction() / 360) * 2 - 1
                dct[f'bullet_{i}_dist_to_plane'] = (dist(bullet.get_pos(), fplane.get_pos()) / math.sqrt(math.pow(self.width, 2) + math.pow(self.height, 2))) * 2 - 1
                dct[f'bullet_{i}_fplane_angle'] = rel_angle(bullet.get_pos(), bullet.get_direction(), fplane_pos) / 360
                dct[f'bullet_{i}_fbase_angle'] = rel_angle(bullet.get_pos(), bullet.get_direction(), fbase_pos) / 360
            else:
                dct[f'bullet_{i}_shot'] = -1
                dct[f'bullet_{i}_team'] = -1
                dct[f'bullet_{i}_x'] = -1
                dct[f'bullet_{i}_y'] = -1
                dct[f'bullet_{i}_direction'] = -1
                dct[f'bullet_{i}_dist_to_plane'] = -1
                dct[f'bullet_{i}_fplane_angle'] = -1
                dct[f'bullet_{i}_fbase_angle'] = -1

        return np.array([x for x in dct.values()], dtype=np.float32)

    def reset(self): # return observation
        self.done = False
        self.winner = 'none'

        self.team['red']['base'].reset()
        self.team['blue']['base'].reset()

        for plane in self.team['red']['planes']:
            plane.reset()
        for plane in self.team['blue']['planes']:
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
        return self._get_observation()

    def step(self, action): # return observation, reward, done, info
        reward = 0

        # Check if over time, if so, end game in tie
        self.total_time += self.time_step
        if self.total_time >= self.max_time:
            self.done = True
            self.ties += 1
            if self.show:
                self.render()
                print("Draw")
            return self._get_observation(), 0, self.done, {}

        # Red turn
        reward += self._process_action(action, 'red', 'blue')
        
        # Blue turn
        self._process_action(random.randint(0, 3), 'blue', 'red')        

        # Check if bullets hit and move them
        for bullet in self.bullets[:]:
            outcome = bullet.update(self.width, self.height, self.time_step)
            if outcome == 'miss':
                reward = reward + self.miss_punishment if bullet.fcolor == 'red' else 0
                self.bullets.remove(bullet)

            elif outcome == 'base': # bullet hits base
                newhp = bullet.obase.hit()
                reward = reward + self.hit_base_reward if bullet.fcolor == 'red' else 0
                if newhp <= 0: # won the game
                    self.winner = bullet.fcolor
                    self.team[bullet.fcolor]['wins'] += 1
                    if not self.winner == 'red':
                        reward += self.lose_punishment
                    self.done = True
                    if self.show:
                        self.render()
                        print(f"{self.winner} wins")
                        return self._get_observation(), reward, self.done, {}
                else:
                    self.bullets.remove(bullet)
            
            elif outcome == 'plane': # bullet hits plane
                i = bullet.hit_plane
                newhp = bullet.oplanes[i].hit()
                reward = reward + self.hit_plane_reward if bullet.fcolor == 'red' else 0
                if newhp <= 0: # won the game
                    self.winner = bullet.fcolor
                    self.team[bullet.fcolor]['wins'] += 1
                    if not self.winner == 'red':
                        reward += self.lose_punishment
                    self.done = True
                    if self.show:
                        self.render()
                        print(f"{self.winner} wins")
                        return self._get_observation(), reward, self.done, {}
                else:
                    self.bullets.remove(bullet)
            
        # Check if past half of max time and give punishment
        if (self.total_time > self.max_time//2):
            reward += self.too_long_punishment

        # Continue game
        if self.show:
            self.render()
        return self._get_observation(), reward, self.done, {}
    
    def _process_action(self, action, fteam, oteam): # friendly and opponent teams
        reward = 0

        fcolor = fteam
        ocolor = oteam

        fplane = self.team[fcolor]['planes'][0]
        obase = self.team[ocolor]['base']
        oplane = self.team[ocolor]['planes'][0]

        fplane_pos = fplane.get_pos()
        fplane_direction = fplane.get_direction()
        obase_pos = obase.get_pos()
        oplane_pos = oplane.get_pos()

        dist_oplane = dist(oplane_pos, fplane_pos)
        dist_obase = dist(obase_pos, fplane_pos)

        rel_angle_oplane = rel_angle(fplane_pos, fplane_direction, oplane_pos)
        rel_angle_obase = rel_angle(fplane_pos, fplane_direction, obase_pos)

        # --------------- FORWARD ---------------
        if action == 0: 
            fplane.forward(self.speed, self.time_step)

         # --------------- SHOOT ---------------
        elif action == 1:
            if len(self.bullets) < self.max_bullets:
                self.bullets.append(Bullet(fplane_pos[0], fplane_pos[1], fplane_direction, self.bullet_speed, fcolor, self.team[ocolor]['planes'], self.team[ocolor]['base']))
            fplane.forward(self.speed, self.time_step)
        
        # --------------- TURN TO ENEMY PLANE ---------------
        elif action == 2:
            if math.fabs(rel_angle_oplane) < self.step_turn:
                fplane.rotate(-rel_angle_oplane)
            elif rel_angle_oplane < 0:
                fplane.rotate(self.step_turn)
            else:
                fplane.rotate(-self.step_turn)
            fplane.forward(self.speed, self.time_step)

        # ---------------- TURN TO ENEMY BASE ----------------
        elif action == 3:
            if math.fabs(rel_angle_obase) < self.step_turn:
                fplane.rotate(-rel_angle_obase)
            elif rel_angle_obase < 0:
                fplane.rotate(self.step_turn)
            else:
                fplane.rotate(-self.step_turn)
            fplane.forward(self.speed, self.time_step)

        # # ---------- TURN RIGHT ----------
        # elif action == 2:
        #     fplane.rotate(self.step_turn)
        #     fplane.forward(self.speed, self.time_step)

        # # ---------- TURN LEFT ----------
        # elif action == 3:
        #     fplane.rotate(-self.step_turn)
        #     fplane.forward(self.speed, self.time_step)

        # ---------------- GIVE REWARDS IF CLOSER (DIST OR ANGLE) ----------------
        new_fplane_pos = fplane.get_pos()
        new_oplane_pos = oplane.get_pos()
        new_obase_pos = obase.get_pos()

        new_dist_oplane = dist(new_oplane_pos, new_fplane_pos)
        new_dist_obase = dist(new_obase_pos, new_fplane_pos)

        if new_dist_oplane < dist_oplane and new_dist_oplane < 300: # If got closer to enemy plane and within 300 miles
            reward += self.closer_to_plane_reward

        if new_dist_obase < dist_obase and new_dist_obase < 300: # If got closer to enemy base
            reward += self.closer_to_base_reward

        return reward
    
    def winner_screen(self):
        if self.show:
            font = pygame.font.Font('freesansbold.ttf', 32)
            if self.winner != 'none':
                text = font.render(f"THE WINNER IS {self.winner.upper()}", True, BLACK)
                textRect = text.get_rect()
                textRect.center = (DISP_WIDTH//2, DISP_HEIGHT//2)
            else:
                text = font.render(f"THE GAME IS A TIE", True, BLACK)
                textRect = text.get_rect()
                textRect.center = (DISP_WIDTH//2, DISP_HEIGHT//2)
            self.display.blit(text, textRect)

    def wins(self):
        return "Wins by red: {}\nWins by blue: {}\nTied games: {}".format(self.team['red']['wins'], self.team['blue']['wins'], self.ties)

    def render(self, mode="human"):
        if self.show: # Just to ensure it won't render if self.show == False
            for event in pygame.event.get():
                # Check for KEYDOWN event
                if event.type == KEYDOWN:
                    # If the Esc key is pressed, then exit the main loop
                    if event.key == K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                # Check for QUIT event. If QUIT, then set running to false.
                elif event.type == QUIT:
                    pygame.quit()
                    self.done = True
                    sys.exit()
                    
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
            for plane in self.team['red']['planes']:
                plane.update()
                plane.draw(self.display)
            for plane in self.team['blue']['planes']:
                plane.update()
                plane.draw(self.display)

            # Winner Screen
            if self.done:
                font = pygame.font.Font('freesansbold.ttf', 32)
                if self.winner != 'none':
                    text = font.render(f"THE WINNER IS {self.winner.upper()}", True, BLACK)
                    textRect = text.get_rect()
                    textRect.center = (DISP_WIDTH//2, DISP_HEIGHT//2)
                else:
                    text = font.render(f"THE GAME IS A TIE", True, BLACK)
                    textRect = text.get_rect()
                    textRect.center = (DISP_WIDTH//2, DISP_HEIGHT//2)
                self.display.blit(text, textRect)
                pygame.display.update()
                pygame.time.wait(1000)
                pygame.quit()
                return
        
            pygame.display.update()
            self.clock.tick(self.fps)

env_checker.check_env(BattleEnvironment())