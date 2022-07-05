import random
import pygame
import numpy as np
import numpy.linalg as LA
import math
from collections import defaultdict
import gym
from gym import spaces
import os
import sys

from pygame.locals import *

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
DISP_WIDTH = 1000
DISP_HEIGHT = 1000
FPS = pygame.time.Clock()

os.environ['KMP_DUPLICATE_LIB_OK']='True'

DEFAULT_CONFIG_DICT = {
    'time_step': 0.1, # hours per time step
    'plane_speed': 500, # mph
    'bullet_speed': 700, # mph
    'max_time': 10, # hours the epoch can last
    'show_viz': False, # show the pygame animation
    'step_turn': 20, # degrees to turn per step
    'hit_base_reward': 1000, # reward for shooting enemy base
    'hit_plane_reward': 300, # reward for shooting enemy plane
    'miss_punishment': -10, # punishment for missing a shot
    'closer_to_base_reward': 1, # reward for getting closer to enemy base
    'closer_to_plane_reward': 1, # reward for getting closer to enemy plane
    'turn_to_base_reward': 1, # reward for turning towards the enemy base
    'turn_to_plane_reward': 1 # reward for turning towards the enemy plane
}

def get_angle(p1, p0):
    return math.degrees(math.atan2(p1[1]-p0[1],p1[0]-p0[0]))

def dist(p1,p0):
    return math.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)

def blitRotate(image, pos, originPos, angle):

    # offset from pivot to center
    image_rect = image.get_rect(topleft = (pos[0] - originPos[0], pos[1]-originPos[1]))
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

class Plane:
    def __init__(self, team): 
        self.team = team
        self.image = pygame.image.load(f"Images/{team}_plane.png")
        self.w, self.h = self.image.get_size()
        self.heading = 0
        self.rect = self.image.get_rect()
        self.reset()

    def reset(self):
        if self.team == 'red':
            x = (DISP_WIDTH - self.w/2)/2 * random.random()
            y = (DISP_HEIGHT - self.h/2) * random.random()
            self.rect.center = (x, y)
            self.heading = 90 * random.random() if random.random() < .5 else 90 * random.random() + 270
            
        else:
            x = (DISP_WIDTH - self.w/2)/2 * random.random() + (DISP_WIDTH - self.w/2)/2
            y = (DISP_HEIGHT - self.h/2) * random.random()
            self.rect.center = (x, y)
            self.heading = 180 * random.random() + 90
        
    def rotate(self, angle):
        self.heading += angle

    def set_heading(self, heading):
        self.heading = heading

    def forward(self, speed, time):
        oldpos = self.rect.center
        self.rect.center = calc_new_xy(oldpos, speed, time, self.heading)

    def draw(self, surface):
        image, rect = blitRotate(self.image, self.rect.center, (self.w/2, self.h/2), self.heading)
        surface.blit(image, rect)

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
        return self.rect.center

class Base:
    
    def __init__(self, team):
        self.team = team
        self.image = pygame.image.load(f"Images/{team}_base.png")
        self.w, self.h = self.image.get_size()
        self.rect = self.image.get_rect()
        self.reset()
        
    def reset(self):
        if self.team == 'red':
            x = (DISP_WIDTH - self.w/2)/2 * random.random()
            y = (DISP_HEIGHT - self.h/2) * random.random()
            self.rect.center = (x, y)
        else:
            x = (DISP_WIDTH - self.w/2)/2 * random.random() + (DISP_WIDTH - self.w/2)/2
            y = (DISP_HEIGHT - self.h/2) * random.random()
            self.rect.center = (x, y)

    def draw(self, surface):
        surface.blit(self.image, self.rect)
            
    def get_pos(self):
        return self.rect.center

class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y, angle, speed, fteam, oteam):
        pygame.sprite.Sprite.__init__(self)
        self.off_screen = False
        self.image = pygame.Surface((8, 4), pygame.SRCALPHA)
        self.fteam = fteam
        self.color = RED if self.fteam == 'red' else BLUE
        self.oteam = oteam
        self.image.fill(self.color)
        self.rect = self.image.get_rect(center=(x, y))
        self.w, self.h = self.image.get_size()
        self.heading = angle
        self.pos = (x, y)
        self.speed = speed

    def update(self, screen_width, screen_height, time):
        oldpos = self.rect.center
        self.rect.center = calc_new_xy(oldpos, self.speed, time, self.heading)
        if self.rect.centerx > screen_width or self.rect.centerx < 0 or self.rect.centery > screen_height or self.rect.centery < 0:
            return 'miss'
        for plane in self.oteam['planes']:
            if self.rect.colliderect(plane.rect):
                return 'plane'
        if self.rect.colliderect(self.oteam['base'].rect):
            return 'base'
        return 'none'

    def draw(self, surface):
        image, rect = blitRotate(self.image, self.rect.center, (self.w/2, self.h/2), self.heading)
        surface.blit(image, rect)

class BattleEnvironment(gym.Env):
    def __init__(self, config: dict=DEFAULT_CONFIG_DICT):
        super(BattleEnvironment, self).__init__()
        self.width = DISP_WIDTH
        self.height = DISP_HEIGHT
        high = np.array( # Observation space: fplane_pos_x, fplane_pos_y, fplane_angle, dist_obase, dist_oplane, rel_angle_obase, rel_angle_oplane
            [
                self.width,
                self.height,
                720,
                math.sqrt(math.pow(self.width, 2) + math.pow(self.height, 2)),
                math.sqrt(math.pow(self.width, 2) + math.pow(self.height, 2)),
                720,
                720,
            ],
            dtype=np.float32,
        )
        # For the Agent, actions are turn left, turn right, turn to enemy, turn to target, go forward, or shoot
        # For the Random choice Agent, the actions are to enemy, to target, shoot enemy, or shoot target
        self.action_space = spaces.Discrete(6)
        self.random_action_space = [0, 1, 4, 5]
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
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
        self.time_step = config['time_step']
        self.speed = config['plane_speed']
        self.bullet_speed = config['bullet_speed']
        self.total_time = 0 # in hours
        self.max_time = config['max_time']
        self.show = config['show_viz']
        self.step_turn = config['step_turn']
        self.hit_base_reward = config['hit_base_reward']
        self.hit_plane_reward = config['hit_plane_reward']
        self.miss_punishment = config['miss_punishment']
        self.closer_to_base_reward = config['closer_to_base_reward']
        self.closer_to_plane_reward = config['closer_to_plane_reward']
        self.turn_to_base_reward = config['turn_to_base_reward']
        self.turn_to_plane_reward = config['turn_to_plane_reward']

    def _get_observation(self):
        # This code will all have to be changed when adding multiple planes
        fplane = self.team['red']['planes'][0]
        oplane = self.team['blue']['planes'][0]
        obase = self.team['red']['base']

        fplane_pos = fplane.get_pos()
        fplane_angle = fplane.heading
        oplane_pos = oplane.get_pos()
        obase_pos = obase.get_pos()

        dist_oplane = dist(oplane_pos, fplane_pos)
        dist_obase = dist(obase_pos, fplane_pos)

        angle_to_oplane = get_angle(oplane_pos, fplane_pos)
        angle_to_obase = get_angle(obase_pos, fplane_pos)
        rel_angle_oplane = (angle_to_oplane - fplane_angle) % 360
        rel_angle_obase = (angle_to_obase - fplane_angle) % 360
        
        self.observation = (fplane_pos[0], fplane_pos[1], fplane_angle, dist_obase, dist_oplane, rel_angle_obase, rel_angle_oplane)
        return np.array(self.observation, dtype=np.float32)

        
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
            self.display = pygame.display.set_mode((DISP_WIDTH, DISP_HEIGHT))
            pygame.display.set_caption("Battlespace Simulator")

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
        self.friendly = 'red'
        self.opponent = 'blue'
        reward += self._process_action(action, self.team[self.friendly], self.team[self.opponent])
        
        # Blue turn
        self.friendly = 'blue'
        self.opponent = 'red'
        self._process_action(self.random_action_space[random.randint(0, 3)], self.team[self.friendly], self.team[self.opponent])        
        

        # Check if bullets hit and move them
        for bullet in self.bullets:
            outcome = bullet.update(self.width, self.height, self.time_step)
            if outcome == 'miss':
                reward += self.miss_punishment
            elif outcome == 'plane' or outcome == 'base': # If a bullet hit
                self.winner = bullet.fteam
                self.team[self.winner]['wins'] += 1
                self.done = True
                reward = reward + self.hit_base_reward if outcome == 'base' else reward + self.hit_plane_reward
                if self.show:
                    self.render()
                    print(f"{self.winner} wins")
                return self._get_observation(), reward, self.done, {}
            else:
                self.bullets.pop(self.bullets.index(bullet))
    
        # Continue game
        if self.show:
            self.render()
        return self._get_observation(), reward, self.done, {}
    
    def _process_action(self, action, fteam, oteam): # friendly and opponent
        reward = 0

        fplane = fteam['planes'][0]
        oplane = oteam['planes'][0]
        obase = oteam['base']

        fplane_pos = fplane.get_pos()
        fplane_angle = fplane.heading
        oplane_pos = oplane.get_pos()
        obase_pos = obase.get_pos()

        dist_oplane = dist(oplane_pos, fplane_pos)
        dist_obase = dist(obase_pos, fplane_pos)

        angle_to_oplane = get_angle(oplane_pos, fplane_pos)
        angle_to_obase = get_angle(obase_pos, fplane_pos)
        rel_angle_oplane = (angle_to_oplane - fplane_angle) % 360
        rel_angle_obase = (angle_to_obase - fplane_angle) % 360

        # --------------- FORWARDS ---------------
        if action == 0: 
            fplane.forward(self.speed, self.time_step)

         # --------------- SHOOT ---------------
        elif action == 1:
            self.bullets.append(Bullet(fplane_pos[0], fplane_pos[1], fplane_angle, self.bullet_speed, self.friendly, oteam))
            fplane.forward(self.speed, self.time_step)
        
        # --------------- TURN RIGHT ---------------
        elif action == 2:
            fplane.rotate(-self.step_turn)
            fplane.forward(self.speed, self.time_step)

        # ---------------- TURN LEFT ----------------
        elif action == 3:
            fplane.rotate(self.step_turn)
            fplane.forward(self.speed, self.time_step)

        # ---------------- TURN TO OPLANE ----------------
        elif action == 4:
            if math.fabs(rel_angle_oplane) < self.step_turn: # within step_turn of base
                fplane.set_heading(angle_to_oplane)

            elif math.fabs(rel_angle_oplane) > 360 - self.step_turn: # within step_turn of base
                fplane.set_heading(angle_to_oplane)

            elif math.fabs(rel_angle_oplane) < 180: # turn right
                fplane.rotate(-self.step_turn)

            else: # turn left
                fplane.rotate(self.step_turn)
                
            fplane.forward(self.speed, self.time_step)

        # ---------------- TURN TO OBASE ----------------
        elif action == 5:
            if math.fabs(rel_angle_obase) < self.step_turn: # within step_turn of base
                fplane.set_heading(angle_to_obase)

            elif math.fabs(rel_angle_obase) > 360 - self.step_turn: # within step_turn of base
                fplane.set_heading(angle_to_obase)

            elif math.fabs(rel_angle_obase) < 180: # turn right
                fplane.rotate(-self.step_turn)

            else: # turn left
                fplane.rotate(self.step_turn)

            fplane.forward(self.speed, self.time_step)
        
        # ---------------- GIVE REWARDS IF CLOSER (DIST OR ANGLE) ----------------
        new_fplane_pos = fplane.get_pos()
        new_fplane_angle = fplane.heading
        new_oplane_pos = oplane.get_pos()
        new_obase_pos = obase.get_pos()

        new_dist_oplane = dist(new_oplane_pos, new_fplane_pos)
        new_dist_obase = dist(new_obase_pos, new_fplane_pos)

        new_angle_to_oplane = get_angle(new_oplane_pos, new_fplane_pos)
        new_angle_to_obase = get_angle(new_obase_pos, new_fplane_pos)
        new_rel_angle_oplane = (new_angle_to_oplane - new_fplane_angle) % 360
        new_rel_angle_obase = (new_angle_to_obase - new_fplane_angle) % 360

        if new_dist_oplane < dist_oplane: # If got closer to enemy plane
            reward += self.closer_to_plane_reward

        if new_dist_obase < dist_obase: # If got closer to enemy base
            reward += self.closer_to_base_reward

        if math.fabs(new_rel_angle_oplane) < math.fabs(rel_angle_oplane): # If aiming closer to enemy plane
            reward += self.turn_to_plane_reward

        if math.fabs(new_rel_angle_obase) < math.fabs(rel_angle_obase): # If aiming closer to enemy base
            reward += self.turn_to_base_reward

        return reward

    def draw_shot(self, hit, friendly_pos, target_pos, team):
        color = (0, 0, 0)
        color = (255, 0, 0) if team == 'red' else (0, 0, 255)
        if not hit: target_pos = (target_pos[0] + (random.random() * 2 - 1) * 100, target_pos[1] + (random.random() * 2 - 1) * 100)
        self.shot_history.append((hit, friendly_pos, target_pos, color))
    
    def winner_screen(self):
        if self.show:
            font = pygame.font.Font('freesansbold.ttf', 32)
            if self.winner != 'none':
                text = font.render(f"THE WINNER IS {self.winner.upper()}", True, (0, 0, 0))
                textRect = text.get_rect()
                textRect.center = (DISP_WIDTH//2, DISP_HEIGHT//2)
            else:
                text = font.render(f"THE GAME IS A TIE", True, (0, 0, 0))
                textRect = text.get_rect()
                textRect.center = (DISP_WIDTH//2, DISP_HEIGHT//2)
            self.display.blit(text, textRect)

    def show_wins(self):
        print("Wins by red:", self.team['red']['wins'])
        print("Wins by blue:", self.team['blue']['wins'])
        print("Tied games:", self.ties)

    def render(self):
        if self.show: # Just to ensure it won't render if self.show == False
            for event in pygame.event.get():
                # Check for KEYDOWN event
                if event.type == KEYDOWN:
                    # If the Esc key is pressed, then exit the main loop
                    if event.key == K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                        return
                # Check for QUIT event. If QUIT, then set running to false.
                elif event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                    return
                    
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
                    text = font.render(f"THE WINNER IS {self.winner.upper()}", True, (0, 0, 0))
                    textRect = text.get_rect()
                    textRect.center = (DISP_WIDTH//2, DISP_HEIGHT//2)
                else:
                    text = font.render(f"THE GAME IS A TIE", True, (0, 0, 0))
                    textRect = text.get_rect()
                    textRect.center = (DISP_WIDTH//2, DISP_HEIGHT//2)
                self.display.blit(text, textRect)
                pygame.display.update()
                pygame.time.wait(3000)
                pygame.quit()
                return
            
            pygame.display.update()
            FPS.tick(7)

