import pygame
import random
import math

WHITE = (255, 255, 255)
RED = (138, 24, 26)
BLUE = (0, 93, 135)
BLACK = (0, 0, 0)
DISP_WIDTH = 1200
DISP_HEIGHT = 800


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
        self.xmin = int(self.w)
        self.xmax = int(DISP_WIDTH - self.w)
        self.ymin = int(self.h)
        self.ymax = int(DISP_HEIGHT - self.h)
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
            x = random.randint(self.xmin, self.xmax // 3)
            y = random.randint(self.ymin, self.ymax)
            self.rect.center = (x, y)
            self.direction = random.randint(270, 450)
            if self.direction >= 360: self.direction -= 360
        else:
            x = random.randint(self.xmax // 3 * 2, self.xmax)
            y = random.randint(self.ymin, self.ymax)
            self.rect.center = (x, y)
            self.direction = random.randint(90, 270)
        
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

        # Draw the name of the plane
        font = pygame.font.Font(pygame.font.get_default_font(), 18)
        text = font.render(self.id, True, self.color)
        text_rect = text.get_rect()
        text_rect.center = (rect.centerx, self.rect.centery + self.h)
        surface.blit(text, text_rect)

        # Draw the health bar
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
        self.xmin = int(self.w)
        self.xmax = int(DISP_WIDTH - self.w)
        self.ymin = int(self.h)
        self.ymax = int(DISP_HEIGHT - self.h)
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
            x = random.randint(self.xmin, self.xmax // 3)
            y = random.randint(self.ymin, self.ymax)
            self.rect.center = (x, y)
        else:
            x = random.randint(self.xmax // 3 * 2, self.xmax)
            y = random.randint(self.ymin, self.ymax)
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
    def __init__(self, x, y, angle, speed, fcolor, oteam, agent_id, shot_dist):
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
        self.max_dist = shot_dist

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