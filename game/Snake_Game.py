"""
Snake Eater
Made with PyGame
"""

import pygame, sys, time, random


class Game(object):
    def __init__(self):
        pygame.init()
        self.overal_reward = 0
        self.reset()

    # Game Over
    def game_over(self):
        self.game_window.fill(self.black)
        pygame.display.flip()
        self.reward -= 10
        self.gameOver = True

    # Score
    def show_score(self, choice, color, font, size):
        score_font = pygame.font.SysFont(font, size)
        score_surface = score_font.render('Score : ' + str(self.score), True, color)
        score_rect = score_surface.get_rect()
        if choice == 1:
            score_rect.midtop = (self.frame_size_x/10, 15)
        else:
            score_rect.midtop = (self.frame_size_x/2, self.frame_size_y/1.25)
        self.game_window.blit(score_surface, score_rect)
        # pygame.display.flip()

    def reset(self):
        # Difficulty settings
        # Easy      ->  10
        # Medium    ->  25
        # Hard      ->  40
        # Harder    ->  60
        # Impossible->  120
        self.difficulty = 10000
        self.gameOver = False
        self.gameLength = 0
        self.total_reward = 0

        # Window size
        self.frame_size_x = 480
        self.frame_size_y = 480

        # Checks for errors encountered
        self.check_errors = pygame.init()
        # pygame.init() example output -> (6, 0)
        # second number in tuple gives number of errors
        if self.check_errors[1] > 0:
            print(f'[!] Had {self.check_errors[1]} errors when initialising game, exiting...')
            sys.exit(-1)
        else:
            print('[+] Game successfully initialised')

        # Initialise game window
        pygame.display.set_caption('Snake Eater')
        self.game_window = pygame.display.set_mode((self.frame_size_x, self.frame_size_y))

        # Colors (R, G, B)
        self.black = pygame.Color(0, 0, 0)
        self.white = pygame.Color(255, 255, 255)
        self.red = pygame.Color(255, 0, 0)
        self.green = pygame.Color(0, 255, 0)
        self.blue = pygame.Color(0, 0, 255)

        # FPS (frames per second) controller
        self.fps_controller = pygame.time.Clock()

        # Game variables
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [100 - 10, 50], [100 - (2 * 10), 50]]

        self.food_pos = [random.randrange(1, (self.frame_size_x // 10)) * 10, random.randrange(1, (self.frame_size_y // 10)) * 10]
        self.food_spawn = True

        self.direction = 'RIGHT'
        self.change_to = self.direction

        self.score = 0

    def play_step(self, action):
        # Main logic
        self.reward = 0

        directions = ["UP", "DOWN", "LEFT", "RIGHT"]

        for i in range(0, 4):
            if action[i] == 1:
                self.move(directions[i])
                break

        self.doStuff()

        self.show_score(1, self.white, 'consolas', 20)
        # Refresh game screen
        pygame.display.update()
        # Refresh rate
        self.fps_controller.tick(self.difficulty)

        self.total_reward += self.reward
        self.overal_reward += self.reward
        self.gameLength += 0.1

        return self.reward, self.gameOver, self.score

    def doStuff(self):
        # Snake body growing mechanism
        self.snake_body.insert(0, list(self.snake_pos))
        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            self.score += 5
            self.food_spawn = False

            self.reward += 10
        else:
            self.snake_body.pop()

        # Spawning food on the screen
        if not self.food_spawn:
            self.food_pos = [random.randrange(1, (self.frame_size_x // 10)) * 10,
                             random.randrange(1, (self.frame_size_y // 10)) * 10]
        self.food_spawn = True

        # GFX
        self.game_window.fill(self.black)
        for pos in self.snake_body:
            # Snake body
            # .draw.rect(play_surface, color, xy-coordinate)
            # xy-coordinate -> .Rect(x, y, size_x, size_y)
            pygame.draw.rect(self.game_window, self.green, pygame.Rect(pos[0], pos[1], 10, 10))

        # Snake food
        pygame.draw.rect(self.game_window, self.white, pygame.Rect(self.food_pos[0], self.food_pos[1], 10, 10))

        # Game Over conditions
        # Getting out of bounds
        if self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_size_x - 10:
            self.game_over()
        if self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_size_y - 10:
            self.game_over()
        # Touching the snake body
        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                self.game_over()

    def isDanger(self, posX, posY):
        # Game Over conditions
        # Getting out of bounds
        if posX < 0 or posX > self.frame_size_x - 10:
            return True
        if posY < 0 or posY > self.frame_size_y - 10:
            return True
        # Touching the snake body
        for block in self.snake_body[1:]:
            if posX == block[0] and posY == block[1]:
                return True

    def move(self, direction):
        self.change_to = direction

        # Making sure the snake cannot move in the opposite direction instantaneously
        if self.change_to == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
        if self.change_to == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'
        if self.change_to == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        if self.change_to == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'

        # Moving the snake
        if self.direction == 'UP':
            self.snake_pos[1] -= 10
        if self.direction == 'DOWN':
            self.snake_pos[1] += 10
        if self.direction == 'LEFT':
            self.snake_pos[0] -= 10
        if self.direction == 'RIGHT':
            self.snake_pos[0] += 10

    def get_state(self):
        posX = self.snake_pos[0]
        posY = self.snake_pos[1]

        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Calculate Danger
        # Convert direction
        if self.direction == "UP":
            state[3] = 1

            # Straight
            if self.isDanger(posX, posY - 10):
                state[0] = 1
            # Left
            if self.isDanger(posX - 10, posY):
                state[1] = 1
            # Right
            if self.isDanger(posX + 10, posY):
                state[2] = 1

        elif self.direction == "DOWN":
            state[4] = 1

            # Straight
            if self.isDanger(posX, posY + 10):
                state[0] = 1
            # Left
            if self.isDanger(posX + 10, posY):
                state[1] = 1
            # Right
            if self.isDanger(posX - 10, posY):
                state[2] = 1

        elif self.direction == "LEFT":
            state[5] = 1

            # Straight
            if self.isDanger(posX - 10, posY):
                state[0] = 1
            # Left
            if self.isDanger(posX, posY + 10):
                state[1] = 1
            # Right
            if self.isDanger(posX, posY - 10):
                state[2] = 1

        elif self.direction == "RIGHT":
            state[6] = 1

            # Straight
            if self.isDanger(posX + 10, posY):
                state[0] = 1
            # Left
            if self.isDanger(posX, posY - 10):
                state[1] = 1
            # Right
            if self.isDanger(posX, posY + 10):
                state[2] = 1

        # Calculate food position relative to snake position
        if self.food_pos[0] < self.snake_pos[0]:
            state[7] = 1

        if self.food_pos[0] > self.snake_pos[0]:
            state[8] = 1

        if self.food_pos[1] < self.snake_pos[1]:
            state[9] = 1

        if self.food_pos[1] > self.snake_pos[1]:
            state[10] = 1

        return state

# game = Game()

# while True:
#     action = [0, 0, 0, 0]
#     action[random.randint(0, 3)] = 1
#     game.get_state()
#     game.play_step(action)
