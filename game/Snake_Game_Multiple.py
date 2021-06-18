"""
Snake Eater
Made with PyGame
"""

import pygame, sys, random


class Game(object):
    def __init__(self, amount_of_snakes):
        pygame.init()
        self.amount_of_snakes = amount_of_snakes
        self.reset()

    # Game Over
    def game_over(self, snake):
        self.rewards[snake] -= 10
        self.game_overs[snake] = True

    def next_generation(self):
        self.game_window.fill(self.black)
        pygame.display.flip()

    # Score
    # def show_score(self, choice, color, font, size):
    #     score_font = pygame.font.SysFont(font, size)
    #     score_surface = score_font.render('Score : ' + str(self.scores), True, color)
    #     score_rect = score_surface.get_rect()
    #     if choice == 1:
    #         score_rect.midtop = (self.frame_size_x/10, 15)
    #     else:
    #         score_rect.midtop = (self.frame_size_x/2, self.frame_size_y/1.25)
    #     self.game_window.blit(score_surface, score_rect)
    #     # pygame.display.flip()

    def random_color(self):
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        return pygame.Color(color)

    def reset(self):
        # Difficulty settings
        # Easy      ->  10
        # Medium    ->  25
        # Hard      ->  40
        # Harder    ->  60
        # Impossible->  120
        self.difficulty = 60
        self.gameLength = 0

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
        # else:
            # print('[+] Game successfully initialised')

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

        self.snake_positions = []
        self.snake_bodies = []
        self.directions = []
        self.change_tos = []
        self.food_positions = []
        self.food_spawns = []
        self.scores = []
        self.game_overs = []
        self.rewards = []
        self.colors = []

        # Game variable
        for i in range(0, self.amount_of_snakes):
            self.snake_positions.append([100, 50])
            self.snake_bodies.append([[100, 50], [100 - 10, 50], [100 - (2 * 10), 50]])
            self.directions.append('RIGHT')
            self.change_tos.append('RIGHT')
            self.food_positions.append([random.randrange(1, (self.frame_size_x // 10)) * 10, random.randrange(1, (self.frame_size_y // 10)) * 10])
            self.food_spawns.append(True)
            self.scores.append(0)
            self.game_overs.append(False)
            self.rewards.append(0)
            self.colors.append(self.random_color())

    def play_step(self, action, snake):
        # Main logic
        self.rewards[snake] = 0

        directions = ["UP", "DOWN", "LEFT", "RIGHT"]

        for i in range(0, 4):
            if action[i] == 1:
                self.move(directions[i], snake)
                break

        self.doStuff(snake)

        # self.show_score(1, self.white, 'consolas', 20)
        # Refresh game screen
        pygame.display.update()

        self.gameLength += 0.1

        return self.rewards[snake], self.game_overs[snake], self.scores[snake]

    def doStuff(self, snake):
        # Snake body growing mechanism
        self.snake_bodies[snake].insert(0, list(self.snake_positions[snake]))
        if self.snake_positions[snake][0] == self.food_positions[snake][0] and self.snake_positions[snake][1] == self.food_positions[snake][1]:
            self.scores[snake] += 5
            self.food_spawns[snake] = False

            self.rewards[snake] += 10
        else:
            self.snake_bodies[snake].pop()

        # Spawning food on the screen
        if not self.food_spawns[snake]:
            self.food_positions[snake] = [random.randrange(1, (self.frame_size_x // 10)) * 10,
                             random.randrange(1, (self.frame_size_y // 10)) * 10]
        self.food_spawns[snake] = True

        # GFX
        self.game_window.fill(self.black)
        for i in range(0, self.amount_of_snakes):
            if not self.game_overs[i]:
                for pos in self.snake_bodies[i]:
                    # Snake body
                    # .draw.rect(play_surface, color, xy-coordinate)
                    # xy-coordinate -> .Rect(x, y, size_x, size_y)
                    pygame.draw.rect(self.game_window, self.colors[i], pygame.Rect(pos[0], pos[1], 10, 10))
                    # Snake food
                pygame.draw.rect(self.game_window, self.colors[i],
                                 pygame.Rect(self.food_positions[i][0], self.food_positions[i][1], 10, 10))


        # Game Over conditions
        # Getting out of bounds
        if self.snake_positions[snake][0] < 0 or self.snake_positions[snake][0] > self.frame_size_x - 10:
            self.game_over(snake)
        if self.snake_positions[snake][1] < 0 or self.snake_positions[snake][1] > self.frame_size_y - 10:
            self.game_over(snake)
        # Touching the snake body
        for block in self.snake_bodies[snake][1:]:
            if self.snake_positions[snake][0] == block[0] and self.snake_positions[snake][1] == block[1]:
                self.game_over(snake)

    def isDanger(self, posX, posY, snake):
        # Game Over conditions
        # Getting out of bounds
        if posX < 0 or posX > self.frame_size_x - 10:
            return True
        if posY < 0 or posY > self.frame_size_y - 10:
            return True
        # Touching the snake body
        for block in self.snake_bodies[snake][1:]:
            if posX == block[0] and posY == block[1]:
                return True

    def move(self, direction, snake):
        self.change_tos[snake] = direction

        # Making sure the snake cannot move in the opposite direction instantaneously
        if self.change_tos[snake] == 'UP' and self.directions[snake] != 'DOWN':
            self.directions[snake] = 'UP'
        if self.change_tos[snake] == 'DOWN' and self.directions[snake] != 'UP':
            self.directions[snake] = 'DOWN'
        if self.change_tos[snake] == 'LEFT' and self.directions[snake] != 'RIGHT':
            self.directions[snake] = 'LEFT'
        if self.change_tos[snake] == 'RIGHT' and self.directions[snake] != 'LEFT':
            self.directions[snake] = 'RIGHT'

        # Moving the snake
        if self.directions[snake] == 'UP':
            self.snake_positions[snake][1] -= 10
        if self.directions[snake] == 'DOWN':
            self.snake_positions[snake][1] += 10
        if self.directions[snake] == 'LEFT':
            self.snake_positions[snake][0] -= 10
        if self.directions[snake] == 'RIGHT':
            self.snake_positions[snake][0] += 10

    def get_state(self, snake):
        posX = self.snake_positions[snake][0]
        posY = self.snake_positions[snake][1]

        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Calculate Danger
        # Convert direction
        if self.directions[snake] == "UP":
            state[3] = 1

            # Straight
            if self.isDanger(posX, posY - 10, snake):
                state[0] = 1
            # Left
            if self.isDanger(posX - 10, posY, snake):
                state[1] = 1
            # Right
            if self.isDanger(posX + 10, posY, snake):
                state[2] = 1

        elif self.directions[snake] == "DOWN":
            state[4] = 1

            # Straight
            if self.isDanger(posX, posY + 10, snake):
                state[0] = 1
            # Left
            if self.isDanger(posX + 10, posY, snake):
                state[1] = 1
            # Right
            if self.isDanger(posX - 10, posY, snake):
                state[2] = 1

        elif self.directions[snake] == "LEFT":
            state[5] = 1

            # Straight
            if self.isDanger(posX - 10, posY, snake):
                state[0] = 1
            # Left
            if self.isDanger(posX, posY + 10, snake):
                state[1] = 1
            # Right
            if self.isDanger(posX, posY - 10, snake):
                state[2] = 1

        elif self.directions[snake] == "RIGHT":
            state[6] = 1

            # Straight
            if self.isDanger(posX + 10, posY, snake):
                state[0] = 1
            # Left
            if self.isDanger(posX, posY - 10, snake):
                state[1] = 1
            # Right
            if self.isDanger(posX, posY + 10, snake):
                state[2] = 1

        # Calculate food position relative to snake position
        if self.food_positions[snake][0] < self.snake_positions[snake][0]:
            state[7] = 1

        if self.food_positions[snake][0] > self.snake_positions[snake][0]:
            state[8] = 1

        if self.food_positions[snake][1] < self.snake_positions[snake][1]:
            state[9] = 1

        if self.food_positions[snake][1] > self.snake_positions[snake][1]:
            state[10] = 1

        return state

# game = Game()

# while True:
#     action = [0, 0, 0, 0]
#     action[random.randint(0, 3)] = 1
#     game.get_state()
#     game.play_step(action)
