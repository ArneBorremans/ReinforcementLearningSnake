from random import Random, randint

from Snake_Game import Game

class Agent:
    def __init__(self):
        self.n_games = 0  # number of games / epochs

    def get_state(self, game):
        return game.get_state()

    def get_action(self, state):

        direction = state[3:7]
        directionString = ""
        directions = ["UP", "DOWN", "LEFT", "RIGHT"]

        if direction == [1, 0, 0, 0]:
            directionString = "UP"
        elif direction == [0, 1, 0, 0]:
            directionString = "DOWN"
        if direction == [0, 0, 1, 0]:
            directionString = "LEFT"
        elif direction == [0, 0, 0, 1]:
            directionString = "RIGHT"

        print(directionString)

        print(state)

        # STATE:
        # 3 bits: DANGER STRAIGHT, LEFT, RIGHT
        # 4 bits: GOING UP, DOWN, LEFT, RIGHT
        # 4 bits: FOOD_X < SNAKE_X, FOOD_X > SNAKE_X, FOOD_Y < SNAKE_Y, FOOD_Y > SNAKE_Y

        final_move_string = "UP"

        if state[7] == 1:
            final_move_string = "LEFT"
        elif state[8] == 1:
            final_move_string = "RIGHT"
        else:
            if state[9] == 1:
                final_move_string = "UP"
            if state[10] == 1:
                final_move_string = "DOWN"

        if final_move_string == directionString:
            directions.remove(directionString)
            final_move_string = directions[randint(0, 2)]

        final_move = [0, 0, 0, 0]

        if final_move_string == "UP":
            final_move = [1, 0, 0, 0]
        elif final_move_string == "DOWN":
            final_move = [0, 1, 0, 0]
        elif final_move_string == "LEFT":
            final_move = [0, 0, 1, 0]
        elif final_move_string == "RIGHT":
            final_move = [0, 0, 0, 1]

        return final_move

def play():
    total_score = 0
    record = 0
    average_reward = 0
    print('creating agent')
    agent = Agent()
    print('creating game')
    game = Game()
    print('initializing game')

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        if done:
            # train long memory, plot result
            gameLength = game.gameLength
            totalReward = game.total_reward
            game.reset()
            agent.n_games += 1

            if score > record:
                record = score

            difference = (game.overal_reward / agent.n_games) - average_reward
            average_reward = game.overal_reward / agent.n_games

            print("----------------------------- Game:", agent.n_games, "-----------------------------")
            print("Survived for: ", gameLength)
            print("Total reward: ", totalReward)

            print('Score:', score, 'Record:', record)
            if (difference >= 0):
                print('Average reward:', average_reward, "(+", difference, ")")
            else:
                print('Average reward:', average_reward, "(-", abs(difference), ")")

if __name__ == '__main__':
    play()
