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
        directions = ["UP", "RIGHT", "DOWN", "LEFT"]
        directionsStatic = ["UP", "RIGHT", "DOWN", "LEFT"]

        if direction == [1, 0, 0, 0]:
            directionString = "UP"
        elif direction == [0, 1, 0, 0]:
            directionString = "DOWN"
        if direction == [0, 0, 1, 0]:
            directionString = "LEFT"
        elif direction == [0, 0, 0, 1]:
            directionString = "RIGHT"

        # STATE:
        # 3 bits: DANGER STRAIGHT, LEFT, RIGHT
        # 4 bits: GOING UP, DOWN, LEFT, RIGHT
        # 4 bits: FOOD_X < SNAKE_X, FOOD_X > SNAKE_X, FOOD_Y < SNAKE_Y, FOOD_Y > SNAKE_Y

        # final_move_string = "UP"

        # remove opposite direction
        directions.remove(directionsStatic[(directionsStatic.index(directionString) + 2) % 4])

        # remove the dangers
        danger = state[0:3]
        if danger[0] == 1:
            directions.remove(directionString)
        if danger[1] == 1:
            directions.remove(directionsStatic[directionsStatic.index(directionString) - 1])
        if danger[2] == 1:
            directions.remove(directionsStatic[(directionsStatic.index(directionString) + 1) % 4])
        if danger == [1, 1, 1]:
            directions.append(directionString)

        # Decide the next move
        if (state[7] == 1) & ("LEFT" in directions):
            final_move_string = "LEFT"
        elif (state[8] == 1) & ("RIGHT" in directions):
            final_move_string = "RIGHT"
        else:
            if (state[9] == 1) & ("UP" in directions):
                final_move_string = "UP"
            elif (state[10] == 1) & ("DOWN" in directions):
                final_move_string = "DOWN"
            else:
                final_move_string = directions[0]

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


def play(games):
    total_score = 0
    record = 0
    average_reward = 0
    print('creating agent')
    agent = Agent()
    print('creating game')
    game = Game()
    print('initializing game')

    for i in range(0, games):
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

                break

    with open("../results/hardcoded.txt", "a") as file:
        file.write('Average reward: {} - Record: {}'.format(average_reward, record))
        file.close()

if __name__ == '__main__':
    play(500)
