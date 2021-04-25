import torch
import random
from collections import deque

from Snake_Game import Game
from datetime import datetime

from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
MAX_MEMORY_INITIAL = 20_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0  # number of games / epochs
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        # self.initial_memory = deque(maxlen=MAX_MEMORY_INITIAL)
        self.model = Linear_QNet(11, 256, 128, 128, 4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        self.initial_epsilon = 1
        self.final_epsilon = 0.1
        self.num_decay_epochs = 100
        self.remember_counter = 0

    def get_state(self, game):
        return game.get_state()

    def remember(self, state, action, reward, next_state, done):
        # if self.remember_counter < 20000:
        #     self.initial_memory.append((state, action, reward, next_state, done))
        #     self.remember_counter += 1
        # else:
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        # combined_memory = self.initial_memory + self.memory

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        # self.epsilon = 500 - self.n_games
        # if self.epsilon < 50:
        #    self.epsilon = 50

        # New version of decaying epsilon
        self.epsilon = self.final_epsilon + (max(self.num_decay_epochs - self.n_games, 0) *
                                             (self.initial_epsilon - self.final_epsilon) / self.num_decay_epochs)

        final_move = [0, 0, 0, 0]

        randomNumber = random.random()

        if randomNumber <= self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    print('creating agent')
    agent = Agent()
    print('creating game')
    game = Game()
    print('initializing game')
    average_reward = 0
    difference = 0

    file = open("Statistics.txt", "a")
    output = "\n----------------------New Training----------------------\n" \
             "{}\n" \
             "Parameters:\n\tMAX_MEMORY: {}\n\tMAX_MEMORY_INITIAL: {}\n\tBATCH_SIZE: {}\n\tLR: {}\n\tgamma: {}" \
             "\n\tmodel: {}\n\tfinal_epsilon: {}\n\tnum_decay_epochs: {}\n" \
             "--------------------------------------------------------\n"\
        .format(datetime.now() ,MAX_MEMORY, MAX_MEMORY_INITIAL, BATCH_SIZE, LR, agent.gamma, agent.model, agent.final_epsilon,
                agent.num_decay_epochs)
    file.write(output)
    file.close()

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            gameLength = game.gameLength
            totalReward = game.total_reward
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

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

            plot_scores.append(totalReward)
            total_score += totalReward

            if (agent.n_games % 100) == 0:
                file = open("Statistics.txt", "a")
                output = "Game: {} -- Average reward: {}\n".format(agent.n_games, average_reward)
                file.write(output)
                file.close()
            # mean_score = total_score / agent.n_games
            # plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)

def loadModelAndPlay(path):
    # Load in the model
    model = Linear_QNet(11, 256, 128, 128, 4)
    print(model.load_state_dict(torch.load(path)))
    print(model.eval())

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    print('creating agent')
    agent = Agent()
    print('creating game')
    game = Game()
    print('initializing game')
    average_reward = 0
    difference = 0

    while True:
        # get old state
        current_state = agent.get_state(game)

        # get move
        final_move = [0, 0, 0, 0]

        state0 = torch.tensor(current_state, dtype=torch.float)
        prediction = model.forward(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1

        # perform move and get new state
        reward, done, score = game.play_step(final_move)

        if done:
            # train long memory, plot result
            gameLength = game.gameLength
            totalReward = game.total_reward
            game.reset()
            agent.n_games += 1

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

            plot_scores.append(totalReward)
            total_score += totalReward

if __name__ == '__main__':
    # train()
    loadModelAndPlay("D:\Documenten\PythonPrograms\Snake-Reinforcement-Learning\model\model 1 hidden 256\model.pth")
