import pickle
import time

import neat
import os
import visualize

from multiprocessing import Process
from Snake_Game_Multiple_NoGUI_Parallel import Game

WEIGHT_VALUES = [-2, -1, -0.5, 0.5, 1, 2]

class Agent:
    def __init__(self, nets):
        self.n_games = 0  # number of games / epochs
        self.nets = nets

    def get_state(self, game, snake):
        return game.get_state(snake)

    def get_action(self, state, snake):
        move = [0, 0, 0, 0]
        net_output = self.nets[snake].activate(state)
        move[net_output.index(max(net_output))] = 1
        return move


def eval_genomes(genomes, config):
    nets = []
    genomes_forwarded = []

    for weight_value in WEIGHT_VALUES:
        print("Evaluating weight: ", weight_value)

        for genome_id, genome in genomes:
            genome.fitness = 0

            for connection in genome.connections.items():
                connection[1].weight = weight_value

            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nets.append(net)
            genomes_forwarded.append(genome)

        play(nets, genomes_forwarded, len(nets))


def play(nets, genomes, population):

    # print('creating agent')
    agent = Agent(nets)
    # print('creating game')
    game = Game(population)
    # print('initializing game')

    all_done = 0
    time_outs = []

    for net in nets:
        time_outs.append(0)

    while True:
        for snake in range(1, population + 1):
            if not game.game_overs[snake - 1]:
                # get old state
                state_old = agent.get_state(game, snake - 1)

                # get move
                final_move = agent.get_action(state_old, snake - 1)

                # perform move and get new state
                reward, done, score = game.play_step(final_move, snake - 1)
                # state_new = agent.get_state(game)

                time_outs[snake - 1] += 1
                if time_outs[snake - 1] >= 500:
                    game.game_over(snake - 1)

                if reward == 10:
                    genomes[snake - 1].fitness += 10
                    time_outs[snake - 1] = 0
                if reward == -10:
                    genomes[snake - 1].fitness -= 10
                    time_outs[snake - 1] = 0

                if done | game.game_overs[snake - 1]:
                    all_done += 1

        if all_done == population:
            game.reset()
            break


def play_generation(game, agent, start, end, time_outs, genomes, all_done):
    local_done = 0




def run(config_file, number_of_generations=100):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, number_of_generations)

    save_path = "WANNmodel" + time.strftime("%Y%m%d-%H%M%S")

    model_folder_path = '../neat-model/' + save_path
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    visualize.plot_stats(stats, False, False, "../neat-model/" + save_path + "/avg_fitness-generations.png")

    # Save winner
    with open("../neat-model/" + save_path + "/winner.pkl", "wb") as file:
        pickle.dump(winner, file)
        file.close()
        config.save("../neat-model/" + save_path + "/config")

    visualize.draw_net(config, winner, True, "../neat-model/" + save_path + "/visualization", None, True, False)



    # Show output of the most fit genome against training data.
    with open("../neat-model/" + save_path + "/output", "a") as file:
        file.write(save_path)

        # Display the winning genome.
        file.write('\nBest genome:\n{!s}'.format(winner))

        file.write("Output:")
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        file.write('\n' + str(winner_net.input_nodes))
        file.write('\n' + str(winner_net.node_evals))
        file.write('\n' + str(winner_net.output_nodes))
        file.write('\n' + str(winner_net.values))

def replay(folder, vis=False, population=1):
    config_location = "../neat-model/" + folder + "/config"
    winner_location = "../neat-model/" + folder + "/winner.pkl"

    # Load requried NEAT config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_location)

    # Unpickle saved winner
    with open(winner_location, "rb") as f:
        genome = pickle.load(f)

    # Convert loaded genome into required data structure
    genomes = [(1, genome)]

    if vis:
        visualize.draw_net(config, genome, True)

    # Call game with only the loaded genome
    eval_genomes(genomes, config)

def run_cpu_tasks_in_parallel(tasks):
    running_tasks = [Process(target=task) for task in tasks]
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    train_or_play = input("Do you want to train (1) or load a model and play (2): ")
    if train_or_play == "1":
        local_dir = os.path.dirname(__file__)
        configName = input("Give the name of the config (default=configWANNs/configWANN): ")
        if configName == "":
            configName = "configWANNs/configWANN"
        config_path = os.path.join(local_dir, configName)

        number_of_generations = input("Give the amount of generations you want to run for (default=100): ")
        if number_of_generations == "":
            number_of_generations = 100

        run(config_path, int(number_of_generations))
    elif train_or_play == "2":
        folder = input("Give the folder where the model is stored: ")
        counter = 0
        while True:
            replay(folder)
