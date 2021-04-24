from neat.config import *
import neat
import os
import visualize

from Snake_Game import Game

POPULATION = 150

class Agent:
    def __init__(self, net):
        self.n_games = 0  # number of games / epochs
        self.net = net

    def get_state(self, game):
        return game.get_state()

    def get_action(self, state):
        move = [0, 0, 0, 0]
        net_output = self.net.activate(state)
        move[net_output.index(max(net_output))] = 1
        return move

def eval_genomes(genomes, config):
    snake_in_generation = 1

    for genome_id, genome in genomes:
        print("Snake: ", snake_in_generation, "/", POPULATION)

        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        play(net, genome)

        snake_in_generation += 1

def play(net, genome):
    print('creating agent')
    agent = Agent(net)
    print('creating game')
    game = Game()
    print('initializing game')

    time_out = 0

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        # state_new = agent.get_state(game)

        time_out += 1
        if time_out == 200:
            game.game_over()

        if reward == 10:
            genome.fitness += 10
            time_out = 0
        if reward == -10:
            genome.fitness -= 10
            time_out = 0

        if done:
            # train long memory, plot result
            game.reset()
            break


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    print(config.save("config_output"))

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, POPULATION)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    print(winner_net)

    node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)