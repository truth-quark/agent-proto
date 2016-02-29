import copy
import random

from components import Grid

# TODO: create a world and populate with agents

# start with simple rules
# agents move one cell at a time
# do one activity per turn (move, get food)
# do a fixed number of turns first, then add die_after attribute

# end sim when all agents are dead


class BasicWorld(object):

    def __init__(self, food_grid_path):
        # grids for current food state & original state for respawns

        with open(food_grid_path) as fd:
            self.food_grid = Grid.from_file(fd)
            self.orig_food_grid = copy.copy(self.food_grid)


class BasicAgent(object):

    # TODO: add a death after x number of turns?
    def __init__(self, _id, vision, metabolism, energy):
        # vision = number of cells the agent can see in all dirs
        # metabolism = rate at which energy is used per turn
        # energy = current stock of food

        self.id = _id
        self.vision = vision
        self.metabolism = metabolism
        self.energy = energy



def generate_agents_deterministic():
    # create 25 agents from pre-canned data (1% coverage of the grid)
    vision = [1, 2, 2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 2, 2,
              2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2]

    metabolism = [2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2,
                  2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2]

    energy = [12, 25, 16, 26, 27, 22, 25, 24, 16, 15, 25, 20, 24,
              17, 23, 12, 18, 26, 20, 26, 23, 26, 16, 19, 26]

    return [BasicAgent(_id, v, m, e) for _id, (v, m, e) in enumerate(zip(vision, metabolism, energy))]


def generate_coords_deterministic():
    xc = [49, 28, 6, 41, 26, 10, 0, 19, 5, 47, 2, 18, 18, 27, 21, 31, 3, 40, 29, 9, 43, 7, 4, 34, 33]
    yc = [48, 2, 48, 24, 17, 33, 43, 8, 26, 47, 2, 18, 29, 38, 14, 31, 6, 7, 7, 1, 7, 19, 3, 25, 12]
    return zip(yc, xc)


def run_simulation(food_grid_path):
    world = BasicWorld(food_grid_path)
    agents = generate_agents_deterministic()
    start_coords = generate_coords_deterministic()


if __name__ == '__main__':
    import os; print os.getcwd()
    run_simulation('../data/basic_grid.txt')
