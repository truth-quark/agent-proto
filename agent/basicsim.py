import copy
import random

import numpy as np

from components import Grid


Y_OFFSETS = (-1, -1, 0, 1, 1, 1, 0, -1)
X_OFFSETS = (0, 1, 1, 1, 0, -1, -1, -1)


# Start with simple rules
# agents move one cell at a time
# do one activity per turn (move, get food)
# do a fixed number of turns first, then add die_after attribute

# end sim when all agents are dead


class BasicWorld(object):

    def __init__(self, food_grid):
        # grids for current food state & original state (for energy respawn)
        self.food_grid = food_grid
        self.orig_food_grid = copy.deepcopy(food_grid)

    # TODO: figure out the coord problems with testing grids with where()
    def on_end_round(self, recovery_rate=1):
        # allow energy cells to recover slowly
        # TODO: row by row until the grid based where() bug is solved
        for r0, r1 in zip(self.food_grid, self.orig_food_grid):
            changed = np.where(r0 != r1)
            r0[changed] += recovery_rate


class BasicAgent(object):

    # TODO: add death after x number of turns?
    def __init__(self, _id, vision, metabolism, energy, coords=None):
        # vision = number of cells the agent can see in all dirs
        # metabolism = rate at which energy is used per turn
        # energy = current stock of food

        self.id = _id
        self.vision = vision
        self.metabolism = metabolism
        self.init_metabolism = metabolism
        self._energy = energy
        self.init_energy = energy
        self._coords = coords

        self.move_history = []
        self.harvest_history = []
        self.last_view = None  # snapshot of area at final turn

    def __str__(self):
        text = 'Agent {}: vis={} metabol={} energy={} coords={}'
        args = (self.id, self.vision, self.metabolism, self.energy, self.coords)
        return text.format(*args)

    def __repr__(self):
        return self.__str__()

    def is_alive(self):
        return self.energy > 0

    def is_dead(self):
        return self.energy <= 0

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, value):
        prev = self._energy
        self._energy = value
        self.harvest_history.append(value - prev)

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, _coords):
        if self._coords:
            self.move_history.append(self._coords)
        self._coords = _coords

    def on_turn_end(self):
        self._energy -= self.metabolism


class Simulation(object):

    def __init__(self, food_grid, agents):
        self.world = BasicWorld(food_grid)
        self.agents = agents

        # stats
        self.round = None
        self.average_energy = []
        self.average_metabolism = []
        self.num_dead_agents = []

    def run(self, num_rounds=200):
        for n in range(num_rounds):
            for a in self.live_agents:
                view = self.world.food_grid.view(*a.coords, size=1)
                adj_crd = self._best_adj_cell(a.coords, view)

                if adj_crd:
                    # TODO: shove into the agent class?
                    a.coords = adj_crd
                    a.energy += self.world.food_grid[adj_crd]
                    self.world.food_grid[adj_crd] = -1  # remove food from cell
                else:
                    # TODO: better deterministic search algorithm?
                    init_dir = a.id
                    count = 0

                    while True:
                        init_dir %= 8
                        tmp = (a.coords[0] + Y_OFFSETS[init_dir],
                               a.coords[1] + X_OFFSETS[init_dir])

                        if self.world.food_grid[tmp] >= 0:  # avoid NODATA
                            a.coords = tmp
                            a.energy += 0  # HACK: records no energy gained during turn
                            break
                        else:
                            init_dir += 1

                        count += 1
                        if count == 8:
                            break  # HACK: get the agent to wait for regrowth

                a.on_turn_end()
                if a.is_dead():
                    a.last_view = self.world.food_grid.view(*a.coords, size=1)

            if self.live_agents:
                self.collect_stats()
                # TODO: self.world.on_end_round()
            else:
                break
        self.round = n

    @property
    def live_agents(self):
        return [a for a in self.agents if a.is_alive()]

    def collect_stats(self):
        # collect basic stats at the end of each round
        n_agents = len(self.live_agents)

        if not n_agents:
            return

        avg_energy = float(sum(a.energy for a in self.live_agents)) / n_agents
        self.average_energy.append(round(avg_energy, 2))
        avg_metabolism = float(sum(a.metabolism for a in self.live_agents)) / n_agents
        self.average_metabolism.append(round(avg_metabolism, 2))
        self.num_dead_agents.append(sum(a.is_dead() for a in self.agents))

    def report(self):
        from pprint import pprint

        def sub_report(_agent):
            print _agent
            print 'Energy harvests:', _agent.harvest_history
            print 'Moves:', _agent.move_history
            print

        print 'Per turn data:'
        print '--------------'
        print 'Got to round:   ', self.round
        print 'Num dead agents:', self.num_dead_agents
        print 'Average energy: ', self.average_energy
        print 'Average metabolism: ', self.average_metabolism

        live_agents = [a for a in self.agents if a.is_alive()]
        live_agents.sort(key=lambda x: x.energy, reverse=True)

        print '\nLive Agents - Stats'
        print '---------------------'
        for a in live_agents:
            sub_report(a)

        print '\nDead Agents - Stats'
        print '---------------------'
        dead_agents = [a for a in self.agents if a.is_dead()]
        for a in dead_agents:
            sub_report(a)
            print 'Final view:\n', a.last_view

    def _best_adj_cell(self, coords, view):
        best = -10  # TODO: to NODATA?
        best_crd = None

        for i, (yoff, xoff) in enumerate(zip(Y_OFFSETS, X_OFFSETS)):
            adj = (coords[0] + yoff, coords[1] + xoff)

            # ignore adjacent cells occupyied by other agents
            found = False
            for a in self.agents:
                if a.coords == adj:
                    found = True
                    break

            if not found:
                adj_energy = self.world.food_grid[adj]

                if adj_energy > 0 and adj_energy > best:
                    best_crd = adj
                    best = adj_energy

        return best_crd


def generate_agents_deterministic():
    # create 25 agents from pre-canned data (1% coverage of the grid)
    vision = [1, 2, 2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 2, 2,
              2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2]

    metabolism = [2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2,
                  2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2]

    energy = [12, 25, 16, 26, 27, 22, 25, 24, 16, 15, 25, 20, 24,
              17, 23, 12, 18, 26, 20, 26, 23, 26, 16, 19, 26]

    xc = [49, 28, 6, 41, 26, 10, 0, 19, 5, 47, 2, 18, 18, 27, 21, 31, 3, 40, 29, 9, 43, 7, 4, 34, 33]
    yc = [48, 2, 48, 24, 17, 33, 43, 8, 26, 47, 2, 18, 29, 38, 14, 31, 6, 7, 7, 1, 7, 19, 3, 25, 12]
    coords = zip(yc, xc)

    return [BasicAgent(_id, v, m, e, c) for _id, (v, m, e, c) in
                enumerate(zip(vision, metabolism, energy, coords))]


if __name__ == '__main__':
    food_grid_path = '../data/basic_grid.txt'

    with open(food_grid_path) as fd:
        food_grid = Grid.from_file(fd)
        agents = generate_agents_deterministic()
        simulation = Simulation(food_grid, agents)
        simulation.run()
        simulation.report()
