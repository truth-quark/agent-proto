import copy
import random

import numpy as np

from components import Grid, NODATA


Y_OFFSETS = (-1, -1, 0, 1, 1, 1, 0, -1)
X_OFFSETS = (0, 1, 1, 1, 0, -1, -1, -1)


# Start with simple rules
# agents move one cell at a time
# do one activity per turn (move, get food)?
# do a fixed number of turns first, then add die_after attribute

# Other features:
# energy cells that can become damaged/not recover - force agents to explore
# need for travel to water

# viewing options:
# lifetime stats for each agent, as a graph (graph harvests and consumption)
# output images of the grid + agents on it, showing changes over time


class BasicWorld(object):

    def __init__(self, food_grid):
        # grids for current food state & original state (for energy respawn)
        self.food_grid = food_grid
        self.orig_food_grid = copy.deepcopy(food_grid)

    def harvest(self, coords, post_harvest=-1):
        """Harvests and returns the energy from a cell."""
        energy = self.food_grid[coords]
        self.food_grid[coords] = post_harvest
        return energy

    # TODO: figure out the coord problems with testing grids with where()
    def on_end_round(self, recovery_rate=1):
        # allow energy cells to recover slowly
        # TODO: row by row until the grid based where() bug is solved
        for r0, r1 in zip(self.food_grid, self.orig_food_grid):
            changed = np.where(r0 != r1)
            r0[changed] += recovery_rate


# FIXME: move behaviour/decision making into agent class
# allows different types of agents to be used/make different decisions
class BasicAgent(object):
    """Simple agent with basic stats."""

    # TODO: add death after x number of turns?
    def __init__(self, _id, vision, metabolism, energy, coords=None):
        # vision = number of cells the agent can see in all dirs
        # metabolism = rate at which energy is used per turn
        # energy = current stock of food

        self.id = _id
        self.vision = vision
        self.metabolism = metabolism
        self.init_metabolism = metabolism  # historical record
        self._energy = energy
        self.init_energy = energy  # historical record
        self._coords = coords

        self.move_history = [coords] if coords else []
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
    def coords(self, coords):
        if self._coords:
            self.move_history.append(coords)
        self._coords = coords

    def on_end_turn(self):
        """Callback to handle changes to the agent at the end of each turn."""
        self._energy -= self.metabolism

    def next_move(self, view):
        """Simulates simple searching behaviour by an agent, simply looking for
        the most productive cell in the adjacent cells."""
        best = NODATA
        best_coord = None
        y, x = self.coords
        energy = [None] * 8  # cache energy data for possible later search

        for i, (yoff, xoff) in enumerate(zip(Y_OFFSETS, X_OFFSETS)):
            adj_coord = (1+yoff, 1+xoff)  # treat current coords as 1,1
            adj_energy = view[adj_coord]

            if adj_energy > -1:
                energy[i] = adj_energy

            if adj_energy and adj_energy > best:
                best_coord = (y + yoff, x + xoff)  # NB: world grid coords
                best = adj_energy

        if not best_coord:
            best_coord = self._search_direction(energy)

        return best_coord

    def _search_direction(self, adjacent):
        # no energy nearby, start a search off in first possible direction
        # won't work on borders as it will cause an agent to run around edges
        # TODO: better deterministic search algorithm?

        direction = self.id  # FIXME: relies on id being numeric

        if direction:  # reorder array to start with ID based direction
            adjacent = adjacent[direction:] + adjacent[:direction]

        # scan in all directions & pick first open direction from initial seed
        for i, energy in enumerate(adjacent):
            if energy == 0:
                y, x = self.coords
                d = (direction + i) % 8
                return (y + Y_OFFSETS[d], x + X_OFFSETS[d])

        # HACK: as final option, have agent not move/wait for energy respawn
        return self.coords


class Simulation(object):

    def __init__(self, food_grid, agents):
        self.world = BasicWorld(food_grid)
        self.agents = agents

        # stats
        self.final_round = None
        self.average_energy = []
        self.average_metabolism = []
        self.num_dead_agents = []

    def run(self, num_rounds):
        for n in range(num_rounds):
            if not(self.do_round()):
                self.final_round = n+1
                return

        self.final_round = num_rounds
        return

    def do_round(self):
        """Run a single round or timestep of the simulation."""
        for a in self.live_agents:
            view = self.world.food_grid.view(*a.coords, size=1)
            # TODO: remove other agents occupied cells here
            # ignore adjacent cells occupyied by other agents
            #found = False
            #for a in self.agents:
                #if a.coords == adj:
                    #found = True
                    #break

            #if not found:

            next_coord = a.next_move(view)

            if next_coord == a.coords:  # agent is stuck/waiting
                assert self.world.food_grid[next_coord] == 0

            a.coords = next_coord
            a.energy += self.world.harvest(next_coord)
            a.on_end_turn()

            if a.is_dead():
                # cache area where the agent died for reference
                data = copy.copy(self.world.food_grid.view(*a.coords, size=1))
                a.last_view = data

        if self.live_agents:
            self.collect_stats()
            self.world.on_end_round()

        return len(self.live_agents)

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
        print 'Got to round:   ', self.final_round
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
        simulation.run(200)
        simulation.report()
