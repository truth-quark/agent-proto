import os
import copy
import random

import numpy as np

import viz
from components import Grid, adjacent_coords
from components import NODATA, Y_OFFSETS, X_OFFSETS

# Start with simple rules
# agents move one cell at a time
# do one activity per turn (move, get food)?
# do a fixed number of turns first, then add die_after attribute

# Other features:
# energy cells that can become damaged/not recover - force agents to explore
# need for travel to water
# have agents interact (trade energy for water etc?)

# viewing options:
# lifetime stats for each agent, as a graph (graph harvests and consumption)


# TODO: refactor out a base simulation class
# TODO: make simulations with topo and water
# TODO: allow agents to go uphill to get a view, then search for food
#       (ie have a memory for locations of interest)

class BasicWorld(object):

    def __init__(self, food_grid):
        # grids for current food state & original state (for energy respawn)
        self.food_grid = food_grid
        self.orig_food_grid = copy.deepcopy(food_grid)

    def harvest(self, coords, post_harvest=-1):
        """Harvests and returns the energy from a cell."""
        energy = self.food_grid[coords]

        if energy > 0:
            self.food_grid[coords] = post_harvest
            return energy

        return 0  # harvest nothing

    # TODO: figure out the coord problems with testing grids with where()
    def on_end_round(self, recovery_rate=1):
        # allow energy cells to recover slowly
        # TODO: row by row until the grid based where() bug is solved
        for r0, r1 in zip(self.food_grid, self.orig_food_grid):
            changed = np.where(r0 != r1)
            r0[changed] += recovery_rate


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

    def next_move(self, view, adj_agents=None):
        """Simulates simple searching behaviour by an agent, simply looking for
        the most productive cell in the adjacent cells."""
        best = NODATA
        best_coord = None
        y, x = self.coords
        adj_energy = {}  # cache energy data for possible later search

        # scan around the *local* view looking for energy and agents
        for d, adj_coord in enumerate(adjacent_coords((1,1))):
            if adj_agents:
                if adj_agents.get(d):
                    continue  # skip cells occupied by other agents

            energy = view[adj_coord]
            if energy > 0:
                adj_energy[d] = energy

                if energy > best:
                    best_coord = (y + Y_OFFSETS[d], x + X_OFFSETS[d])  # NB: world grid coords
                    best = energy
            elif energy == NODATA:
                # cache NODATA cells to prevent illegal agent moves
                adj_energy[d] = NODATA

        return best_coord if best_coord else self._search_direction(adj_energy)

    def _search_direction(self, adj_energy):
        # no energy nearby, so move in first possible direction using id as seed
        # won't always work well as some agents will run around borders
        # TODO: better deterministic search algorithm?
        direction = self.id  # FIXME: relies on id being numeric
        for _ in range(8):  # scan all directions & pick 1st direction from initial seed
            direction %= 8
            if adj_energy.get(direction) != NODATA:
                y, x = self.coords
                return (y + Y_OFFSETS[direction], x + X_OFFSETS[direction])
            else:
                direction += 1

        # HACK: as final option, have agent not move/wait for energy respawn?
        # return self.coords
        raise NotImplementedError('Add better search algorithm')


# TODO: supply dict to allow runtime settings to be tweaked
# eg. post harvest cell recovery time
class Simulation(object):

    def __init__(self, food_grid, agents, config=None):
        self.world = BasicWorld(food_grid)
        self.agents = agents
        self.config = config if config else {}

        # stats
        self.final_round = None
        self.average_energy = []
        self.average_metabolism = []
        self.num_dead_agents = []

        # viz coroutine
        _dir = self.config.get('VIZ_OUTPUT_DIR')
        if _dir:
            self.take_snapshot = viz.snapshot_image(self.world.food_grid, _dir, scale=10)
            self.take_snapshot.next()

    def run(self, num_rounds):
        for n in range(num_rounds):
            if not(self.do_round()):
                self.final_round = n+1
                return

        self.final_round = num_rounds
        return

    def do_round(self):
        """Run a single round or timestep of the simulation."""
        living_agents = self.live_agents

        for a in living_agents:
            view = self.world.food_grid.view(*a.coords, size=1)
            adj_agents = self.adjacent_agents(a)
            next_coord = a.next_move(view, adj_agents)

            if next_coord == a.coords:  # agent is stuck/waiting
                assert self.world.food_grid[next_coord] == 0

            a.coords = next_coord
            a.energy += self.world.harvest(next_coord)  # TODO: add recovery time setting
            a.on_end_turn()

            if a.is_dead():
                # cache view where the agent died for reference
                data = copy.copy(self.world.food_grid.view(*a.coords, size=1))
                a.last_view = data

        if hasattr(self, 'take_snapshot'):
            # snapshots here show agents that just died
            # TODO: sometimes can't see fully respawned cells as agents move onto them
            self.take_snapshot.send(living_agents)

        if self.live_agents:
            self.collect_stats()
            self.world.on_end_round()

        return len(self.live_agents)

    def adjacent_agents(self, agent):
        """Scan around given agent for any adjacent agents."""

        def close_range(i):
            return range(i-1, i+2)

        y, x = agent.coords
        first_pass = [a for a in self.live_agents if a.coords[0] in close_range(y)]
        second_pass = [a for a in first_pass if a.coords[1] in close_range(x)]

        adj_agents = {}
        if second_pass:
            for i, adj_coord in enumerate(adjacent_coords(agent.coords)):
                for a in second_pass:
                    if a.coords == adj_coord:
                        adj_agents[i] = a

        return adj_agents

    @property
    def live_agents(self):
        # TODO: could cache previous live agents
        return [a for a in self.agents if a.is_alive()]

    def collect_stats(self):
        n_agents = len(self.live_agents)

        if not n_agents:
            return

        avg_energy = float(sum(a.energy for a in self.live_agents)) / n_agents
        self.average_energy.append(round(avg_energy, 2))
        avg_metabolism = float(sum(a.metabolism for a in self.live_agents)) / n_agents
        self.average_metabolism.append(round(avg_metabolism, 2))
        self.num_dead_agents.append(sum(a.is_dead() for a in self.agents))

    def report(self, out):
        """Prints rough report of simulation details."""

        def sub_report(_agent):
            print >> out, _agent
            print >> out, 'Energy harvests:', _agent.harvest_history
            print >> out, 'Moves:', _agent.move_history
            print >> out

        print >> out, 'Per turn data:'
        print >> out, '--------------'
        print >> out, 'Got to round:   ', self.final_round
        print >> out, 'Num dead agents:', self.num_dead_agents
        print >> out, 'Average energy: ', self.average_energy
        print >> out, 'Average metabolism: ', self.average_metabolism

        live_agents = [a for a in self.agents if a.is_alive()]
        live_agents.sort(key=lambda x: x.energy, reverse=True)

        print >> out, '\nLive Agents - Stats'
        print >> out, '---------------------'
        for a in live_agents:
            sub_report(a)
            print >> out, '--------------------'

        print >> out, '\nDead Agents - Stats'
        print >> out, '---------------------'
        dead_agents = [a for a in self.agents if a.is_dead()]
        for a in dead_agents:
            sub_report(a)
            print >> out, 'Final view:\n', a.last_view
            print >> out, '--------------------'


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


def default_filename(_dir):
    from datetime import datetime
    n = datetime.now()
    attrs = [getattr(n, a) for a in ('year', 'month', 'day', 'hour', 'minute')]
    name = 'simrun_{}_{:02d}_{:02d}_{:02d}_{:02d}.txt'.format(*attrs)
    return os.path.join(_dir, name)


def get_config(path):
    # TODO: import ConfigParser
    with open(path) as fd:
        lines = fd.readlines()
        config = dict(line.strip().split('=') for line in lines if '=' in line)
    return config


if __name__ == '__main__':
    # run the default simulation
    # TODO: cmd line option for skipping viz
    food_grid_path = '../data/basic_grid.txt'
    settings_path = os.path.join(os.environ['HOME'], '.agentsim.rc')
    config = get_config(settings_path)

    with open(food_grid_path) as fd:
        food_grid = Grid.from_file(fd)
        agents = generate_agents_deterministic()
        simulation = Simulation(food_grid, agents, config)
        simulation.run(200)

        path = default_filename(config['REPORT_OUTPUT_DIR'])
        with open(path, 'w') as f:
            simulation.report(f)
            print path, 'saved'
