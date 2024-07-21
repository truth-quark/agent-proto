import os
import sys
import copy
from typing import Union
from datetime import datetime
from dataclasses import dataclass

import numpy as np

from agent import viz
from agent.components import Grid, adjacent_coords
from agent.components import NODATA, Y_OFFSETS, X_OFFSETS

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
# TODO: allow agents to go uphill to spot food/water, then search for it
#       (have them retain memory for N locations of interest)
# TODO: could also have agents start/return to home base (daily simulation)
# TODO: vision: build a 2-3 cell radius algorithm (check PCTL?)

# TODO: with numpy 2.0
#       find out why int + np.int resolves to an np.int (breaks the code)
#       find/solve overflow in average energy calc (an int8 issue with np 2.0?)
#       debug overflow in scalar subtract too

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


# TODO: record history in the workflow/separate concerns
#       init_energy (get from setup steps?)
#       move_history
#       harvest_history


@dataclass
class Agent:
    """Simple agent with basic stats."""

    # TODO: add death/max # of lifecycles ?
    id: Union[int, str]
    vision: int
    metabolism: int
    energy: int = 0
    coords: tuple = None  # TODO move coords to simulation or metadata container?

    def is_alive(self):
        return self.energy > 0

    def is_dead(self):
        return self.energy <= 0

    @property
    def name(self):
        return f"Agent {self.id}"


def on_end_turn(agent: Agent):
    """Callback to handle changes to the agent at the end of each turn."""
    agent.energy -= agent.metabolism


def next_move(agent, view, adj_agents=None):
    """Simulates simple searching behaviour by an agent, simply looking for
    the most productive cell in the adjacent cells."""
    best_energy = NODATA
    best_dir = -1
    adj_energy = {}  # cache energy data for possible later search

    # scan around the *local* view looking for energy and agents
    # TODO: loop approach is clockwise, which biases agent search & move to
    #       same default every time. Can tweak/make search patterns different
    #       by agent. ALT: add proximity/vision search to automatically go
    #       for the weighted best looking area
    #
    # TODO: look at pushing decision process out to a navigation module?
    #       Could add pluggable nav behaviour/different for each agent
    for d, adj_coord in enumerate(adjacent_coords((1, 1))):
        if adj_agents:
            if adj_agents.get(d):
                continue  # skip cells occupied by other agents

        energy = view[adj_coord]
        if energy > 0:
            adj_energy[d] = energy

            if energy > best_energy:
                best_energy = energy
                best_dir = d

        elif energy == NODATA:
            # cache NODATA cells to prevent illegal agent moves
            adj_energy[d] = NODATA

    if best_energy > 0:
        d = best_dir
    else:
        # must be surrounded by NODATA/bounds or zero energy land
        if not isinstance(agent.id, int):
            raise NotImplementedError()

        d = _search_direction(adj_energy, default_direction=agent.id % 8)

    y, x = agent.coords
    return y + Y_OFFSETS[d], x + X_OFFSETS[d]  # on world grid


def _search_direction(adj_energy, default_direction):
    # no energy nearby, so move in first possible direction using id as seed
    # won't always work well as some agents will run around borders
    #
    # TODO: better deterministic search algorithm
    # TODO: experiment with more intelligent agents (climb hill or follow river)
    # TODO: fix agents getting stuck in corners/repeating same move
    assert 0 <= default_direction <= 7
    direction = default_direction

    for _ in range(8):  # scan all directions & pick 1st direction from initial seed
        direction %= 8
        if adj_energy.get(direction) != NODATA:
            return direction

        direction += 1

    # HACK: as final option, have agent not move/wait for energy respawn?
    # return self.coords
    raise NotImplementedError('Add better search algorithm')


# TODO: supply dict to allow runtime settings to be tweaked
#       eg. post harvest cell recovery time
class Simulation:

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
        # TODO: coroutine decorator?
        _dir = self.config.get('VIZ_OUTPUT_DIR')
        if _dir:
            self.take_snapshot = viz.snapshot_image(self.world.food_grid, _dir, scale=10)
            self.take_snapshot.__next__()

        self.move_history = {agent.id: [] for agent in agents}
        self.last_view = {}

    def run(self, num_rounds):
        for n in range(num_rounds):
            if not(self.do_step()):
                self.final_round = n+1
                return

        self.final_round = num_rounds
        return

    def do_step(self):
        """Run a single round or timestep of the simulation."""
        living_agents = self.live_agents

        for a in living_agents:
            view = self.world.food_grid.view(*a.coords, size=1)  # TODO: change to vision size
            adj_agents = self.adjacent_agents(a)
            next_coord = next_move(a, view, adj_agents)

            if next_coord == a.coords:  # agent is stuck/waiting
                assert self.world.food_grid[next_coord] <= 0

            self.move_history[a.id].append(a.coords)

            a.coords = next_coord

            # TODO: only harvest if energy > 0
            # TODO: add one action per turn logic (move OR harvest OR wait)
            a.energy += self.world.harvest(a.coords)  # TODO: add recovery time setting
            on_end_turn(a)  # can kill an agent

            if a.is_dead():
                # cache view where the agent died for reporting
                self.last_view[a.id] = copy.copy(view)

        if hasattr(self, 'take_snapshot'):
            # snapshots here show agents that just died
            # TODO: sometimes can't see fully respawned cells as agents move onto them
            self.take_snapshot.send(living_agents)

        if self.live_agents:
            self.collect_stats()
            self.world.on_end_round()

        # TODO: can snapshot here to display respawns before next round of moves
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

    def report(self, out, basename):
        """Prints rough report of simulation details."""

        def sub_report(_agent):
            print(file=out)
            print(_agent, file=out)

            if hasattr(_agent, "harvest_history"):
                print('Energy harvests:', _agent.harvest_history, file=out)

            if hasattr(_agent, "move_history"):
                print('\nMoves:', _agent.move_history, file=out)
            print(file=out)

        print(f'Agent Simulation: {basename}\n', file=out)
        print('Per turn data:', file=out)
        print('--------------', file=out)
        print('Got to round:   ', self.final_round, file=out)
        print('\nNum dead agents:', self.num_dead_agents, file=out)
        print('\nAverage energy by step: ', self.average_energy, file=out)
        print('\nAverage metabolism by step: ', self.average_metabolism, file=out)

        live_agents = [a for a in self.agents if a.is_alive()]
        live_agents.sort(key=lambda x: x.energy, reverse=True)

        print('\nLive Agents - Stats', file=out)
        print('-------------------', file=out)
        for a in live_agents:
            sub_report(a)
            print('--------------------', file=out)

        print('\nDead Agents - Stats', file=out)
        print('-------------------', file=out)

        dead_agents = [a for a in self.agents if a.is_dead()]

        for a in dead_agents:
            sub_report(a)

            # TODO: fix move history
            # print(f"Died at step {len(a.move_history)}", file=out)
            # print('Final view:\n', a.last_view, file=out)
            print('--------------------', file=out)


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

    return [Agent(n, *args) for n, args in enumerate(zip(vision, metabolism, energy, coords))]


def format_date(n):
    attrs = [getattr(n, a) for a in ('year', 'month', 'day', 'hour', 'minute')]
    name = 'simrun_{}_{:02d}_{:02d}_{:02d}_{:02d}.txt'.format(*attrs)
    return name


def get_config(path):
    # TODO: use ConfigParser?
    with open(path) as fd:
        lines = fd.readlines()
        config = dict(line.strip().split('=') for line in lines if '=' in line)
    return config


def main():
    # run the default simulation
    # TODO: cmd line option for skipping viz
    food_grid_path = './data/basic_grid.txt' if len(sys.argv) == 1 else sys.argv[1]
    settings_path = os.path.join(os.environ['HOME'], '.agentsim.rc')
    config = get_config(settings_path)

    with open(food_grid_path) as fd:
        food_grid = Grid.from_file(fd)
        agents = generate_agents_deterministic()
        simulation = Simulation(food_grid, agents, config)
        simulation.run(200)

        now = datetime.now()
        basename = format_date(now)
        path = os.path.join(config['REPORT_OUTPUT_DIR'], basename)

        with open(path, 'w') as f:
            simulation.report(f, basename)
            print(path, 'saved')


if __name__ == '__main__':
    main()
