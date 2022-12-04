import unittest
import StringIO

import components
from basicsim import BasicAgent, Simulation


def generate_basic_simulation():
    food_grid = components.Grid.from_file(StringIO.StringIO(DATA))
    agent = BasicAgent(_id=0, vision=2, metabolism=1, energy=21, coords=(2,2))
    return Simulation(food_grid, [agent])


def test_simulation_single_step():
    sim = generate_basic_simulation()
    sim.do_round()

    agent = sim.live_agents[0]
    assert len(sim.live_agents) == 1
    assert agent.coords == (1,1)
    assert agent.energy == 21  # loses 1, gains 1

    # check the world
    assert sim.world.food_grid[1,1] == 0  # harvested & should respawn 1 point


# run a full test of a small scale simulation
def test_simulation():
    sim = generate_basic_simulation()
    sim.run(3)

    agent = sim.live_agents[0]
    assert len(sim.live_agents) == 1

    # TODO: add more testing ...


def test_respawn_trail():
    # check that the respawn behind an agent moving along is correct
    # TODO: this might be better in the world tests?
    # TODO: check harvest is working properly? (maybe change the rate to every 2 turns?)
    1/0


def test_adjacent_agents():
    sim = generate_basic_simulation()
    agent2 = BasicAgent(_id=1, vision=1, metabolism=2, energy=33, coords=(3,1))
    agent3 = BasicAgent(_id=2, vision=1, metabolism=2, energy=44, coords=(1,1))
    sim.agents += [agent2, agent3]
    res = sim.adjacent_agents(sim.agents[0])
    assert res == {5:agent2, 7: agent3}


class HarvestTests(unittest.TestCase):

    def setUp(self):
        self.sim = generate_basic_simulation()
        self.agent = self.sim.agents[0]
        self.agent.id = 7  # force NW travel

    def test_harvest_on_zero_cell(self):
        # agents should not harvest anything on a zero cell
        self.sim.world.food_grid[:] = 0
        self.sim.do_round()
        assert self.agent.energy == 20  # only metabolism amount
        assert self.agent.coords == (1,1)

    def test_harvest_on_negative_cell(self):
        # agents shouldn't harvest negative values from recovering energy cells
        self.sim.world.food_grid[:] = -5
        self.sim.do_round()
        assert self.agent.energy == 20  # only metabolism amount
        assert self.agent.coords == (1,1)


DATA = '''21000
11000
00000
00000
'''
