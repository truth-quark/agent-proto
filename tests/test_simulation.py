from agent import basicsim, components
from agent.basicsim import Agent, Simulation

from io import StringIO

import numpy as np
import pytest


# TODO: split sim funcs out to standalone pieces, can test SANS simulation steps


ENERGY_DATA = '''21000
11000
00000
00000
'''


@pytest.fixture
def sim():
    energy_grid = components.Grid.from_file(StringIO(ENERGY_DATA))
    agent = Agent(id=0, vision=2, metabolism=1, energy=21, coords=(2, 2))
    return Simulation(energy_grid, [agent])


def test_simulation_single_step(sim):
    sim.do_step()

    agent = sim.live_agents[0]
    assert len(sim.live_agents) == 1
    assert agent.coords == (1, 1)
    assert agent.energy == 21  # loses 1, gains 1

    # check the world
    assert sim.world.food_grid[1, 1] == 0  # harvested & should respawn 1 point


# run a full test of a small scale simulation
def test_simulation(sim):
    sim.run(3)

    # agent = sim.live_agents[0]
    assert len(sim.live_agents) == 1

    # TODO: add more testing ...


def test_respawn_trail():
    # check that the respawn behind an agent moving along is correct
    # TODO: this might be better in the world tests?
    # TODO: check harvest is working properly? (maybe change the rate to every 2 turns?)
    raise NotImplementedError


def test_adjacent_agents(sim):
    agent2 = Agent(id=1, vision=1, metabolism=2, energy=33, coords=(3, 1))
    agent3 = Agent(id=2, vision=1, metabolism=2, energy=44, coords=(1, 1))
    sim.agents.extend([agent2, agent3])
    res = sim.adjacent_agents(sim.agents[0].coords)
    assert res == {5: agent2, 7: agent3}


# @pytest.mark.skip
# class HarvestTests(unittest.TestCase):
#
#     def setUp(self):
#         self.sim = sim()
#         self.agent = self.sim.agents[0]
#         self.agent.id = 7  # force NW travel


DEFAULT_NW_ID = 7

ENERGY_DATA_ALT = '''00000
00000
00000
00000
'''


@pytest.fixture
def agent():
    return Agent(id=DEFAULT_NW_ID, vision=2, metabolism=1, energy=21, coords=(2, 2))


@pytest.fixture
def sim2(agent):
    energy_grid = components.Grid.from_file(StringIO(ENERGY_DATA_ALT))
    return Simulation(energy_grid, [agent])


def test_harvest_on_zero_cell(agent, sim2):
    # agents should not harvest anything on a zero cell
    assert np.sum(sim2.world.food_grid) == 0
    sim2.do_step()

    assert agent.energy == 20  # only metabolism amount
    assert agent.coords == (1, 1)


def test_harvest_on_negative_cell(agent, sim2):
    # agents shouldn't harvest negative values from recovering energy cells
    sim2.world.food_grid[:] = -5
    sim2.do_step()

    assert agent.energy == 20  # only metabolism amount
    assert agent.coords == (1, 1)


def test_agent_energy_on_end_turn(agent):
    start_energy = agent.energy
    basicsim.on_end_turn(agent)
    assert agent.energy == start_energy - agent.metabolism
