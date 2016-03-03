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


DATA = '''21000
11000
00000
00000
'''
