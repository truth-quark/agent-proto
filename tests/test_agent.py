import unittest

import numpy as np

from agent import basicsim
from agent import components
from agent.components import NODATA


def make_basic_agent():
    return basicsim.BasicAgent(_id=0, vision=2, metabolism=1, energy=23)


def test_living():
    agent = make_basic_agent()
    assert agent.is_alive()
    assert not agent.is_dead()

    for energy in (-10, -1, 0):
        agent.energy = energy
        assert agent.is_dead()
        assert not agent.is_alive()


def test_on_end_turn():
    energy = 5
    metabolism = 2
    agent = make_basic_agent()
    agent.energy = energy
    agent.metabolism = metabolism
    agent.on_end_turn()
    assert agent.energy == energy - metabolism


def test_move_history():
    # agent should record each cell visited when visited
    s0, s1, s2 = [(1,i+1) for i in range(3)]
    agent = basicsim.BasicAgent(_id='Test Agent', vision=2, metabolism=1,
                                energy=23, coords=s0)

    assert agent.move_history == [s0]  # log initial start point
    agent.coords = s1
    assert agent.move_history == [s0, s1]
    agent.coords = s2
    assert agent.move_history == [s0, s1, s2]


def test_energy_history():
    agent = make_basic_agent()
    assert agent.harvest_history == []
    agent.energy = 25
    assert agent.harvest_history == [2]
    agent.energy -= 4
    assert agent.harvest_history == [2, -4]

    # NB: on_end_turn() shouldn't be recorded as an energy change
    agent.on_end_turn()
    assert agent.harvest_history == [2, -4]


class MoveTests(unittest.TestCase):
    """Checks the search behaviour algorithms of the BasicAgent objs."""

    def setUp(self):
        self.agent = make_basic_agent()
        self.agent.coords = (2,3)

    def test_next_move(self):
        view = np.array([[1,2,3], [0,0,0], [0,0,0]])
        assert self.agent.next_move(view) == (1,4)  # cell with 3 in it

    def test_next_move_nothing(self):
        view = np.array([[0,0,0], [0,0,0], [0,0,0]])
        assert self.agent.next_move(view) == (1,3)  # for id=0

        self.agent.id = 5
        assert self.agent.next_move(view) == (3,2)

    def test_next_move_with_agents(self):
        # get a view and block off cells with agents / special value
        adj_agents = {4:object()}
        view = np.array([[0, 0, 0], [-1, -1, -1], [2, 3, 0]])
        assert self.agent.next_move(view, adj_agents) == (3,2)

    def test_search_direction(self):
        # agent's search direction should be based on modulo of id 0=N, 2=E
        for i, crd in enumerate(components.adjacent_coords(self.agent.coords)):
            self.agent.id = i
            assert self.agent._search_direction(adj_energy={}) == crd

    def test_search_direction_nodata(self):
        self.agent.id = 5  # default is SW direction of travel
        adj_energy = {5:NODATA, 6:NODATA}
        assert self.agent._search_direction(adj_energy) == (1,2)

    def test_search_direction_multiple_turns(self):
        # check if some agents get stuck/go back & forth uselessly between 2 cells
        self.agent.id = 3  # force SE travel
        start = (0,0)
        self.agent.coords = start
        view = np.zeros((3,3), dtype=np.int8)

        for i in range(5):
            exp = tuple(i + n + 1 for n in start)
            act = self.agent.next_move(view, {})
            assert act == exp
            self.agent.coords = exp  # move the agent

        # ensure agent hasn't re-visited a prev square
        assert len(set(self.agent.move_history)) == 6  # start + 5 moves
