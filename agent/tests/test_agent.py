import unittest

import basicsim

# basicsim tests
def make_basic_agent():
    return basicsim.BasicAgent(_id='Test Agent', vision=2, metabolism=1, energy=23)


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
    raise NotImplementedError
