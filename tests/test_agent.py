"""
Tests for autonomous Agents in simulations.
"""

from agent import basicsim
import pytest


@pytest.fixture
def base_agent(_id=0, vision=2, metabolism=1, energy=23):
    return basicsim.Agent(_id, vision, metabolism, energy)


def test_living_agent(base_agent):
    assert base_agent.is_alive()
    assert not base_agent.is_dead()

    for energy in (100, 10, 1):
        base_agent.energy = energy
        assert base_agent.is_alive()


def test_dead_agent(base_agent):
    for energy in (-10, -1, 0):
        base_agent.energy = energy
        assert base_agent.is_dead()
        assert not base_agent.is_alive()
