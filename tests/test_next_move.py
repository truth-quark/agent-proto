"""
Tests for next move/search behaviour algorithms.
"""

from agent.basicsim import Agent, next_move, search_direction
from agent.components import NODATA
from agent import components

import numpy as np

import pytest


@pytest.fixture
def agent():
    return Agent("Agent 0", 2, 1, 15, coords=(2, 3))


def test_next_move(agent):
    view = np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0]])
    assert next_move(agent, view) == (1, 4)  # cell with 3 in it


@pytest.fixture
def zero_energy():
    return np.zeros((3, 3))


def test_next_move_nothing_default_north(agent, zero_energy):
    agent.id = 0
    assert next_move(agent, zero_energy) == (1, 3)  # for id=0


def test_next_move_nothing(agent, zero_energy):
    agent.id = 5
    assert next_move(agent, zero_energy) == (3, 2)


def test_next_move_with_agents(agent):
    # get a view and block off cells with agents / special value
    adj_agents = {4: object()}  # block cell with energy=3
    view = np.array([[0, 0, 0], [-1, -1, -1], [2, 3, 0]])
    assert next_move(agent, view, adj_agents) == (3, 2)


def test_search_direction(agent, zero_energy):
    # agent's search direction should be based on modulo of id 0=N, 2=E
    for i, crd in enumerate(components.adjacent_coords(agent.coords)):
        search_dir = search_direction(zero_energy, default_dir=i)
        assert 0 <= search_dir <= 7
        assert search_dir == i  # returns default dir as it's not the NODATA border


def test_search_direction_nodata(agent, zero_energy):
    # Ensure search direction is next best in clockwise compass direction
    direction = 5  # default is SW direction of travel
    zero_energy[1:3, 0] = NODATA
    assert search_direction(zero_energy, direction) == 7  # Head NW


@pytest.mark.skip
def test_search_direction_multiple_turns(agent):
    # TODO: check if some agents get stuck/go back & forth uselessly between 2 cells
    #       requires reimplementation of agent history
    direction = 3  # force SE travel
    start = (0, 0)
    agent.coords = start
    view = np.zeros((3, 3), dtype=np.int8)

    for i in range(5):
        exp = tuple(i + n + 1 for n in start)
        act = search_direction(view, direction)
        assert act == exp
        agent.coords = exp  # move the agent

    # ensure agent hasn't re-visited a prev square
    assert len(set(agent.move_history)) == 6  # start + 5 moves
