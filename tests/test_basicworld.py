import unittest
from io import StringIO

from agent import basicsim
from agent import components


def generate_basicworld():
    fd = StringIO('12\n00\n')
    food_grid = components.Grid.from_file(fd)
    return basicsim.BasicWorld(food_grid)


def test_harvest():
    coords = (0,1)
    world = generate_basicworld()
    assert world.harvest(coords) == 2
    assert world.food_grid[coords] == -1
    world.on_end_round()
    assert world.food_grid[coords] == 0
    world.on_end_round()
    assert world.food_grid[coords] == 1
    world.on_end_round()
    assert world.food_grid[coords] == 2


def test_basicworld_respawn():
    world = generate_basicworld()
    crd = (0,1)
    assert world.food_grid[crd]
    world.food_grid[crd] = -1  # assume all energy harvested from cell
    world.on_end_round()
    assert world.food_grid[crd] == 0
    world.on_end_round()
    assert world.food_grid[crd] == 1
    world.on_end_round()
    assert world.food_grid[crd] == 2
    world.on_end_round()
    assert world.food_grid[crd] == 2
