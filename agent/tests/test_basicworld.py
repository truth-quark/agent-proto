import unittest
import StringIO

import basicsim
import components


def generate_food_grid():
    fd = StringIO.StringIO('12\n00\n')
    return components.Grid.from_file(fd)


def test_basicworld():
    food_grid = generate_food_grid()
    world = basicsim.BasicWorld(food_grid)


# TODO: add harvest_energy() to encapsulate the logic?
def test_basicworld_respawn():
    world = basicsim.BasicWorld(generate_food_grid())
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
