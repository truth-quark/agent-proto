import unittest
import StringIO

import components


def generate_test_grid():
    fd = StringIO.StringIO('1,2,3\n7,8,9\n')
    return components.Grid.from_file(fd)


def test_create_grid():
    ncols, nrows = 3, 9
    grid = components.Grid(nrows, ncols)
    assert grid.nrows == nrows
    assert grid.ncols == ncols
    assert grid.border_size == 1


def test_read_grid():
    grid = generate_test_grid()
    assert grid.nrows == 2
    assert grid.ncols == 3
    assert grid[0,0] == 1
    assert grid[1,1] == 8
    assert (grid[0,:] == [1,2,3]).all()
    assert (grid[1,:] == [7,8,9]).all()


def test_write_grid():
    grid = generate_test_grid()
    assert grid[0,0] == 1
    value = 4
    grid[0,0] = value
    assert grid[0,0] == value


#def test_view():
    # for coords, get a view square
