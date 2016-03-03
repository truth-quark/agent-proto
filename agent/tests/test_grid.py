import unittest
import StringIO
import copy

import numpy as np
import numpy.testing as npt

import components
from components import NODATA


def test_adjacent_coords():
    exp = [(0,1), (0,2), (1,2), (2,2), (2,1), (2,0), (1,0), (0,0)]
    result = components.adjacent_coords((1,1))


def generate_test_grid():
    fd = StringIO.StringIO('123\n789\n234\n890\n')
    return components.Grid.from_file(fd)


def test_create_grid():
    ncols, nrows = 3, 9
    grid = components.Grid(nrows, ncols)
    assert grid.nrows == nrows
    assert grid.ncols == ncols
    assert grid.border_size == 1
    assert len(grid) == 9


def test_read_grid():
    grid = generate_test_grid()
    assert grid.nrows == 4
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


def test_view():
    # for coords, get a view square
    grid = generate_test_grid()
    assert grid._grid.dtype == np.int8

    exp = np.array([1,2,3,7,8,9,2,3,4]).reshape((3,3))
    res = grid.view(1,1,size=1)
    npt.assert_equal(exp, res)

    exp = np.array([NODATA,NODATA,NODATA,NODATA,1,2,NODATA,7,8]).reshape((3,3))
    res = grid.view(0,0,size=1)
    npt.assert_equal(exp, res)


def test_grid_iter():
    grid = generate_test_grid()
    exp = [[1,2,3], [7,8,9], [2,3,4], [8,9,0]]

    for row, e in zip(grid, exp):
        npt.assert_equal(row, e)


def test_copy_grid():
    # safety check: ensure underlying grid is copied to new array
    grid = generate_test_grid()
    cgrid = copy.copy(grid)
    npt.assert_equal(grid._grid, cgrid._grid)

    # ensure separate data grids underneath
    cgrid[(0,0)] += 15
    npt.assert_equal(grid._grid, cgrid._grid) is False
