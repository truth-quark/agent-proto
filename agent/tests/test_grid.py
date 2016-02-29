import unittest
import StringIO

import numpy as np
import numpy.testing as npt

import components
from components import NODATA


def generate_test_grid():
    fd = StringIO.StringIO('1,2,3\n7,8,9\n11,12,13\n18,19,20\n')
    return components.Grid.from_file(fd)


def test_create_grid():
    ncols, nrows = 3, 9
    grid = components.Grid(nrows, ncols)
    assert grid.nrows == nrows
    assert grid.ncols == ncols
    assert grid.border_size == 1


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

    exp = np.array([1,2,3,7,8,9,11,12,13]).reshape((3,3))
    res = grid.view(1,1,size=1)
    npt.assert_equal(exp, res)

    exp = np.array([NODATA,NODATA,NODATA,NODATA,1,2,NODATA,7,8]).reshape((3,3))
    res = grid.view(0,0,size=1)
    npt.assert_equal(exp, res)
