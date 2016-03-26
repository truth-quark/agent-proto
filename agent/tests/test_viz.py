import numpy as np
import numpy.testing as npt

import viz


def test_monochrome_remap():
    agents = []
    grid = np.array([-128,0,1,2,3,4])
    res = viz.monochrome_remap(grid, agents)
    exp = np.array([255,255,200, 150,100,50])
    npt.assert_array_equal(res, exp)


def test_upscale():
    data = np.arange(4, dtype=np.int8).reshape(2,2)
    exp = np.array([[0,0,0,1,1,1],
                    [0,0,0,1,1,1],
                    [0,0,0,1,1,1],
                    [2,2,2,3,3,3],
                    [2,2,2,3,3,3],
                    [2,2,2,3,3,3], ])

    res = viz.upscale(data, 3)
    npt.assert_equal(exp, res)
