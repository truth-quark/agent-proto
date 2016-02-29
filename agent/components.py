import csv
import numpy as np


DEFAULT_BORDER = 1


class Grid(object):

    def __init__(self, nrows, ncols, border=DEFAULT_BORDER):
        if border !=1:
            raise NotImplementedError

        self._border_size = border

        # make grid with border cells
        x_size = ncols + (2 * self._border_size)
        y_size = nrows + (2 * self._border_size)
        self._grid = np.zeros((y_size, x_size), dtype=np.int8)

    @classmethod
    def from_file(cls, fd, border=DEFAULT_BORDER):
        """ TODO """
        grid_values = []
        for row in csv.reader(fd, delimiter=','):
            values = [int(i) for i in row]
            grid_values.append(values)

        nrows, ncols = len(grid_values), len(grid_values[0])
        grid = Grid(nrows, ncols, border)

        for i, values in enumerate(grid_values):
            grid[i,:] = values

        return grid

    def _slice_offset(self, _slice, values=None):
        start = self._border_size
        if _slice.start:
            start += self._border_size

        if _slice.stop:
            raise NotImplementedError
        else:
            # coords are likely to be [start:]
            if values:
                stop = start + len(values)
            else:
                # no slice stop implies continuing to the end
                stop = -self._border_size

        return slice(start, stop, _slice.step)

    def _offset_coord(self, coords, values=None):
        """TODO: coord can be a slice or a cell"""
        if hasattr(coords, '__len__'):  # TODO: test is a container
            # treat as a tuple of coordinates
            # TODO: replace isinstance
            f = lambda(e): self._slice_offset(e) if isinstance(e, slice) else e + self._border_size
            offset = tuple(f(e) for e in coords)
        elif isinstance(coords, slice):
            offset = self._slice_offset(coords)
        elif isinstance(coords, int):
            offset = coords + self._border_size
        else:
            raise NotImplementedError(coords)
        return offset

    def __str__(self):
        """TODO: only show the data component of the grid"""
        s = slice(self._border_size, -self._border_size)
        return str(self._grid[s, s])

    def __getitem__(self, coords):
        return self._grid[self._offset_coord(coords)]

    def __setitem__(self, coords, value):
        self._grid[self._offset_coord(coords, value)] = value

    @property
    def border_size(self):
        return self._border_size

    @property
    def nrows(self):
        return self._grid.shape[0] - (2 * self._border_size)

    @property
    def ncols(self):
        return self._grid.shape[1] - (2 * self._border_size)

    def view(self, y, x, size):
        """TODO: coords needs to be tuple of ints"""
        y, x = [self._offset_coord(i) for i in (y,x)]
        return self._grid[y - size:y + size + 1, x - size:x + size + 1]
