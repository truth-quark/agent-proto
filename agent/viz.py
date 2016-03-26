import itertools

import Image
import numpy as np


# TODO: refactor into coroutine to generate images each time round data is sent
# use it to keep file paths/numbering etc out of the simulation code
def image_dump(grid, agents, path, scale=1):
    raw = grid._grid[1:-1, 1:-1]  # TODO: pass in without borders?
    mono = monochrome_remap(raw, agents)
    final = upscale(mono, scale) if scale > 1 else mono
    image = Image.fromarray(final)
    image.save(path)


def monochrome_remap(raw, agents):
    """Quick & dirty function to map BasicSim world to 0-255 colour array."""
    data = np.zeros(raw.shape, dtype=np.uint8)
    data[raw <= 0] = 255
    data[raw == 1] = 200
    data[raw == 2] = 150
    data[raw == 3] = 100
    data[raw == 4] = 50

    # represent agents as black dot
    for a in agents:
        data[a.coords] = 0
    return data


def upscale(data, factor):
    """Scale an array up in size."""
    assert factor
    ys, xs = [i*factor for i in data.shape]
    output = np.zeros((ys, xs), dtype=data.dtype)

    for y, x in itertools.product(*(xrange(i) for i in data.shape)):
        value = data[y, x]
        block = np.ones((factor, factor), dtype=data.dtype) * value
        yoff = y * factor
        xoff = x * factor
        output[yoff:yoff+factor, xoff:xoff+factor] = block

    return output
