import os
import itertools

from PIL import Image
import numpy as np


def snapshot_image(grid, _dir, scale=1):
    """Coroutine to generate image snapshots of the simulation."""
    if not os.path.isdir(_dir):
        raise IOError('Need a directory: {}'.format(_dir))

    try:
        count = 0
        while True:
            count += 1
            agents = (yield) # receive data
            path = os.path.join(_dir, '{:03d}.png'.format(count))
            image_dump(grid, agents, path, scale)
    except GeneratorExit:
        return

def image_dump(grid, agents, path, scale=1):
    """"Dumps a single image of the simulation to a file."""
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

    for y, x in itertools.product(*(range(i) for i in data.shape)):
        value = data[y, x]
        block = np.ones((factor, factor), dtype=data.dtype) * value
        yoff = y * factor
        xoff = x * factor
        output[yoff:yoff+factor, xoff:xoff+factor] = block

    return output
