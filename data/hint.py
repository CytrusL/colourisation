import numpy as np
from skimage.draw import disk


def generate_hint(color,
                  radius_range=(2, 16),
                  std=0.0002,
                  ):
    activation = np.random.rand(*color.shape[:2])
    threshold = np.random.normal(1, std)

    interest = activation >= threshold

    hint = np.zeros((*color.shape[:2], 4), dtype=np.float32)

    for position in zip(*np.nonzero(interest)):
        radius = np.random.randint(*radius_range)
        rr, cc = disk(position, radius, shape=color.shape[:2])
        hint[rr, cc, -1] = 1.
        hint[rr, cc, :3] = color[position] / 127.5 - 1.

    return hint