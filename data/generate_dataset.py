import itertools
import numpy as np

# 3 colors, 3 shapes, 2 sizes, 3 position y, 3 position x
SHAPES_ATTRIBUTES = [3, 3, 2, 3, 3]


def _one_hot(a):
    ncols = a.max() + 1
    out = np.zeros((a.size, ncols), dtype=np.uint8)
    out[np.arange(a.size), a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out


def generate_dataset(atttribute_vector: list = SHAPES_ATTRIBUTES):
    """
    Generates a dataset based on the vector of attributes passed
    """

    possible_values = []
    for attribute in atttribute_vector:
        possible_values.append(list(range(attribute)))

    # get all possible values
    all_possible_values = np.array(list(itertools.product(*possible_values)))

    # one hot encode
    one_hot_derivations = _one_hot(all_possible_values).reshape(
        all_possible_values.shape[0], -1
    )

    # compress one hot encoding (remove all 0 only columns)
    remove_idx = np.argwhere(np.all(one_hot_derivations[..., :] == 0, axis=0))
    one_hot_derivations = np.delete(one_hot_derivations, remove_idx, axis=1)

    # randomly samply from possible combinations
    # idxs = np.random.choice(range(len(one_hot_derivations)), size, replace=True)
    return one_hot_derivations
