import itertools
import numpy as np
import scipy.stats
import warnings
from typing import Sequence, Callable, Optional


def rsa(
    space_x: Sequence[Sequence],
    space_y: Sequence[Sequence],
    distance_function_x: Callable[[Sequence, Sequence], float],
    distance_function_y: Callable[[Sequence, Sequence], float],
    number_of_samples: Optional[int] = None,
) -> float:
    """
    Calculates RSA using all possible pair combinations in both space given distance functions
    Args:
        space_x: representations in space x
        space_y: representations in space y (note these represensations must match in the 1st dimension)
        distance_function_x: distance function used to measure space x
        distance_function_y: distance function used to measure space y
        number_of_samples (int, optional): if passed, uses a random number of pairs instead of all combinations
    Returns:
        topographical_similarity (float): correlation between similarity of pairs in both spaces
    """
    assert len(space_x) == len(space_y)

    N = len(space_x)

    # if no number of sample is passed
    # using all possible pair combinations in space
    if number_of_samples is None:
        combinations = list(itertools.combinations(range(N), 2))
    else:
        combinations = np.random.choice(
            np.arange(N), size=(number_of_samples, 2), replace=True
        )

    sim_x = np.zeros(len(combinations))
    sim_y = np.zeros(len(combinations))

    for i, c in enumerate(combinations):
        s1, s2 = c[0], c[1]

        sim_x[i] = distance_function_x(space_x[s1], space_x[s2])
        sim_y[i] = distance_function_y(space_y[s1], space_y[s2])

    # check if standard deviation is not 0
    if sim_x.std() == 0.0 or sim_y.std() == 0.0:
        warnings.warn("Standard deviation of a space is 0 given distance function")
        rho = 0.0
    else:
        rho = scipy.stats.pearsonr(sim_x, sim_y)[0]

    return rho


if __name__ == "__main__":
    from scipy import spatial

    A = np.random.randint(100, size=(100, 10))
    B = np.random.randint(100, size=(100, 15))
    distance_A = spatial.distance.cosine
    distance_B = distance_A
    r = rsa(A, B, distance_A, distance_B)
    print(r)
    r = rsa(A, B, distance_A, distance_B, number_of_samples=5000)
    print(r)
