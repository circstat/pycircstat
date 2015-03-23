import numpy as np

def simple_bootstrap(data, iterations):
    """
    Generator to perform iterations bootstrap iterations along the first axis.

    :param data: data
    :param iterations: iterations
    """
    m = data.shape[0]
    for _ in range(iterations):
        yield data[np.random.randint(0,m,m)]

def index_bootstrap(m, iterations):
    """
    Generator to perform iterations bootstrap selections among m elements. Returns indices.

    :param data: data
    :param iterations: iterations
    """

    for _ in range(iterations):
        yield np.random.randint(0,m,m, dtype=int)


def nd_bootstrap(data, iterations, axis=None, strip_tuple_if_one=True):
    """
    Bootstrap iterator for several n-dimensional data arrays.

    :param data: Iterable containing the data arrays
    :param iterations: Number of bootstrap iterations.
    :param axis: Bootstrapping is performed along this axis.
    """
    shape0 = data[0].shape
    if axis is None:
        axis = 0
        data = [d.ravel() for d in data]

    n = len(data[0].shape)
    K = len(data)
    data0 = []

    if axis is not None:
        m = data[0].shape[axis]
        to = tuple([axis]) + tuple(range(axis)) + tuple(range(axis + 1, n))
        fro = tuple(range(1, axis + 1)) + (0,) + tuple(range(axis + 1, n))
        for i in range(K):
            data0.append(data[i].transpose(to))

        for i in range(iterations):
            idx = np.random.randint(m, size=(m,))
            if len(data) == 1 and strip_tuple_if_one:
                yield (data0[0][np.ix_(idx), ...].squeeze().
                       transpose(fro).reshape(shape0))
            else:
                yield tuple(a[np.ix_(idx), ...].squeeze().
                            transpose(fro).reshape(shape0) for a in data0)
