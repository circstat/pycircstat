import numpy as np
from . import descriptive as descr

class AggCluster1D(object):

    """
    Performs a simple agglomerative clustering of angular data.


    :param numclust: number of clusters desired, default: 2
    """

    def __init__(self, numclust=2):
        self.numclust = numclust

    def train(self, alpha):
        """
        Finds the agglomerative clustering on the data alpha
        :param alpha: angles in radians
        :returns: data, cluster ids

        """
        assert len(alpha.shape) == 1, 'Clustering works only for 1d data'
        n = len(alpha)
        cid = np.arange(n, dtype=int)

        nu = n


        while nu > self.numclust:
            mu = np.asarray([descr.mean(alpha[cid == j]) if j in cid else np.Inf for j in range(n)])
            D = np.abs(descr.pairwise_cdiff(mu))
            idx = np.triu_indices(n,1)
            min = np.nanargmin(D[idx])
            cid[cid == cid[idx[0][min]]] = cid[idx[1][min]]
            nu -= 1


        cid2 = np.empty_like(cid)
        for i,j in enumerate(np.unique(cid)):
            cid2[cid == j] = i
        ucid = np.unique(cid2)
        self.centroids = np.asarray([descr.mean(alpha[cid2 == i]) for i in ucid])
        self.cluster_ids = ucid
        self.r = np.asarray([descr.resultant_vector_length(alpha[cid2 == i]) for i in ucid])

        return alpha, cid2

    def test(self, alpha):
        """
        Finds closests centroids to the data and returns their ids.

        :param alpha: angles in radians
        :return: data, cluster ids
        """
        D = np.abs(descr.pairwise_cdiff(self.centroids, alpha))
        idx = np.argmin(D, axis=0)
        return alpha, np.asarray([self.cluster_ids[i] for i in idx])
