import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:

    def __init__(self, k=2, tol=1e-4):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.k = k
        self.centroids = np.empty((self.k,))
        self.tol = tol

    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """

        # if there are more centroids than samples simply set the centroids list as the samples list
        if X.shape[0] <= self.k:
            self.centroids = X.to_numpy()
            return

        # define arbitrarily k centroids
        self.centroids = X.sample(n=self.k).to_numpy()

        previous_error = self.tol + 1
        new_error = 0
        # loop until convergence
        while abs(previous_error - new_error) > self.tol:
            # step 1 : find the closest centroid for each point

            # array which for each sample contains the index of the closest centroid, in the centroid list of the class
            centroids_assignations = np.zeros(X.shape[0], dtype=int)

            # new_error becomes previous error
            previous_error = new_error
            new_error = 0

            for sample_index, row in X.iterrows():
                min_dist = euclidean_distance(row, self.centroids[0])
                closest_centroid = 0
                for id_centroid, centroid in enumerate(self.centroids[1:], start=1):
                    dist = euclidean_distance(row, centroid)
                    if dist < min_dist:
                        min_dist = dist
                        closest_centroid = id_centroid
                new_error += min_dist
                centroids_assignations[sample_index] = closest_centroid

            # step 2 : for each cluster compute the centroid which is the mean of all points that have the same centroid

            # k centroids, each of them has the same dimension as the samples
            new_centroids = np.zeros((self.k, X.shape[1]))
            cluster_point_count = np.zeros((self.k, 1))
            for index, coordinates in X.iterrows():
                point_centroid = centroids_assignations[index]
                new_centroids[point_centroid] += coordinates
                cluster_point_count[point_centroid] += 1

            self.centroids = new_centroids / cluster_point_count


    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        X = X.apply(
            lambda row:
            np.argmin(
                np.array(
                    [euclidean_distance(row, centroid) for centroid in self.centroids])),
            axis=1)
        return X.to_numpy()

    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids


# --- Some utility functions 

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)


def cross_euclidean_distance(x, y=None):
    """
    
    
    """
    y = x if y is None else y
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]

    distortion = 0.0
    clusters = np.unique(z)
    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()

    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]

    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)

    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)

    return np.mean((b - a) / np.maximum(a, b))
