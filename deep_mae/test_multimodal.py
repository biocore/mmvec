import unittest
import numpy as np
import pandas as pd
from skbio.stats.composition import clr_inv as softmax
from skbio.stats.composition import closure
from sklearn.utils import check_random_state
from scipy.stats import spearmanr
from scipy.sparse import coo_matrix
from deep_mae.multimodal import (
    onehot,  cross_validation, rank_hits, Autoencoder)
from skbio.util import get_data_path
import numpy.testing as npt
import pandas.util.testing as pdt
from tensorflow import set_random_seed
import tensorflow as tf


def random_multimodal(num_microbes=20, num_metabolites=100, num_samples=100,
                      num_latent=5, low=-1, high=1,
                      microbe_total=10, metabolite_total=100,
                      uB=0, sigmaB=2, sigmaQ=0.1,
                      uU=0, sigmaU=1, uV=0, sigmaV=1,
                      seed=0):
    """ Generates two random tables

    Parameters
    ----------
    num_microbes : int
       Number of microbial species to simulate
    num_metabolites : int
       Number of molecules to simulate
    num_samples : int
       Number of samples to generate
    num_latent_microbes :
       Number of latent microbial dimensions
    num_latent_metabolites
       Number of latent metabolite dimensions
    num_latent_shared
       Number of dimensions in shared representation
    low : float
       Lower bound of gradient
    high : float
       Upper bound of gradient
    microbe_total : int
       Total number of microbial species
    metabolite_total : int
       Total number of metabolite species
    uB : float
       Mean of regression coefficient distribution
    sigmaB : float
       Standard deviation of regression coefficient distribution
    sigmaQ : float
       Standard deviation of error distribution
    uU : float
       Mean of microbial input projection coefficient distribution
    sigmaU : float
       Standard deviation of microbial input projection
       coefficient distribution
    uV : float
       Mean of metabolite input projection coefficient distribution
    sigmaU : float
       Standard deviation of metabolite input projection
       coefficient distribution
    seed : float
       Random seed

    Returns
    -------
    microbe_counts : pd.DataFrame
       Count table of microbial counts
    metabolite_counts : pd.DataFrame
       Count table of metabolite counts
    """
    state = check_random_state(seed)
    # only have two coefficients
    beta = state.normal(uB, sigmaB, size=(2, num_microbes))

    X = np.vstack((np.ones(num_samples),
                   np.linspace(low, high, num_samples))).T

    microbes = softmax(state.normal(X @ beta, sigmaQ))

    U = state.normal(
        uU, sigmaU, size=(num_microbes, num_latent))
    V = state.normal(
        uV, sigmaV, size=(num_latent, num_metabolites))

    probs = softmax(U @ V)
    microbe_counts = np.zeros((num_samples, num_microbes))
    metabolite_counts = np.zeros((num_samples, num_metabolites))
    n1 = microbe_total
    n2 = metabolite_total // n1
    for n in range(num_samples):
        otu = state.multinomial(n1, microbes[n, :])
        for i in range(num_microbes):
            ms = state.multinomial(otu[i] * n2, probs[i, :])
            metabolite_counts[n, :] += ms
        microbe_counts[n, :] += otu

    otu_ids = ['OTU_%d' % d for d in range(microbe_counts.shape[1])]
    ms_ids = ['metabolite_%d' % d for d in range(metabolite_counts.shape[1])]
    sample_ids = ['sample_%d' % d for d in range(metabolite_counts.shape[0])]

    microbe_counts = pd.DataFrame(
        microbe_counts, index=sample_ids, columns=otu_ids)
    metabolite_counts = pd.DataFrame(
        metabolite_counts, index=sample_ids, columns=ms_ids)

    return microbe_counts, metabolite_counts, X, beta, U, V


class TestAutoencoder(unittest.TestCase):
    def setUp(self):
        seed = 0
        # build small simulation
        res = random_multimodal(
            uB=-5,
            num_microbes=2, num_metabolites=4, num_samples=100,
            num_latent=2, low=-1, high=1,
            microbe_total=10, metabolite_total=10,
            seed=seed
        )
        self.microbes, self.metabolites, self.X, self.B, self.U, self.V = res

    def test_fit(self):
        np.random.seed(1)
        tf.reset_default_graph()
        n, d1 = self.microbes.shape

        n, d2 = self.metabolites.shape
        with tf.Graph().as_default(), tf.Session() as session:
            set_random_seed(0)
            model = Autoencoder(d1, d2, dropout_rate=10e-6, latent_dim=2)
            model(session, coo_matrix(self.microbes.values), self.metabolites.values)
            model.fit(epoch=20)
            res = softmax(np.hstack((np.zeros((d1, 1)), model.U @ model.V)))

            exp = softmax(self.U @ self.V)
            npt.assert_allclose(res, exp,
                                rtol=1e-1, atol=1e-1)

    def test_cross_validate(self):
        np.random.seed(1)
        tf.reset_default_graph()
        x = self.microbes.iloc[-5:]
        y = self.metabolites.iloc[-5:]
        n, d1 = self.microbes.shape
        n, d2 = self.metabolites.shape
        with tf.Graph().as_default(), tf.Session() as session:
            set_random_seed(0)
            model = Autoencoder(d1, d2, dropout_rate=10e-6, latent_dim=2)
            model(session, coo_matrix(self.microbes.values), self.metabolites.values)
            model.fit(epoch=50)
            cv_loss = model.cross_validate(x.values, y.values)
            self.assertAlmostEqual(2.714532, cv_loss, places=5)

    def test_predict(self):
        np.random.seed(1)
        tf.reset_default_graph()
        n, d1 = self.microbes.shape
        n, d2 = self.metabolites.shape
        with tf.Graph().as_default(), tf.Session() as session:
            set_random_seed(0)
            model = Autoencoder(d1, d2, dropout_rate=10e-6, latent_dim=2)
            model(session, coo_matrix(self.microbes.values), self.metabolites.values)
            model.fit(epoch=50)
            res = model.predict(self.microbes.values)
            exp = np.array([[0.030375, 0.168372, 0.619572, 0.181681],
                            [0.069957, 0.180757, 0.564098, 0.185187]])
            npt.assert_allclose(exp, np.unique(res, axis=0), atol=1e-1, rtol=1e-1)


class TestOnehot(unittest.TestCase):
    def setUp(self):
        seed = 0
        # build small simulation
        res = random_multimodal(
            uB=-5,
            num_microbes=2, num_metabolites=4, num_samples=10,
            num_latent=2, low=-1, high=1,
            microbe_total=10, metabolite_total=10,
            seed=seed
        )
        self.microbes, self.metabolites, self.X, self.B, self.U, self.V = res

    def test_onehot(self):
        otu_hits, _ = onehot(self.microbes.values)

        exp_otu_hits = np.loadtxt(get_data_path('otu_hits.txt'))

        npt.assert_allclose(exp_otu_hits, otu_hits)

    def test_onehot_simple(self):
        seed = 0
        # build small simulation
        res = random_multimodal(
            uB=-5,
            num_microbes=2, num_metabolites=2, num_samples=3,
            num_latent=1, low=-1, high=1,
            microbe_total=3, metabolite_total=3,
            seed=seed
        )
        microbes, metabolites, X, B, U, V = res
        otu_hits, sample_ids = onehot(microbes.values)
        exp_otu_hits = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])

        exp_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        npt.assert_allclose(exp_ids, sample_ids)
        npt.assert_allclose(exp_otu_hits, otu_hits)


class TestRankHits(unittest.TestCase):

    def test_rank_hits(self):
        ranks = pd.DataFrame(
            [
                [1., 4., 1., 5., 7.],
                [2., 6., 9., 2., 8.],
                [2., 2., 6., 8., 4.]
            ],
            index=['OTU_1', 'OTU_2', 'OTU_3'],
            columns=['MS_1', 'MS_2', 'MS_3', 'MS_4', 'MS_5']
        )
        res = rank_hits(ranks, k=2)
        exp = pd.DataFrame(
            [
                ['OTU_1', 5., 'MS_4'],
                ['OTU_2', 8., 'MS_5'],
                ['OTU_3', 6., 'MS_3'],
                ['OTU_1', 7., 'MS_5'],
                ['OTU_2', 9., 'MS_3'],
                ['OTU_3', 8., 'MS_4']
            ], columns=['src', 'rank', 'dest'],
        )

        pdt.assert_frame_equal(res, exp)


if __name__ == "__main__":
    unittest.main()
