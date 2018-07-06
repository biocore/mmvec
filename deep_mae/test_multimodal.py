import unittest
import numpy as np
import pandas as pd
from skbio.stats.composition import clr_inv as softmax
from skbio.stats.composition import closure
from sklearn.utils import check_random_state
from scipy.stats import spearmanr
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
    """
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

            model = Autoencoder(session, n, d1, d2, dropout_rate=10e-6, latent_dim=2)
            model.fit(self.microbes.values, self.metabolites.values, epoch=50)
            res = softmax(np.hstack((np.zeros((d1, 1)), model.U @ model.V)))
            exp = softmax(self.U @ self.V)
            npt.assert_allclose(res, exp,
                                rtol=1e-1, atol=1e-1)

    def test_cross_validate(self):
        pass


class TestMultimodal(unittest.TestCase):
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
        otu_hits, ms_hits, _ = onehot(self.microbes.values,
                                      closure(self.metabolites.values))
        npt.assert_allclose(ms_hits, closure(ms_hits))

        exp_ms_hits = np.loadtxt(get_data_path('ms_hits.txt'))
        exp_otu_hits = np.loadtxt(get_data_path('otu_hits.txt'))
        npt.assert_allclose(exp_ms_hits, ms_hits)
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
        otu_hits, ms_hits, _ = onehot(microbes.values,
                                      closure(metabolites.values))
        exp_otu_hits = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])
        exp_ms_hits = np.array(
            [[1., 0.],
             [1., 0.],
             [1., 0.],
             [0.33333333, 0.66666667],
             [0.33333333, 0.66666667],
             [0.33333333, 0.66666667],
             [0.66666667, 0.33333333],
             [0.66666667, 0.33333333],
             [0.66666667, 0.33333333]]
        )
        npt.assert_allclose(exp_otu_hits, otu_hits)
        npt.assert_allclose(exp_ms_hits, ms_hits)

class TestMultimodalModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        set_random_seed(0)
        seed = 0
        # build small simulation
        self.res = random_multimodal(
            uB=-5,
            num_microbes=2, num_metabolites=4, num_samples=500,
            num_latent=2, low=-1, high=1,
            microbe_total=10, metabolite_total=10,
            seed=seed
        )

    def test_build_model(self):
        microbes, metabolites, X, B, U, V = self.res

        epochs = 10
        batch_size = 100
        model = build_model(microbes, metabolites,
                            latent_dim=2, dropout_rate=0., lam=0,
                            beta_1=0.999, beta_2=0.9999, clipnorm=10.)
        otu_hits, ms_hits, _ = onehot(microbes.values,
                                      closure(metabolites.values))
        model.fit(
            {
                'otu_input': otu_hits,
            },
            {
                'ms_output': ms_hits
            },
            verbose=0,
            #callbacks=[tbCallBack],
            epochs=epochs, batch_size=batch_size)
        weights = model.get_weights()
        rU, rV = weights[0], weights[1]

        r, p = spearmanr(np.ravel(rU @ rV), np.ravel(U @ V))
        self.assertGreater(r, 0.15)

    def test_cross_validation(self):

        res = random_multimodal(
            uB=-5,
            num_microbes=2, num_metabolites=5, num_samples=500,
            num_latent=2, low=-1, high=1,
            microbe_total=10, metabolite_total=10,
            seed=0
        )
        microbes, metabolites, X, B, U, V = res

        epochs = 10
        batch_size = 100
        model = build_model(microbes, metabolites,
                            latent_dim=2, dropout_rate=0., lam=0,
                            beta_1=0.999, beta_2=0.9999, clipnorm=10.)
        otu_hits, ms_hits, sample_ids = onehot(microbes.values,
                                      closure(metabolites.values))
        model.fit(
            {
                'otu_input': otu_hits,
            },
            {
                'ms_output': ms_hits
            },
            verbose=0,
            #callbacks=[tbCallBack],
            epochs=epochs, batch_size=batch_size)
        sids = list(sample_ids[:50]) + list(sample_ids[-50:])
        params, stats = cross_validation(
            model, microbes.iloc[sids], metabolites.iloc[sids], top_N=4)


        exp_params = pd.Series(
            [900.000000, 900.000000, 100.000000, 3100.000000,
             0.688889, 0.3007, 0.775000, 0.620000],
            index=['FN', 'FP', 'TN', 'TP', 'f1_score',
                   'meanRK', 'precision', 'recall'])

        tuples = [('OTU_0', 'sample_0'),
                  ('OTU_0', 'sample_1'),
                  ('OTU_0', 'sample_2'),
                  ('OTU_0', 'sample_3'),
                  ('OTU_0', 'sample_4'),
                  ('OTU_0', 'sample_495'),
                  ('OTU_0', 'sample_496'),
                  ('OTU_0', 'sample_497'),
                  ('OTU_0', 'sample_498'),
                  ('OTU_0', 'sample_499'),
                  ('OTU_1', 'sample_4'),
                  ('OTU_1', 'sample_495'),
                  ('OTU_1', 'sample_496'),
                  ('OTU_1', 'sample_497'),
                  ('OTU_1', 'sample_498'),
                  ('OTU_1', 'sample_499')]
        index = pd.MultiIndex.from_tuples(tuples, names=['OTU', 'sample_ids'])

        exp_stats = pd.DataFrame(
            [[1, 1, 0, 3, 0.683772, -0.316228],
             [1, 1, 0, 3, 0.367544, 0.632456],
             [1, 1, 0, 3, 0.000000, 1.000000],
             [1, 1, 0, 3, 0.789181, 0.210819],
             [0, 0, 1, 4, 0.367544, -0.632456],
             [1, 1, 0, 3, 0.741801, 0.258199],
             [1, 1, 0, 3, 0.741801, 0.258199],
             [1, 1, 0, 3, 0.400000, 0.600000],
             [1, 1, 0, 3, 0.262135, 0.737865],
             [1, 1, 0, 3, 0.741801, 0.258199],
             [0, 0, 1, 4, 0.367544, -0.632456],
             [1, 1, 0, 3, 0.741801, 0.258199],
             [1, 1, 0, 3, 0.741801, 0.258199],
             [1, 1, 0, 3, 0.400000, 0.600000],
             [1, 1, 0, 3, 0.262135, 0.737865],
             [1, 1, 0, 3, 0.741801, 0.258199]],
            index=index,
            columns=['FN', 'FP', 'TN', 'TP', 'pvalue', 'spearman_r']
        )

        npt.assert_allclose(params.values, exp_params.values,
                            atol=1e-4, rtol=1e-4)

        npt.assert_allclose(stats.values, exp_stats.values,
                            atol=1e-4, rtol=1e-4)

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
