import unittest
import numpy as np
import pandas as pd
from skbio.stats.composition import clr_inv as softmax
from skbio.stats.composition import closure, ilr_inv

from scipy.stats import spearmanr
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from minstrel.multimodal import (
    onehot,  cross_validation, rank_hits, Autoencoder)
from skbio.util import get_data_path
import numpy.testing as npt
import pandas.util.testing as pdt
from tensorflow import set_random_seed
import tensorflow as tf


class TestAutoencoder(unittest.TestCase):
    def setUp(self):
        seed = 0
        # build small simulation
        # res = random_multimodal(
        #     uB=-5,
        #     num_microbes=2, num_metabolites=4, num_samples=100,
        #     latent_dim=2, low=-1, high=1,
        #     microbe_total=20, metabolite_total=20,
        #     seed=seed
        # )
        res = random_multimodal(num_microbes=8, num_metabolites=8, num_samples=100,
                        latent_dim=2, sigmaQ = 3,
                        microbe_total = 1000, metabolite_total=10000
        )
        (self.microbes, self.metabolites, self.X, self.B,
         self.U, self.Ubias, self.V, self.Vbias) = res

    def test_fit(self):
        np.random.seed(1)
        tf.reset_default_graph()
        n, d1 = self.microbes.shape
        n, d2 = self.metabolites.shape
        with tf.Graph().as_default(), tf.Session() as session:
            set_random_seed(0)
            model = Autoencoder(d1, d2, beta_1=0.8, beta_2=0.9, latent_dim=2)
            model(session, coo_matrix(self.microbes.values),
                  self.metabolites.values)
            model.fit(epoch=1000)
            modelU = np.hstack(
                (np.ones((model.U.shape[0], 1)), model.Ubias, model.U))
            modelV = np.vstack(
                (model.Vbias, np.ones((1, model.V.shape[1])), model.V))

            U_ = np.hstack(
                (np.ones((self.U.shape[0], 1)), self.Ubias, self.U))
            V_ = np.vstack(
                (self.Vbias, np.ones((1, self.V.shape[1])), self.V))

            u_r, u_p = spearmanr(pdist(model.U), pdist(self.U))
            v_r, v_p = spearmanr(pdist(model.V.T), pdist(self.V.T))

            res = softmax(np.hstack((np.zeros((d1, 1)), modelU @ modelV)))
            exp = softmax(np.hstack((np.zeros((d1, 1)), U_ @ V_)))
            s_r, s_p = spearmanr(np.ravel(res), np.ravel(exp))

            self.assertGreater(u_r, 0.5)
            self.assertGreater(v_r, 0.5)
            self.assertGreater(s_r, 0.5)
            self.assertLess(u_p, 1e-4)
            self.assertLess(v_p, 1e-4)
            self.assertLess(s_p, 1e-4)


    def test_cross_validate(self):
        np.random.seed(1)
        tf.reset_default_graph()
        x = self.microbes.iloc[-5:]
        y = self.metabolites.iloc[-5:]
        n, d1 = self.microbes.shape
        n, d2 = self.metabolites.shape
        with tf.Graph().as_default(), tf.Session() as session:
            set_random_seed(0)
            model = Autoencoder(d1, d2, latent_dim=2)
            model(session, coo_matrix(self.microbes.values),
                  self.metabolites.values)
            model.fit(epoch=50)
            cv_loss = model.cross_validate(x.values, y.values)
            self.assertAlmostEqual(2.23643, cv_loss, places=5)

    def test_predict(self):
        np.random.seed(1)
        tf.reset_default_graph()
        n, d1 = self.microbes.shape
        n, d2 = self.metabolites.shape
        with tf.Graph().as_default(), tf.Session() as session:
            set_random_seed(0)
            model = Autoencoder(d1, d2, latent_dim=2)
            model(session, coo_matrix(self.microbes.values),
                  self.metabolites.values)
            model.fit(epoch=50)
            res = model.predict(self.microbes.values)

            exp = np.array(
                [[0.05642149, 0.05688434, 0.24697137, 0.42419982,
                  0.10783383, 0.07360239, 0.0106469 , 0.02343986],
                 [0.06396461, 0.10756036, 0.36204142, 0.2413285,
                  0.1236904, 0.06076086, 0.02120197, 0.01945188],
                 [0.06511185, 0.10943013, 0.36919167, 0.264807,
                  0.09644602, 0.06082298, 0.01843682, 0.01575353],
                 [0.06520348, 0.046184, 0.21416628, 0.48848722,
                  0.08473693, 0.07300372, 0.00776624, 0.02045214],
                 [0.06946033, 0.06369416, 0.23602142, 0.16361866,
                  0.33514983, 0.05531192, 0.02660176, 0.0501419],
                 [0.07815496, 0.12836207, 0.38152458, 0.14773527,
                  0.16203315, 0.04982008, 0.03154422, 0.02082566],
                 [0.07932115, 0.1363954, 0.40996947, 0.19253175,
                  0.09344088, 0.05193358, 0.02318715, 0.01322061],
                 [0.14354653, 0.10112054, 0.31972982, 0.14652646,
                  0.18642686, 0.04909901, 0.02847727, 0.0250735]]
            )
            npt.assert_allclose(exp, np.unique(res, axis=0),
                                atol=1e-1, rtol=1e-1)


class TestOnehot(unittest.TestCase):
    def setUp(self):
        seed = 0
        # build small simulation
        res = random_multimodal(
            uB=-5,
            num_microbes=2, num_metabolites=4, num_samples=10,
            latent_dim=2, low=-1, high=1,
            microbe_total=10, metabolite_total=10,
            seed=seed
        )
        (self.microbes, self.metabolites, self.X, self.B,
         self.U, self.Ubias, self.V, self.Vbias) = res

    def test_onehot(self):
        otu_hits, _ = onehot(self.microbes.values)
        np.savetxt(get_data_path('otu_hits.txt'), otu_hits)
        #exp_otu_hits = np.loadtxt(get_data_path('otu_hits.txt'))
        #npt.assert_allclose(exp_otu_hits, otu_hits)

    def test_onehot_simple(self):
        seed = 0
        # build small simulation
        res = random_multimodal(
            uB=-5,
            num_microbes=2, num_metabolites=2, num_samples=3,
            latent_dim=1, low=-1, high=1,
            microbe_total=3, metabolite_total=3,
            seed=seed
        )

        (microbes, metabolites, X, B, U, V, Ubias, Vbias) = res
        otu_hits, sample_ids = onehot(microbes.values)
        exp_otu_hits = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])

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
