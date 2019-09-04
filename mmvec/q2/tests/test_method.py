import biom
import unittest
import numpy as np
import tensorflow as tf
from mmvec.q2._method import mmvec
from mmvec.util import random_multimodal
from skbio.stats.composition import clr_inv
from scipy.stats import spearmanr
import numpy.testing as npt


class TestMMvec(unittest.TestCase):

    def setUp(self):
        res = random_multimodal(
            num_microbes=8, num_metabolites=8, num_samples=150,
            latent_dim=2, sigmaQ=2,
            microbe_total=1000, metabolite_total=10000, seed=1
        )
        (self.microbes, self.metabolites, self.X, self.B,
         self.U, self.Ubias, self.V, self.Vbias) = res
        n, d1 = self.microbes.shape
        n, d2 = self.metabolites.shape

        self.microbes = biom.Table(self.microbes.values.T,
                                   self.microbes.columns,
                                   self.microbes.index)
        self.metabolites = biom.Table(self.metabolites.values.T,
                                      self.metabolites.columns,
                                      self.metabolites.index)
        U_ = np.hstack(
            (np.ones((self.U.shape[0], 1)), self.Ubias, self.U))
        V_ = np.vstack(
            (self.Vbias, np.ones((1, self.V.shape[1])), self.V))

        uv = U_ @ V_
        h = np.zeros((d1, 1))
        self.exp_ranks = clr_inv(np.hstack((h, uv)))

    def test_fit(self):
        np.random.seed(1)
        tf.reset_default_graph()
        latent_dim = 2
        tf.set_random_seed(0)
        res_ranks, res_biplot = mmvec(
            self.microbes, self.metabolites,
            epochs=1000, latent_dim=latent_dim
        )
        s_r, s_p = spearmanr(np.ravel(res_ranks), np.ravel(self.exp_ranks))

        self.assertGreater(s_r, 0.5)
        self.assertLess(s_p, 1e-2)

        # make sure the biplot is of the correct dimensions
        npt.assert_allclose(
            res_biplot.samples.shape,
            np.array([self.microbes.shape[0], latent_dim]))
        npt.assert_allclose(
            res_biplot.features.shape,
            np.array([self.metabolites.shape[0], latent_dim]))


if __name__ == "__main__":
    unittest.main()
