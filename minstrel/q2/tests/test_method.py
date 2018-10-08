import biom
import qiime2
import unittest
import numpy as np
import tensorflow as tf
from minstrel.q2._method import autoencoder
from minstrel.util import random_multimodal
from skbio import OrdinationResults
from skbio.stats.composition import clr, clr_inv, centralize
import numpy.testing as npt
from scipy.stats import spearmanr


class TestAutoencoder(unittest.TestCase):

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
        self.exp = clr_inv(np.hstack((h, uv)))

    def test_fit(self):
        np.random.seed(1)
        tf.reset_default_graph()
        latent_dim = 2
        with tf.Graph().as_default(), tf.Session() as session:
            tf.set_random_seed(0)
            res = autoencoder(
                self.microbes, self.metabolites,
                epochs=1000, latent_dim=latent_dim
            )


            s_r, s_p = spearmanr(np.ravel(res), np.ravel(self.exp))

            self.assertGreater(s_r, 0.5)
            self.assertLess(s_p, 1e-2)


if __name__ == "__main__":
    unittest.main()
