import unittest
import numpy as np
import torch
from mmvec.q2._method import mmvec
from mmvec.util import random_multimodal
from scipy.stats import spearmanr
import numpy.testing as npt


class TestMMvec(unittest.TestCase):

    def setUp(self):
        self.latent_dim = 2
        self.num_microbes = 15
        self.num_metabolites = 16

        res = random_multimodal(
            num_microbes=self.num_microbes,
            num_metabolites=self.num_metabolites, num_samples=220,
            latent_dim=self.latent_dim, sigmaQ=0.1, sigmaU=3, sigmaV=3,
            microbe_total=100, metabolite_total=1000, seed=1
        )
        (self.microbes, self.metabolites, self.X, self.B,
         self.U, self.Ubias, self.V, self.Vbias) = res
        d1, n = self.microbes.shape
        d2, n = self.metabolites.shape

        U_ = np.hstack(
            (np.ones((self.U.shape[0], 1)), self.Ubias, self.U))
        V_ = np.vstack(
            (self.Vbias, np.ones((1, self.V.shape[1])), self.V))

        self.exp_ranks = np.hstack((np.zeros((d1, 1)), U_ @ V_))

    def test_fit(self):
        np.random.seed(1)
        torch.manual_seed(1)
        latent_dim = 2

        res_ranks, res_biplot = mmvec(
            self.microbes, self.metabolites,
            batch_size=50, epochs=3, learning_rate=0.1,
            latent_dim=latent_dim, min_feature_count=1,
            num_testing_examples=10,
        )
        s_r, s_p = spearmanr(np.ravel(res_ranks), np.ravel(self.exp_ranks))
        self.assertGreater(s_r, 0.1)
        self.assertLess(s_p, 0.1)

        # make sure the biplot is of the correct dimensions
        npt.assert_allclose(
            res_biplot.features.shape,
            np.array([self.microbes.shape[0], latent_dim]))

        npt.assert_allclose(
            res_biplot.samples.shape,
            np.array([self.metabolites.shape[0], latent_dim]))


if __name__ == "__main__":
    unittest.main()
