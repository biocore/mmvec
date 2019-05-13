import glob
import torch
import shutil
import unittest
import numpy as np
from skbio.stats.composition import clr_inv as softmax
from scipy.stats import spearmanr
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial.distance import pdist
from rhapsody.multimodal import MMvec
from rhapsody.util import random_multimodal


class TestMMvec(unittest.TestCase):
    def setUp(self):
        # build small simulation
        self.latent_dim = 2
        res = random_multimodal(
            num_microbes=8, num_metabolites=8, num_samples=150,
            latent_dim=self.latent_dim, sigmaQ=2, sigmaU=1, sigmaV=1,
            microbe_total=100, metabolite_total=1000, seed=1
        )
        (self.microbes, self.metabolites, self.X, self.B,
         self.U, self.Ubias, self.V, self.Vbias) = res
        num_test = 10
        self.trainX = self.microbes.iloc[:-num_test]
        self.testX = self.microbes.iloc[-num_test:]
        self.trainY = self.metabolites.iloc[:-num_test]
        self.testY = self.metabolites.iloc[-num_test:]

    def tearDown(self):
        # remove all log directories
        for r in glob.glob("logdir*"):
            shutil.rmtree(r)

    def test_fit(self):
        np.random.seed(1)
        torch.manual_seed(1)

        n, d1 = self.trainX.shape
        n, d2 = self.trainY.shape
        latent_dim = self.latent_dim

        model = MMvec(num_microbes=d1, num_metabolites=d2, latent_dim=latent_dim,
                      batch_size=10, subsample_size=300,
                      device='cpu')
        model, losses, cv = model.fit(
            csr_matrix(self.trainX.values), self.trainY.values,
            csr_matrix(self.testX.values), self.testY.values,
            epochs=10, gamma=0.1,
            learning_rate=1, mc_samples=5,
            beta1=0.9, beta2=0.95, step_size=1)

        # Just check to see if the loss / cross validation accuracy decreased
        # since the object is non-convex, the results will be different for
        # every random initialization.
        self.assertGreater(losses[0], losses[-1])
        self.assertGreater(cv[0], cv[-1])


if __name__ == "__main__":
    unittest.main()
