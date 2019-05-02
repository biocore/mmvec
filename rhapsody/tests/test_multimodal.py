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
        self.num_samples = 150
        res = random_multimodal(
            num_microbes=8, num_metabolites=8, num_samples=150,
            latent_dim=self.latent_dim, sigmaQ=2,
            microbe_total=1000, metabolite_total=10000, seed=1
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
        k = self.latent_dim
        num_samples = self.num_samples

        model = MMvec(num_microbes=d1, num_metabolites=d2, latent_dim=k)
        model.fit(csr_matrix(self.trainX.values), self.trainY.values,
                  device='cpu', learning_rate=1e-3, mc_samples=5,
                  beta1=0.8, beta2=0.9)

        U = model.embeddings.weight.detach().numpy()
        Ub = model.bias.weight.detach().numpy()
        V = model.muV.detach().numpy()
        Vb = model.muVb.detach().numpy()

        modelU = np.hstack(
            (np.ones((U.shape[0], 1)), Ub, U))
        modelV = np.vstack(
            (Vb, np.ones((1, V.shape[1])), V))

        U_ = np.hstack(
            (np.ones((self.U.shape[0], 1)), self.Ubias, self.U))
        V_ = np.vstack(
            (self.Vbias, np.ones((1, self.V.shape[1])), self.V))

        u_r, u_p = spearmanr(pdist(U), pdist(self.U))
        v_r, v_p = spearmanr(pdist(V.T), pdist(self.V.T))

        res = softmax(np.hstack((np.zeros((d1, 1)), modelU @ modelV)))
        exp = softmax(np.hstack((np.zeros((d1, 1)), U_ @ V_)))
        s_r, s_p = spearmanr(np.ravel(res), np.ravel(exp))

        self.assertGreater(u_r, 0.5)
        self.assertGreater(v_r, 0.5)
        self.assertGreater(s_r, 0.5)
        self.assertLess(u_p, 1e-2)
        self.assertLess(v_p, 1e-2)
        self.assertLess(s_p, 1e-2)

if __name__ == "__main__":
    unittest.main()
