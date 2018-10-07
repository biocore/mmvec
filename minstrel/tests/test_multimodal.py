import glob
import shutil
import unittest
import numpy as np
from skbio.stats.composition import clr_inv as softmax
from scipy.stats import spearmanr
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist
from minstrel.multimodal import Autoencoder
from minstrel.util import random_multimodal
from tensorflow import set_random_seed
import tensorflow as tf


class TestAutoencoder(unittest.TestCase):
    def setUp(self):
        # build small simulation
        res = random_multimodal(
            num_microbes=8, num_metabolites=8, num_samples=150,
            latent_dim=2, sigmaQ=2,
            microbe_total=1000, metabolite_total=10000, seed=1
        )
        (self.microbes, self.metabolites, self.X, self.B,
         self.U, self.Ubias, self.V, self.Vbias) = res
        num_train = 10
        self.trainX = self.microbes.iloc[:-num_train]
        self.testX = self.microbes.iloc[-num_train:]
        self.trainY = self.metabolites.iloc[:-num_train]
        self.testY = self.metabolites.iloc[-num_train:]

    def tearDown(self):
        # remove all log directories
        for r in glob.glob("logdir*"):
            shutil.rmtree(r)

    def test_fit(self):
        np.random.seed(1)
        tf.reset_default_graph()
        n, d1 = self.trainX.shape
        n, d2 = self.trainY.shape
        with tf.Graph().as_default(), tf.Session() as session:
            set_random_seed(0)
            model = Autoencoder(beta_1=0.8, beta_2=0.9, latent_dim=2)
            model(session,
                  coo_matrix(self.trainX.values), self.trainY.values,
                  coo_matrix(self.testX.values), self.testY.values)
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
            self.assertLess(u_p, 1e-2)
            self.assertLess(v_p, 1e-2)
            self.assertLess(s_p, 1e-2)

            # sanity check cross validation
            self.assertLess(model.cv.eval(), 500)


if __name__ == "__main__":
    unittest.main()
