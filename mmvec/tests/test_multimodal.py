import glob
import shutil
import unittest
import numpy as np
import pandas as pd
from biom import load_table
from skbio.stats.composition import clr_inv as softmax
from skbio.util import get_data_path
from scipy.stats import spearmanr
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist
from mmvec.ALR import MMvecALR
from mmvec.train import mmvec_training_loop
from mmvec.util import random_multimodal


class TestMMvec(unittest.TestCase):
    def setUp(self):
        # build small simulation
        np.random.seed(1)
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
        #tf.reset_default_graph()
        n, d1 = self.trainX.shape
        n, d2 = self.trainY.shape
        model = MMvecALR(self.trainX, self.trainY, latent_dim=2)
        mmvec_training_loop(model=model, learning_rate=0.01, batch_size=50,
                            epochs=10000)

        U_ = np.hstack(
            (np.ones((self.U.shape[0], 1)), self.Ubias, self.U))
        V_ = np.vstack(
            (self.Vbias, np.ones((1, self.V.shape[1])), self.V))


        res = softmax(model.ranks().numpy())
        exp = softmax(np.hstack((np.zeros((d1, 1)), U_ @ V_)))

        s_r, s_p = spearmanr(np.ravel(res), np.ravel(exp))

        u_r, u_p = spearmanr(pdist(model.U), pdist(self.U))
        v_r, v_p = spearmanr(pdist(model.V.T), pdist(self.V.T))

        self.assertGreater(u_r, 0.5)
        self.assertGreater(v_r, 0.5)
        self.assertGreater(s_r, 0.5)
        self.assertLess(u_p, 5e-2)
        self.assertLess(v_p, 5e-2)
        self.assertLess(s_p, 5e-2)


        assert False
#        with tf.Graph().as_default(), tf.Session() as session:
#            set_random_seed(0)
#            model = MMvec(beta_1=0.8, beta_2=0.9, latent_dim=2)
#            model(session,
#                  coo_matrix(self.trainX.values), self.trainY.values,
#                  coo_matrix(self.testX.values), self.testY.values)
#            model.fit(epoch=1000)
#
#
#            # sanity check cross validation
#            self.assertLess(model.cv.eval(), 500)


#class TestMMvecSoilsBenchmark(unittest.TestCase):
#    def setUp(self):
#        self.microbes = load_table(get_data_path('soil_microbes.biom'))
#        self.metabolites = load_table(get_data_path('soil_metabolites.biom'))
#        X = self.microbes.to_dataframe().T
#        Y = self.metabolites.to_dataframe().T
#        X = X.loc[Y.index]
#        self.trainX = X.iloc[:-2]
#        self.trainY = Y.iloc[:-2]
#        self.testX = X.iloc[-2:]
#        self.testY = Y.iloc[-2:]
#
#    def tearDown(self):
#        # remove all log directories
#        for r in glob.glob("logdir*"):
#            shutil.rmtree(r)

#    def test_soils(self):
#        np.random.seed(1)
#        n, d1 = self.trainX.shape
#        n, d2 = self.trainY.shape
#
#        with tf.Graph().as_default(), tf.Session() as session:
#            set_random_seed(0)
#            model = MMvec(beta_1=0.8, beta_2=0.9, latent_dim=1,
#                          learning_rate=1e-3)
#            model(session,
#                  coo_matrix(self.trainX.values), self.trainY.values,
#                  coo_matrix(self.testX.values), self.testY.values)
#            model.fit(epoch=1000)
#
#            ranks = pd.DataFrame(
#                model.ranks(),
#                index=self.microbes.ids(axis='observation'),
#                columns=self.metabolites.ids(axis='observation'))
#
#            microcoleus_metabolites = [
#                '(3-methyladenine)', '7-methyladenine', '4-guanidinobutanoate',
#                'uracil', 'xanthine', 'hypoxanthine', '(N6-acetyl-lysine)',
#                'cytosine', 'N-acetylornithine', 'N-acetylornithine',
#                'succinate', 'adenosine', 'guanine', 'adenine']
#            mprobs = ranks.loc['rplo 1 (Cyanobacteria)']
#            self.assertEqual(np.sum(mprobs.loc[microcoleus_metabolites] > 0),
#                             len(microcoleus_metabolites))
#

#class TestMMvecBenchmark(unittest.TestCase):
#    def setUp(self):
#        # build small simulation
#        res = random_multimodal(
#            num_microbes=100, num_metabolites=1000, num_samples=300,
#            latent_dim=2, sigmaQ=2,
#            microbe_total=5000, metabolite_total=10000, seed=1
#        )
#        (self.microbes, self.metabolites, self.X, self.B,
#         self.U, self.Ubias, self.V, self.Vbias) = res
#        num_train = 10
#        self.trainX = self.microbes.iloc[:-num_train]
#        self.testX = self.microbes.iloc[-num_train:]
#        self.trainY = self.metabolites.iloc[:-num_train]
#        self.testY = self.metabolites.iloc[-num_train:]
#
#    @unittest.skip("Only for benchmarking")
#    def test_gpu(self):
#        np.random.seed(1)
#        tf.reset_default_graph()
#        n, d1 = self.trainX.shape
#        n, d2 = self.trainY.shape
#
#        with tf.Graph().as_default(), tf.Session() as session:
#            set_random_seed(0)
#            model = MMvec(beta_1=0.8, beta_2=0.9, latent_dim=2,
#                          batch_size=2000,
#                          device_name="/device:GPU:0")
#            model(session,
#                  coo_matrix(self.trainX.values), self.trainY.values,
#                  coo_matrix(self.testX.values), self.testY.values)
#            model.fit(epoch=10000)

    #@unittest.skip("Only for benchmarking")
    #def test_cpu(self):
    #    print('CPU run')
    #    np.random.seed(1)
    #    tf.reset_default_graph()
    #    n, d1 = self.trainX.shape
    #    n, d2 = self.trainY.shape

    #    with tf.Graph().as_default(), tf.Session() as session:
    #        set_random_seed(0)
    #        model = MMvec(beta_1=0.8, beta_2=0.9, latent_dim=2,
    #                      batch_size=2000)
    #        model(session,
    #              coo_matrix(self.trainX.values), self.trainY.values,
    #              coo_matrix(self.testX.values), self.testY.values)
    #        model.fit(epoch=10000)


if __name__ == "__main__":
    unittest.main()
