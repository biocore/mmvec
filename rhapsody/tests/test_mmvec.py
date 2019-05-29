import unittest
import torch
import copy
import glob
import shutil
import numpy as np
from rhapsody.mmvec import MMvec
from rhapsody.sim import random_bimodal
from scipy.spatial.distance import pdist
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr


class TestMMvec(unittest.TestCase):

    def setUp(self):

        self.num_microbes = 100
        self.num_metabolites = 20
        self.num_samples = 100
        self.latent_dim = 5
        self.means = (-3, 3)
        self.microbe_total = 300
        self.metabolite_total = 900
        self.uB = 0
        self.sigmaB = 1
        self.sigmaQ = 2
        self.uU = 0
        self.sigmaU = 0.01
        self.uV = 0
        self.sigmaV = 0.01
        self.s = 0.2
        self.seed = 1
        self.eUun = 0.2
        self.eVun = 0.2
        res = random_bimodal(num_microbes=self.num_microbes,
                             num_metabolites=self.num_metabolites,
                             num_samples=self.num_samples,
                             latent_dim=self.latent_dim,
                             means=self.means,
                             microbe_total=self.microbe_total,
                             metabolite_total=self.metabolite_total,
                             uU=0,
                             sigmaU=1,
                             uV=0,
                             sigmaV=1,
                             seed=0)
        (self.microbe_counts, self.metabolite_counts,
         self.eUmain, self.eVmain, self.eUbias, self.eVbias) = res

        # fit parameters
        self.epochs = 100
        self.mc_samples = 1
        self.beta1 = 0.8
        self.beta2 = 0.9
        self.batch_size = 10
        self.subsample_size = 500
        self.learning_rate = 1e-1

    def tearDown(self):
        for f in glob.glob("logdir*"):
            shutil.rmtree(f)

    def test_mmvec_loss(self):
        # test to see if errors have been decreased
        torch.manual_seed(0)
        microbe_read_total = self.microbe_counts.sum()
        model = MMvec(
            self.num_samples, self.num_microbes, self.num_metabolites,
            microbe_read_total, self.latent_dim, 1,
            self.subsample_size)
        trainX = csr_matrix(self.microbe_counts)
        trainY = self.metabolite_counts
        testX = csr_matrix(self.microbe_counts)
        testY = self.metabolite_counts
        res = model.fit(trainX, trainY, testX, testY,
                        epochs=self.epochs, learning_rate=self.learning_rate,
                        beta1=self.beta1, beta2=self.beta2)

        losses, klds, likes, errs = res
        self.assertGreater(np.mean(losses[:3]), np.mean(losses[:-3]))
        self.assertLess(np.mean(likes[:3]), np.mean(likes[:-3]))
        self.assertGreater(np.mean(klds[:3]), np.mean(klds[:-3]))
        self.assertGreater(np.mean(errs[:3]), np.mean(errs[:-3]))

    def test_mmvec_param_change(self):
        # test to see if the parameters have all been trained
        torch.manual_seed(0)
        microbe_read_total = self.microbe_counts.sum()
        model = MMvec(
            self.num_samples, self.num_microbes, self.num_metabolites,
            microbe_read_total, self.latent_dim, self.batch_size,
            self.subsample_size)
        before_model = copy.deepcopy(model)

        trainX = csr_matrix(self.microbe_counts)
        trainY = self.metabolite_counts
        testX = csr_matrix(self.microbe_counts)
        testY = self.metabolite_counts
        model.fit(trainX, trainY, testX, testY,
                  epochs=self.epochs, learning_rate=self.learning_rate,
                  beta1=self.beta1, beta2=self.beta2)

        after_model = copy.deepcopy(model)

        before_u = np.array(before_model.encoder.embedding.weight.detach())
        before_v = np.array(before_model.decoder.mean.weight.detach())
        before_ubias = np.array(before_model.encoder.bias.weight.detach())
        before_vbias = np.array(before_model.decoder.mean.bias.detach())

        after_u = np.array(after_model.encoder.embedding.weight.detach())
        after_v = np.array(after_model.decoder.mean.weight.detach())
        after_ubias = np.array(after_model.encoder.bias.weight.detach())
        after_vbias = np.array(after_model.decoder.mean.bias.detach())

        self.assertFalse(np.allclose(before_u, after_u))
        self.assertFalse(np.allclose(before_v, after_v))
        self.assertFalse(np.allclose(before_ubias, after_ubias))
        self.assertFalse(np.allclose(before_vbias, after_vbias))

    @unittest.skip("Sanity check fit.  Note that this is random.")
    def test_mmvec_fit(self):
        # test to see if the parameters from the model are actually legit
        torch.manual_seed(0)
        np.random.seed(0)

        microbe_read_total = self.microbe_counts.sum()
        model = MMvec(
            self.num_samples, self.num_microbes, self.num_metabolites,
            microbe_read_total, self.latent_dim, self.batch_size,
            self.subsample_size)
        trainX = csr_matrix(self.microbe_counts)
        trainY = self.metabolite_counts
        testX = csr_matrix(self.microbe_counts)
        testY = self.metabolite_counts
        model.fit(trainX, trainY, testX, testY,
                  epochs=self.epochs, learning_rate=self.learning_rate,
                  beta1=self.beta1, beta2=self.beta2)

        u = np.array(model.encoder.embedding.weight.detach())
        v = np.array(model.decoder.mean.weight.detach())
        ubias = np.array(model.encoder.bias.weight.detach())
        vbias = np.array(model.decoder.mean.bias.detach())
        # test to see if the U distances are correct
        r, p = spearmanr(pdist(self.eUmain), pdist(u))
        self.assertLess(p, 0.001)
        # test to see if the V distances are correct
        r, p = spearmanr(pdist(self.eVmain.T), pdist(v))
        self.assertLess(p, 0.001)
        # test to see if the ranks correct
        exp = np.hstack(
            (np.ones((self.num_microbes, 1)), self.eUbias, self.eUmain)
        ) @  np.vstack(
            (self.eVbias, np.ones((1, self.num_metabolites)), self.eVmain)
        )
        res = np.hstack(
            (np.ones((u.shape[0], 1)), ubias, u)
        ) @ np.hstack(
            (vbias.reshape(-1, 1), np.ones((v.shape[0], 1)), v)
        ).T
        r, p = spearmanr(exp.ravel(), res.ravel())
        self.assertLess(p, 0.001)

    @unittest.skip("Sanity check fit with gpu.  Note that this is random.")
    def test_mmvec_fit_gpu(self):
        # test to see if the parameters from the model are actually legit
        torch.manual_seed(0)
        np.random.seed(0)

        microbe_read_total = self.microbe_counts.sum()
        model = MMvec(
            self.num_samples, self.num_microbes, self.num_metabolites,
            microbe_read_total, self.latent_dim, self.batch_size,
            self.subsample_size)

        device_name = 'cuda'
        model = model.to(device=device_name)
        trainX = csr_matrix(self.microbe_counts)
        trainY = self.metabolite_counts
        testX = csr_matrix(self.microbe_counts)
        testY = self.metabolite_counts
        model.fit(trainX, trainY, testX, testY,
                  epochs=self.epochs, learning_rate=self.learning_rate,
                  beta1=self.beta1, beta2=self.beta2,
                  device_name=device_name)

        u = np.array(model.encoder.embedding.weight.detach())
        v = np.array(model.decoder.mean.weight.detach())
        ubias = np.array(model.encoder.bias.weight.detach())
        vbias = np.array(model.decoder.mean.bias.detach())
        # test to see if the U distances are correct
        r, p = spearmanr(pdist(self.eUmain), pdist(u))
        self.assertLess(p, 0.001)
        # test to see if the V distances are correct
        r, p = spearmanr(pdist(self.eVmain.T), pdist(v))
        self.assertLess(p, 0.001)
        # test to see if the ranks correct
        exp = np.hstack(
            (np.ones((self.num_microbes, 1)), self.eUbias, self.eUmain)
        ) @  np.vstack(
            (self.eVbias, np.ones((1, self.num_metabolites)), self.eVmain)
        )
        res = np.hstack(
            (np.ones((u.shape[0], 1)), ubias, u)
        ) @ np.hstack(
            (vbias.reshape(-1, 1), np.ones((v.shape[0], 1)), v)
        ).T
        r, p = spearmanr(exp.ravel(), res.ravel())
        self.assertLess(p, 0.001)


if __name__ == "__main__":
    unittest.main()
