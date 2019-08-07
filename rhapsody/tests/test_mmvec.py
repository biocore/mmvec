import glob
import torch
import shutil
import unittest
import numpy as np
from biom import load_table
from skbio.stats.composition import clr_inv as softmax
from skbio.util import get_data_path
from scipy.stats import spearmanr
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial.distance import pdist
from rhapsody.mmvec import MMvec
from rhapsody.util import random_multimodal
from rhapsody.dataset import PairedDataset, split_tables
from torch.utils.data import DataLoader


class TestMMvecSim(unittest.TestCase):
    def setUp(self):
        # build small simulation
        np.random.seed(0)
        self.latent_dim = 2
        res = random_multimodal(
            num_microbes=15, num_metabolites=15, num_samples=220,
            latent_dim=self.latent_dim, sigmaQ=1, sigmaU=1, sigmaV=1,
            microbe_total=100, metabolite_total=1000, seed=1
        )
        (self.microbes, self.metabolites, self.X, self.B,
         self.U, self.Ubias, self.V, self.Vbias) = res
        num_test = 10
        min_feature_count = 1
        self.train_dataset, self.test_dataset = split_tables(
            self.microbes, self.metabolites,
            num_test=num_test,
            min_samples=min_feature_count)

    def tearDown(self):
        # remove all log directories
        for r in glob.glob("logdir*"):
            shutil.rmtree(r)

    def test_fit(self):
        np.random.seed(1)
        torch.manual_seed(1)
        train_dataloader = DataLoader(self.train_dataset, batch_size=50,
                                      shuffle=True, num_workers=1)
        test_dataloader = DataLoader(self.test_dataset, batch_size=50,
                                     shuffle=True, num_workers=1)

        d1, n = self.train_dataset.microbes.shape
        d2, n = self.train_dataset.metabolites.shape
        latent_dim = self.latent_dim
        total = self.train_dataset.microbes.sum().sum()

        model = MMvec(num_samples=n, num_microbes=d1, num_metabolites=d2,
                      microbe_total=total, latent_dim=latent_dim,
                      batch_size=20, subsample_size=100,
                      device='cpu')
        _ = model.fit(train_dataloader,
                      test_dataloader,
                      epochs=100, learning_rate=.1,
                      beta1=0.9, beta2=0.999)

        # Loose checks on the weight matrices to make sure
        # that we aren't learning complete garbage
        u = model.encoder.embedding.weight.detach().numpy()
        v = model.decoder.weight.detach().numpy()

        ubias = model.encoder.bias.weight.detach().numpy()
        vbias = model.decoder.bias.detach().numpy()
        res = spearmanr(pdist(self.U), pdist(u))
        self.assertGreater(res.correlation, 0.4)
        self.assertLess(res.pvalue, 1e-5)
        res = spearmanr(pdist(self.V.T), pdist(v))
        self.assertGreater(res.correlation, 0.4)
        self.assertLess(res.pvalue, 1e-5)


class TestMMvecSoils(unittest.TestCase):
    def setUp(self):
        microbe_file = get_data_path('microbes.biom')
        metabolite_file = get_data_path('metabolites.biom')
        self.microbes = load_table(microbe_file)
        self.metabolites = load_table(metabolite_file)

        self.known_metabolites = {
            '(3-methyladenine)', '7-methyladenine', '4-guanidinobutanoate', 'uracil',
            'xanthine', 'hypoxanthine', '(N6-acetyl-lysine)', 'cytosine',
            'N-acetylornithine', 'N-acetylornithine', 'succinate',
            'adenosine', 'guanine', 'adenine'
        }

        self.train_dataset, self.test_dataset = split_tables(
            self.microbes, self.metabolites,
            num_test=1,
            min_samples=1)


    def test_soils(self):
        np.random.seed(1)
        torch.manual_seed(1)

        train_dataloader = DataLoader(self.train_dataset, batch_size=50,
                                      shuffle=True, num_workers=1)
        test_dataloader = DataLoader(self.test_dataset, batch_size=50,
                                     shuffle=True, num_workers=1)

        d1, n = self.train_dataset.microbes.shape
        d2, n = self.train_dataset.metabolites.shape

        latent_dim = 1
        total = self.train_dataset.microbes.sum().sum()
        model = MMvec(num_samples=n, num_microbes=d1, num_metabolites=d2,
                      microbe_total=total, latent_dim=latent_dim,
                      batch_size=3, subsample_size=500,
                      device='cpu')
        losses, errs = model.fit(
            train_dataloader, test_dataloader,
            epochs=100, learning_rate=0.1,
            beta1=0.9, beta2=0.95)
        ranks = model.ranks()
        ranks = ranks.detach().numpy()
        # pull out microcoleus
        idx = ranks[0, :] > 0

        res = set(self.metabolites.ids(axis='observation')[idx])
        self.assertEqual(len(res & self.known_metabolites), 13)



if __name__ == "__main__":
    unittest.main()
