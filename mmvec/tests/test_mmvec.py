import glob
import shutil
import unittest
import numpy as np
from biom import load_table
from skbio.util import get_data_path
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from mmvec.mmvec import MMvec
from mmvec.util import random_multimodal
from mmvec.dataset import split_tables
from mmvec.scheduler import AlternatingStepLR
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import datetime


class TestMMvecTrack(unittest.TestCase):
    """ Test simulation while tracking improvement over iterations"""
    def setUp(self):
        # build small simulation
        np.random.seed(0)
        self.latent_dim = 2
        self.num_microbes = 15
        self.num_metabolites = 16
        res = random_multimodal(
            num_microbes=self.num_microbes,
            num_metabolites=self.num_metabolites, num_samples=220,
            latent_dim=self.latent_dim, sigmaQ=1.0, sigmaU=1, sigmaV=1,
            microbe_total=100, metabolite_total=1000, seed=1
        )
        (self.microbes, self.metabolites, self.X, self.B,
         self.U, self.Ubias, self.V, self.Vbias) = res
        num_test = 10
        min_feature_count = 1
        self.train_dataset, self.test_dataset = split_tables(
            self.microbes, self.metabolites,
            num_test=num_test,
            iterable=False,
            min_samples=min_feature_count)

        U_ = np.hstack(
            (np.ones((self.num_microbes, 1)), self.Ubias, self.U))
        V_ = np.vstack(
            (self.Vbias, np.ones((1, self.num_metabolites - 1)), self.V))
        self.exp_ranks = np.hstack((np.zeros((self.num_microbes, 1)), U_ @ V_))

    def test_track(self):
        batch = 50
        epochs = 100
        learning_rate = 0.1
        step_size = 10
        beta1 = 0.9
        beta2 = 0.999
        clip_norm = 10
        train_dataloader = DataLoader(self.train_dataset, batch_size=batch,
                                      shuffle=True, num_workers=0)
        test_dataloader = DataLoader(self.test_dataset, batch_size=50,
                                     shuffle=True, num_workers=0)

        d1, n = self.train_dataset.microbes.shape
        d2, n = self.train_dataset.metabolites.shape
        latent_dim = self.latent_dim
        total = self.train_dataset.microbes.sum().sum()
        model = MMvec(num_samples=n, num_microbes=d1, num_metabolites=d2,
                      microbe_total=total, latent_dim=latent_dim,
                      clip_norm=clip_norm,
                      device='cpu')

        save_path = 'epochs%d_lr%f_step%d_batch%d_clip%.2f_b%.2f_bb%.2f' % (
            epochs, learning_rate, step_size, batch, clip_norm, beta1, beta2
        )
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        writer = SummaryWriter(suffix + save_path)
        optimizer = optim.Adamax(
            [
                {'params': model.encoder.parameters(), 'lr': learning_rate},
                {'params': model.decoder.parameters(), 'lr': learning_rate}
            ],
            betas=(beta1, beta2))

        scheduler = AlternatingStepLR(optimizer, step_size)
        for iteration in range(epochs):
            model.train()
            for inp, out in train_dataloader:
                inp = inp.to(model.device, non_blocking=True)
                out = out.to(model.device, non_blocking=True)

                optimizer.zero_grad()
                pred = model.forward(inp)
                loss = model.loss(pred, out)
                loss.backward()
                clip_grad_norm_(model.parameters(),
                                model.clip_norm)
                optimizer.step()
            scheduler.step()

            # write down summary stats after each epoch
            err = torch.tensor(0.)
            for inp, out in test_dataloader:
                inp = inp.to(model.device, non_blocking=True)
                out = out.to(model.device, non_blocking=True)
                pred = model.forward(inp)
                mt = torch.sum(out, 1).view(-1, 1)
                err += torch.mean(
                    torch.abs(
                        F.softmax(pred, dim=1) * mt - out
                    )
                )

            writer.add_scalar(
                'cv_mae', err, iteration)
            writer.add_scalar(
                'log_likelihood', loss, iteration)
            writer.add_scalar(
                'encoder_lr', optimizer.param_groups[0]['lr'], iteration)
            writer.add_scalar(
                'decoder_lr', optimizer.param_groups[1]['lr'], iteration)

            u = model.encoder.embedding.weight.detach().numpy()
            v = model.decoder.weight.detach().numpy()

            res = spearmanr(pdist(self.U), pdist(u))
            writer.add_scalar('u_fit', res[0], iteration)
            res = spearmanr(pdist(self.V.T), pdist(v))
            writer.add_scalar('v_fit', res[0], iteration)

            res_ranks = model.ranks(np.arange(self.num_microbes),
                                    np.arange(self.num_metabolites))

            res = spearmanr(self.exp_ranks.ravel(), res_ranks.values.ravel())
            writer.add_scalar('sim_rank_err', res[0], iteration)

        print('U', spearmanr(pdist(self.U), pdist(u)))
        print('V', spearmanr(pdist(self.V.T), pdist(v)))
        print('ranks', spearmanr(self.exp_ranks.ravel(),
                                 res_ranks.values.ravel()))


class TestMMvecSim(unittest.TestCase):

    def setUp(self):
        # build small simulation
        np.random.seed(0)
        self.latent_dim = 2
        self.num_microbes = 15
        self.num_metabolites = 16
        res = random_multimodal(
            num_microbes=self.num_microbes,
            num_metabolites=self.num_metabolites,
            num_samples=220, latent_dim=self.latent_dim, sigmaQ=1,
            sigmaU=1, sigmaV=1, microbe_total=100,
            metabolite_total=1000, seed=1
        )
        (self.microbes, self.metabolites, self.X, self.B,
         self.U, self.Ubias, self.V, self.Vbias) = res
        num_test = 10
        min_feature_count = 1
        self.train_dataset, self.test_dataset = split_tables(
            self.microbes, self.metabolites,
            num_test=num_test,
            min_samples=min_feature_count)

        U_ = np.hstack(
            (np.ones((self.num_microbes, 1)), self.Ubias, self.U))
        V_ = np.vstack(
            (self.Vbias, np.ones((1, self.num_metabolites - 1)), self.V))
        self.exp_ranks = np.hstack((np.zeros((self.num_microbes, 1)), U_ @ V_))

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
                      microbe_total=total, latent_dim=latent_dim, clip_norm=20,
                      device='cpu')
        model.fit(train_dataloader,
                  test_dataloader,
                  epochs=100, learning_rate=.1,
                  beta1=0.9, beta2=0.999)

        res_ranks = model.ranks(np.arange(self.num_microbes),
                                np.arange(self.num_metabolites))

        res = spearmanr(self.exp_ranks.ravel(), res_ranks.values.ravel())
        self.assertGreater(res.correlation, 0.25)
        self.assertLess(res.pvalue, 1e-4)


class TestMMvecSimIterable(unittest.TestCase):
    def setUp(self):
        # build small simulation
        np.random.seed(0)
        self.latent_dim = 2
        self.num_microbes = 15
        self.num_metabolites = 16

        res = random_multimodal(
            num_microbes=self.num_microbes,
            num_metabolites=self.num_metabolites,
            num_samples=220, latent_dim=self.latent_dim,
            sigmaQ=1, sigmaU=1, sigmaV=1,
            microbe_total=100, metabolite_total=1000, seed=1
        )
        (self.microbes, self.metabolites, self.X, self.B,
         self.U, self.Ubias, self.V, self.Vbias) = res
        num_test = 10
        min_feature_count = 1
        self.train_dataset, self.test_dataset = split_tables(
            self.microbes, self.metabolites,
            num_test=num_test, iterable=True,
            min_samples=min_feature_count)

        U_ = np.hstack(
            (np.ones((self.num_microbes, 1)), self.Ubias, self.U))
        V_ = np.vstack(
            (self.Vbias, np.ones((1, self.num_metabolites - 1)), self.V))
        self.exp_ranks = np.hstack((np.zeros((self.num_microbes, 1)), U_ @ V_))

    def test_fit_iterable(self):
        np.random.seed(1)
        torch.manual_seed(1)
        train_dataloader = DataLoader(self.train_dataset, batch_size=50,
                                      shuffle=False, num_workers=1)
        test_dataloader = DataLoader(self.test_dataset, batch_size=50,
                                     shuffle=False, num_workers=1)

        d1, n = self.train_dataset.microbes.shape
        d2, n = self.train_dataset.metabolites.shape
        latent_dim = self.latent_dim
        total = self.train_dataset.microbes.sum().sum()

        model = MMvec(num_samples=n, num_microbes=d1, num_metabolites=d2,
                      microbe_total=total, latent_dim=latent_dim,
                      clip_norm=20, device='cpu')
        model.fit(train_dataloader,
                  test_dataloader,
                  epochs=3, learning_rate=.1,
                  beta1=0.9, beta2=0.999)

        res_ranks = model.ranks(np.arange(self.num_microbes),
                                np.arange(self.num_metabolites))

        res = spearmanr(self.exp_ranks.ravel(), res_ranks.values.ravel())
        self.assertGreater(res.correlation, 0.25)
        self.assertLess(res.pvalue, 1e-4)


class TestMMvecSoils(unittest.TestCase):
    def setUp(self):
        microbe_file = get_data_path('microbes.biom')
        metabolite_file = get_data_path('metabolites.biom')
        self.microbes = load_table(microbe_file)
        self.metabolites = load_table(metabolite_file)

        self.known_metabolites = {
            '(3-methyladenine)', '7-methyladenine', '4-guanidinobutanoate',
            'uracil', 'xanthine', 'hypoxanthine', '(N6-acetyl-lysine)',
            'cytosine', 'N-acetylornithine', 'N-acetylornithine',
            'succinate', 'adenosine', 'guanine', 'adenine'
        }

        self.train_dataset, self.test_dataset = split_tables(
            self.microbes, self.metabolites,
            num_test=1,
            min_samples=1)

    def test_soils(self):
        np.random.seed(1)
        torch.manual_seed(1)

        train_dataloader = DataLoader(self.train_dataset, batch_size=50,
                                      shuffle=True, num_workers=0)
        test_dataloader = DataLoader(self.test_dataset, batch_size=50,
                                     shuffle=True, num_workers=0)

        d1, n = self.train_dataset.microbes.shape
        d2, n = self.train_dataset.metabolites.shape

        latent_dim = 1
        total = self.train_dataset.microbes.sum().sum()
        model = MMvec(num_samples=n, num_microbes=d1, num_metabolites=d2,
                      microbe_total=total, latent_dim=latent_dim,
                      clip_norm=20, device='cpu')
        model.fit(
            train_dataloader, test_dataloader,
            epochs=1000, step_size=250, learning_rate=1.0,
            beta1=0.9, beta2=0.95)
        rowids = np.arange(d1)
        colids = np.arange(d2)
        ranks = model.ranks(rowids, colids)
        # pull out microcoleus
        idx = ranks.iloc[0, :] > 0

        res = set(self.metabolites.ids(axis='observation')[idx])
        self.assertEqual(len(res & self.known_metabolites), 13)


if __name__ == "__main__":
    unittest.main()
