import os
import time
import datetime
import biom
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multinomial import Multinomial
from torch.distributions.normal import Normal
from rhapsody.dataset import split_tables
from torch.utils.data import DataLoader
from rhapsody.layers import VecEmbedding, VecLinear
from rhapsody.batch import get_batch
from rhapsody.util import format_params
from skbio import OrdinationResults
from scipy.sparse.linalg import svds
from tensorboardX import SummaryWriter


class MMvec(torch.nn.Module):
    def __init__(self, num_samples, num_microbes, num_metabolites,
                 microbe_total, latent_dim,
                 in_prior=1, out_prior=1, device='cpu',
                 save_path=None):
        super(MMvec, self).__init__()
        self.num_microbes = num_microbes
        self.num_metabolites = num_metabolites
        self.num_samples = num_samples
        self.device = device
        self.microbe_total = microbe_total
        self.in_prior = 1
        self.out_prior = 1
        self.latent_dim = latent_dim

        # TODO: enable max norm in embedding to account for scale identifiability
        self.encoder = VecEmbedding(num_microbes, latent_dim)
        self.decoder = VecLinear(latent_dim, num_metabolites)

        if save_path is None:
            basename = "logdir"
            suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            self.save_path = "_".join([basename, suffix])
        else:
            self.save_path = save_path

    def forward(self, x):
        code = self.encoder(x)
        log_probs = self.decoder(code)
        return log_probs

    def loss(self, pred, obs):
        """ Computes the loss function to be minimized. """
        n = self.microbe_total * self.num_samples
        likelihood = n * torch.mean(Multinomial(logits=pred).log_prob(obs))
        encoder_prior = self.encoder.log_prob(self.in_prior)
        decoder_prior = self.decoder.log_prob(self.out_prior)
        prior = encoder_prior + decoder_prior
        return -(likelihood + prior)

    def fit(self, train_dataloader, test_dataloader, epochs=1000,
            learning_rate=1e-3, beta1=0.9, beta2=0.99,
            summary_interval=60, checkpoint_interval=3600):
        """ Fit the model

        Parameters
        ----------
        train_dataloader: torch.data_utils.DataLoader
            Torch DataLoader iterator for training samples
        test_dataloader: torch.data_utils.DataLoader
            Torch DataLoader iterator for testing samples
        epochs : int
            Number of epochs to train model
        learning_rate : float
            The initial learning rate
        beta1 : float
            First ADAM momentum constant
        beta2 : float
            Second ADAM momentum constant

        Returns
        -------
        losses : list of float
            Log likelihood of model
        errs : list of float
            Cross validation error between model and testing dataset
        """
        losses = []
        klds = []
        likes = []
        errs = []
        iteration = 0
        # custom make scheduler for alternating minimization
        baseline = 1e-8
        writer = SummaryWriter(self.save_path)

        last_checkpoint_time = 0
        last_summary_time = 0
        lrs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        with tqdm(total=epochs*len(lrs)*2) as pbar:
            for lr in lrs:
                # setup optimizer for alternating optimization
                for (b, l) in [[baseline, lr], [lr, baseline]]:
                    optimizer = optim.Adamax(
                        [
                            {'params': self.encoder.parameters(), 'lr': b},
                            {'params': self.decoder.parameters(), 'lr': l}
                        ],
                        betas=(beta1, beta2))
                    for _ in range(0, epochs):
                        now = time.time()
                        self.train()
                        for inp, out in train_dataloader:
                            inp = inp.to(self.device, non_blocking=True)
                            out = out.to(self.device, non_blocking=True)
                            for _ in range(5):
                                optimizer.zero_grad()
                                pred = self.forward(inp)
                                loss = self.loss(pred, out)
                                loss.backward()
                                iteration += 1
                                optimizer.step()

                        # write down summary stats
                        # Validation
                        err = torch.tensor(0.)
                        for inp, out in test_dataloader:
                            inp = inp.to(self.device, non_blocking=True)
                            out = out.to(self.device, non_blocking=True)
                            pred = self.forward(inp)
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

                        pbar.update(1)

                    # write down checkpoint after end of epoch
                    now = time.time()
                    if now - last_checkpoint_time > checkpoint_interval:
                        suffix = datetime.datetime.now().strftime(
                            "%y%m%d_%H%M%S")
                        torch.save(self.state_dict(),
                                   os.path.join(self.save_path,
                                                'checkpoint_' + suffix))
                        last_checkpoint_time = now

    def ranks(self, rowids, columnids):
        U = self.encoder.embedding.weight.cpu().detach().numpy()
        Ub = self.encoder.bias.weight.cpu().detach().numpy()
        V = self.decoder.weight_.cpu().detach().numpy()
        Vb = self.decoder.bias_.cpu().detach().numpy()
        res = Ub.reshape(-1, 1) + (U @ V.T) + Vb
        return pd.DataFrame(res, index=rowids, columns=columnids)

    def embeddings(self, rowids, columnids):
        U = self.encoder.embedding.weight.cpu().detach().numpy()
        Ub = self.encoder.bias.weight.cpu().detach().numpy()
        V = self.decoder.weight_.cpu().detach().numpy()
        Vb = self.decoder.bias_.cpu().detach().numpy()

        pc_ids = ['PC%d' % i for i in range(self.latent_dim)]
        df = pd.concat(
            (
                format_params(U, pc_ids, rowids, 'microbe'),
                format_params(V, pc_ids, columnids, 'metabolite'),
                format_params(Ub, ['bias'], rowids, 'microbe'),
                format_params(Vb, ['bias'], columnids, 'metabolite')
            ), axis=0)
        return df

    def ordination(self, rowids, columnids):
        pc_ids = ['PC%d' % i for i in range(self.latent_dim)]
        res = self.ranks(rowids, columnids)
        res = res - res.mean(axis=0).values
        u, s, v = svds(res.values, k=self.latent_dim)
        microbe_embed = u @ np.diag(s)
        metabolite_embed = v.T

        features = pd.DataFrame(
            microbe_embed, columns=pc_ids,
            index=rowids)
        samples = pd.DataFrame(
            metabolite_embed, columns=pc_ids,
            index=columnids)
        short_method_name = 'mmvec biplot'
        long_method_name = 'Multiomics mmvec biplot'
        eigvals = pd.Series(s, index=pc_ids)
        proportion_explained = pd.Series(s**2 / np.sum(s**2), index=pc_ids)
        biplot = OrdinationResults(
            short_method_name, long_method_name, eigvals,
            samples=samples, features=features,
            proportion_explained=proportion_explained)
        return biplot


def run_mmvec(microbes: biom.Table,
              metabolites: biom.Table,
              metadata: pd.DataFrame = None,
              training_column: str = None,
              num_testing_examples: int = 5,
              min_feature_count: int = 10,
              epochs: int = 100,
              batch_size: int = 50,
              latent_dim: int = 3,
              input_prior: float = 1,
              output_prior: float = 1,
              beta1: float = 0.9,
              beta2: float = 0.99,
              num_workers: int = 1,
              learning_rate: float = 0.001,
              arm_the_gpu: bool = False,
              summary_interval: int = 60,
              checkpoint_interval: int = 3600,
              summary_dir: str = None) -> MMvec:
    """ Basic procedure behind running mmvec """

    train_dataset, test_dataset = split_tables(
        microbes, metabolites,
        metadata=metadata, training_column=training_column,
        num_test=num_testing_examples,
        min_samples=min_feature_count, iterable=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers,
                                  pin_memory=arm_the_gpu)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers,
                                 pin_memory=arm_the_gpu)

    microbe_ids = microbes.ids(axis='observation')
    metabolite_ids = metabolites.ids(axis='observation')

    params = []

    d1, n = train_dataset.microbes.shape
    d2, n = train_dataset.metabolites.shape

    if arm_the_gpu:
        # pick out the first GPU
        device_name='cuda:0'
    else:
        device_name='cpu'

    total = train_dataset.microbes.sum().sum()
    model = MMvec(num_samples=n, microbe_total=total,
                  num_microbes=d1, num_metabolites=d2,
                  latent_dim=latent_dim,
                  in_prior=1, out_prior=1,
                  device=device_name,
                  save_path=summary_dir)
    model.to(device_name)
    model.fit(train_dataloader, test_dataloader,
              epochs=epochs, learning_rate=learning_rate,
              beta1=beta1, beta2=beta2,
              summary_interval=summary_interval,
              checkpoint_interval=checkpoint_interval)

    embeds = model.embeddings(
        train_dataset.microbes.ids(axis='observation'),
        train_dataset.metabolites.ids(axis='observation'))
    ranks = model.ranks(
        train_dataset.microbes.ids(axis='observation'),
        train_dataset.metabolites.ids(axis='observation'))
    ordination = model.ordination(
        train_dataset.microbes.ids(axis='observation'),
        train_dataset.metabolites.ids(axis='observation'))

    return embeds, ranks, ordination
