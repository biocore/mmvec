import datetime
from tqdm import tqdm
from rhapsody.batch import get_batch
from rhapsody.layers import GaussianEmbedding, GaussianDecoder

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multinomial import Multinomial


class MMvec(nn.Module):
    def __init__(self, num_samples, num_microbes, num_metabolites,
                 microbe_total, latent_dim, batch_size=10,
                 subsample_size=100):
        super(MMvec, self).__init__()
        self.num_microbes = num_microbes
        self.num_metabolites = num_metabolites
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.subsample_size = subsample_size
        self.microbe_total = microbe_total
        self.encoder = GaussianEmbedding(in_features=num_microbes,
                                         out_features=latent_dim)
        self.decoder = GaussianDecoder(in_features=latent_dim,
                                       out_features=num_metabolites)

    def forward(self, x):
        code = self.encoder(x)
        log_probs = self.decoder(code)
        return log_probs

    def loss(self, pred, obs):
        """ Computes the loss function to be minimized. """
        kld = self.encoder.divergence() + self.decoder.divergence()
        n = self.microbe_total * self.num_samples
        likelihood = n * torch.mean(Multinomial(logits=pred).log_prob(obs))
        metabolite_total = torch.sum(obs, 1).view(-1, 1)
        err = torch.mean(
            torch.abs(F.softmax(pred, dim=1) * metabolite_total - obs))
        elbo = kld + likelihood
        return -elbo, kld, likelihood, err

    def validate(self, inp, out):
        """ Computes cross-validation scores on holdout train/test set.

        Here, the mean absolute error is computed, which can be interpreted
        as the average number of counts that were incorrectly predicted.
        """
        logprobs = self.forward(inp)
        n = torch.sum(out, 1)
        probs = torch.nn.functional.softmax(logprobs, 1)
        pred = n.view(-1, 1) * probs

        # computes mean absolute error.
        mae = torch.mean(torch.abs(out - pred))
        return mae
