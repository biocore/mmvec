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
                 subsample_size=100, mc_samples=10,
                 device='cpu'):
        super(MMvec, self).__init__()
        self.num_microbes = num_microbes
        self.num_metabolites = num_metabolites
        self.num_samples = num_samples
        self.device = device
        self.batch_size = batch_size
        self.subsample_size = subsample_size
        self.mc_samples = mc_samples
        self.microbe_total = microbe_total
        # TODO: enable max norm in embedding to account for
        # scale identifiability
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
        elbo = kld + likelihood
        return -elbo, kld, likelihood

    def fit(self, trainX, trainY, epochs=10, learning_rate=0.1,
            beta1=0.9, beta2=0.99):
        losses = []
        klds = []
        likes = []
        errs = []

        optimizer = optim.Adam(self.parameters(), betas=(beta1, beta2),
                               lr=learning_rate)
        for ep in tqdm(range(0, epochs)):

            self.train()
            for i in range(0, self.num_samples, self.batch_size):
                optimizer.zero_grad()

                inp, out = get_batch(trainX, trainY, i % self.num_samples,
                                     self.subsample_size, self.batch_size)

                pred = self.forward(inp)
                loss, kld, like = self.loss(pred, out)
                metabolite_total = torch.sum(out, 1).view(-1, 1)
                err = torch.mean(
                    torch.abs(F.softmax(pred, dim=1) * metabolite_total - out))
                loss.backward()

                errs.append(err.item())
                losses.append(loss.item())
                klds.append(kld.item())
                likes.append(like.item())

                optimizer.step()

        return losses, klds, likes, errs
