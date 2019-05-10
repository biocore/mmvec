import os
import time
import copy
from tqdm import tqdm
import pandas as pd
import numpy as np
from skbio.stats.composition import clr_inv as softmax
from scipy.stats import spearmanr
import datetime
from .util import onehot, get_batch

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.multinomial import Multinomial



class MMvec(nn.Module):
    def __init__(self, num_microbes, num_metabolites, latent_dim,
                 batch_size=10, subsample_size=100,
                 device='cpu'):
        super(MMvec, self).__init__()
        self.num_microbes = num_microbes
        self.num_metabolites = num_metabolites
        self.device = device
        self.batch_size = batch_size
        self.subsample_size = subsample_size

        # input layer parameter (for the microbes)
        self.embeddings = nn.Embedding(num_microbes, latent_dim).to(device)
        self.bias = nn.Embedding(num_microbes, 1).to(device)
        self.logstdU = nn.Embedding(num_microbes, latent_dim).to(device)
        self.logstdUb = nn.Embedding(num_microbes, 1).to(device)

        # output layer parameters (for the metabolites)
        self.muV = Variable(torch.randn(latent_dim, num_metabolites-1, device=device).float(),
                            requires_grad=True)
        self.muVb = Variable(torch.randn(1, num_metabolites-1, device=device).float(),
                             requires_grad=True)
        self.logstdV = Variable(torch.randn(latent_dim, num_metabolites-1, device=device).float(),
                                requires_grad=True)
        self.logstdVb = Variable(torch.randn(1, num_metabolites-1, device=device).float(),
                                 requires_grad=True)

    def encode(self, inputs):
        """ Transforms inputs into lower dimensional space"""
        embeds = self.reparameterize(
            self.embeddings(inputs),
            self.logstdU(inputs)
        )
        biases = self.reparameterize(
            self.bias(inputs),
            self.logstdUb(inputs)
        )
        return embeds, biases

    def forward(self, inputs):
        """ Predicts output abundances """

        embeds, biases = self.encode(inputs)

        V = self.reparameterize(self.muV, self.logstdV)
        Vb = self.reparameterize(self.muVb, self.logstdVb)
        lam = biases + embeds @ V + Vb
        zeros = torch.zeros(self.batch_size * self.subsample_size, 1, device=self.device)

        lam = torch.cat((zeros, lam), dim=1)
        m = torch.mean(lam, dim=1)
        log_probs = (lam - m.view(-1, 1))
        return log_probs

    def reparameterize(self, mu, logvar):
        """ Samples from the posterior distribution via
        reparameterization gradients"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=self.device)
        return mu + eps*std

    def divergence(self, mu, logvar):
        """ Computes the KL divergence between posterior and prior. """
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        return 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def loss(self, pred, obs):
        """ Computes the loss function to be minimized. """
        d1 = self.divergence(self.embeddings.weight, self.logstdU.weight)
        d2 = self.divergence(self.bias.weight, self.logstdUb.weight)
        d3 = self.divergence(self.muV, self.logstdV)
        d4 = self.divergence(self.muVb, self.logstdVb)
        kld = d1 + d2 + d3 + d4
        total = torch.sum(obs)
        likelihood = Multinomial(logits=pred).log_prob(obs)
        return - torch.mean(kld + likelihood)

    def fit(self, trainX, trainY, epochs=1000,
            learning_rate=1e-3, mc_samples=5,
            beta1=0.8, beta2=0.9, gamma=0.1, step_size=1):
        """ Train the actual model

        Parameters
        ----------
        trainX : scipy.sparse.csr
            Input data (samples x features)
        trainY : np.array
            Output data (samples x features)
        epochs : int
            Number of training iterations over the entire dataset
        batch_size : int
            Number of samples to train per iteration
        beta2 : float
            Second momentum constant for ADAM gradient descent Values
            can only be between (0, 1). Values close to 1 indicate
            sparse updates.
        gamma : float
            Percentage decrease of the learning rate per scheduler step.
        step_size: int
            Number of epochs before the scheduler step is incremented.
        """

        num_samples = trainY.shape[0]

        optimizer = optim.Adam(self.parameters(), betas=(beta1, beta2),
                               lr=learning_rate)
        #scheduler = torch.optim.lr_scheduler.StepLR(
        #    optimizer, step_size=step_size, gamma=gamma)


        best_loss = np.inf

        losses = []
        for ep in tqdm(range(0, epochs)):

            self.train()
            #scheduler.step()
            for i in range(0, num_samples, self.batch_size):
                optimizer.zero_grad()

                inp, out = get_batch(trainX, trainY, i % num_samples,
                                     self.subsample_size, self.batch_size)
                inp = inp.to(device=self.device)
                out = out.to(device=self.device)
                mean_loss = torch.zeros(mc_samples, device=self.device)
                # allow for MC sampling
                for j in range(mc_samples):
                    pred = self.forward(inp)
                    loss = self.loss(pred, out)
                    mean_loss[j] = loss

                mean_loss = torch.mean(mean_loss)
                mean_loss.backward()
                ml = mean_loss.item()
                losses.append(ml)
                # remember the best model
                if ml < best_loss:
                    best_self = copy.deepcopy(self)
                    best_loss = ml
                optimizer.step()

            #print('epoch:', ep, 'loss:', ml, 'lr:', scheduler.get_lr())

        return best_self, losses
