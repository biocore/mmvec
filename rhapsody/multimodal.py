import os
import time
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
    def __init__(self, num_microbes, num_metabolites, latent_dim, device='cpu'):
        super(MMvec, self).__init__()
        self.num_microbes = num_microbes
        self.num_metabolites = num_metabolites
        self.device = device

        # input layer parameter (for the microbes)
        self.embeddings = nn.Embedding(num_microbes, latent_dim)
        self.bias = nn.Embedding(num_microbes, 1)
        self.logstdU = nn.Embedding(num_microbes, latent_dim)
        self.logstdUb = nn.Embedding(num_microbes, 1)

        # output layer parameters (for the metabolites)
        self.muV = Variable(torch.randn(latent_dim, num_metabolites-1).float(),
                            requires_grad=True)
        self.muVb = Variable(torch.randn(1, num_metabolites-1).float(),
                             requires_grad=True)
        self.logstdV = Variable(torch.randn(latent_dim, num_metabolites-1).float(),
                                requires_grad=True)
        self.logstdVb = Variable(torch.randn(1, num_metabolites-1).float(),
                                 requires_grad=True)
        # allocate to devices
        # self.embeddings = self.embeddings.to(device)
        # self.bias = self.bias.to(device)
        # self.logstdU = self.logstdU.to(device)
        # self.logstdUb = self.logstdUb.to(device)
        # self.muV = self.muV.to(device)
        # self.muVb = self.muVb.to(device)
        # self.logstdV = self.logstdV.to(device)
        # self.logstdVb = self.logstdVb.to(device)

    def encode(self, inputs):
        embeds = self.reparameterize(
            self.embeddings(inputs),
            self.logstdU(inputs)
        )
        biases = self.reparameterize(
            self.bias(inputs),
            self.logstdUb(inputs)
        )
        return embeds, biases

    def decode(self, embeds, biases):
        V = self.reparameterize(self.muV, self.logstdV)
        Vb = self.reparameterize(self.muVb, self.logstdVb)
        lam = biases + embeds @ V + Vb
        lam = torch.cat((torch.zeros(biases.shape[0], 1), lam), dim=1)
        m = torch.mean(lam, dim=1)
        log_probs = (lam - m.view(-1, 1))
        return log_probs

    def forward(self, inputs):
        embeds, biases = self.encode(inputs)
        log_probs = self.decode(embeds, biases)
        return log_probs

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def divergence(self, mu, logvar):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        return 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def loss(self, pred, obs):
        d1 = self.divergence(self.embeddings.weight, self.logstdU.weight)
        d2 = self.divergence(self.bias.weight, self.logstdUb.weight)
        d3 = self.divergence(self.muV, self.logstdV)
        d4 = self.divergence(self.muVb, self.logstdVb)
        kld = d1 + d2 + d3 + d4
        total = torch.sum(obs)
        m = Multinomial(logits=pred)
        likelihood = m.log_prob(obs)
        return - torch.mean(kld + likelihood)

    def fit(self, trainX, trainY, epochs=1000, batch_size=100, device='cpu',
            learning_rate=1e-3, beta1=0.8, beta2=0.9):
        optimizer = optim.Adam(self.parameters(), betas=(beta1, beta2),
                               lr=learning_rate)
        self.train()
        num_samples = trainX.shape[0]
        train_loss = 0
        losses = []
        iterations = epochs * trainX.nnz // batch_size
        print(iterations)
        for i in tqdm(range(0, iterations)):
            inp, out = get_batch(trainX, trainY, i % num_samples, batch_size)

            inp = inp.to(device)
            out = out.to(device)
            optimizer.zero_grad()
            pred = self.forward(inp)
            loss = self.loss(pred, out)
            loss.backward()
            train_loss += loss.item()
            losses.append(loss.item())
            optimizer.step()

        return losses

