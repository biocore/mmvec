import pandas as pd

import torch
import torch.nn as nn
import torch.linalg as linalg
import torch.nn.functional as F
from torch.distributions import Multinomial, Normal


class LinearALR(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim - 1)

    def forward(self, x):
        y = self.linear(x)
        z = torch.zeros((y.shape[0], y.shape[1], 1))
        y = torch.cat((z, y), dim=2)

        return F.softmax(y, dim=2)


class MMvecALR(nn.Module):
    def __init__(self, num_microbes, num_metabolites, latent_dim, sigma_u,
            sigma_v):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_microbes = num_microbes
        self.num_metabolites = num_metabolites

        self.u_bias = nn.parameter.Parameter(torch.randn((num_microbes, 1)))

        self.encoder = nn.Embedding(num_microbes, latent_dim)
        self.decoder = LinearALR(latent_dim, num_metabolites)

        self.sigma_u = sigma_u
        self.sigma_v = sigma_v


    def forward(self, X, Y):
        # Three likelihoods, the likelihood of each weight and the likelihood
        # of the data fitting in the way that we thought
        # LYs
        z = self.encoder(X)
        z = z + self.u_bias[X].reshape((*X.shape, 1))
        y_pred = self.decoder(z)

        forward_dist = Multinomial(total_count=0,
                                   validate_args=False,
                                   probs=y_pred)

        forward_dist = forward_dist.log_prob(Y)

        l_y = forward_dist.mean(0).mean()

        u_weights = self.encoder.weight
        l_u = Normal(0, self.sigma_u).log_prob(u_weights).sum()
        l_ubias = Normal(0, self.sigma_u).log_prob(self.u_bias).sum()

        v_weights = self.decoder.linear.weight
        l_v = Normal(0, self.sigma_v).log_prob(v_weights).sum()
        l_vbias = Normal(0, self.sigma_v).log_prob(self.decoder.linear.bias).sum()

        likelihood_sum = l_y + l_u + l_v + l_ubias + l_vbias
        return likelihood_sum

    def get_ordination(self, equalize_biplot=False):
        ranks = self.ranks_matrix - self.ranks_matrix.mean(dim=0)
        u, s, v = linalg.svd(ranks, full_matrices=False)
        print(u)
        print(s)
        print(v)

    def ranks(self, microbe_ids, metabolite_ids):
        U = torch.cat(
            (torch.ones((self.num_microbes, 1)),
            self.u_bias.detach(),
            self.encoder.weight.detach()),
            dim=1)

        V = torch.cat(
            (self.decoder.linear.bias.detach().unsqueeze(dim=0),
             torch.ones((1, self.num_metabolites - 1)),
             self.decoder.linear.weight.detach().T),
            dim=0)

        res = torch.cat((torch.zeros((self.num_microbes, 1)), U @ V), dim=1)
        res = res - res.mean(axis=1).reshape(-1, 1)

        self.ranks_matrix = res
        self.ranks_df = pd.DataFrame(res, index=microbe_ids,
                columns=metabolite_ids)
