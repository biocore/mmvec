
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Multinomial, Normal

import numpy as np

from gneiss.cluster import random_linkage
from gneiss.balances import sparse_balance_basis



class MMvecILR(nn.Module):
    def __init__(self, num_microbes, num_metabolites, latent_dim, sigma_u,
            sigma_v):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_microbes = num_microbes
        self.num_metabolites = num_metabolites

        self.u_bias = nn.parameter.Parameter(torch.randn((num_microbes, 1)))

        self.encoder = nn.Embedding(num_microbes, latent_dim)
        #self.decoder =  nn.Sequential(
        #        nn.Linear(latent_dim, num_metabolites),
        #        nn.Softmax(dim=2)
        #        )
        self.decoder = LinearILR(latent_dim, num_metabolites)

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

        # LU
        u_weights = self.encoder.weight
        l_u = Normal(0, self.sigma_u).log_prob(u_weights).sum()
        #l_u = torch.normal(0, self.sigma_u).log_prob(z

        # LV
        # index zero currently holds "linear", may need to be changed later
        v_weights = self.decoder.linear.weight
        l_v = Normal(0, self.sigma_v).log_prob(v_weights).sum()

        likelihood_sum = l_y + l_u + l_v
        return likelihood_sum



    def ILRranks(self):
        #modelU = np.hstack(
        #    (np.ones((self.num_microbes, 1)),
        #    self.u_bias.detach().numpy(),
        #    self.encoder.weight.detach().numpy()))

        U = torch.cat(
            (torch.from_numpy(np.ones((self.num_microbes, 1))),
            self.u_bias.detach(),
            self.encoder.weight.detach()),
            dim=-1)

        V = torch.stack(
            (self.decoder.linear.bias.detach(),
             torch.from_numpy(np.ones((1, self.num_metabolites))),
             self.decoder.linear.weight.detach().T),
            dim=0)

        #V = torch.sparse.mm(modelV, self.decoder.Psi.T)
        #res = modelU  V
        res = U @ V @ self.decoder.Psi.to_dense().T
        #res = modelU @ modelV @ self.decoder.Psi.T
        #print(res)
        #res = res - res.mean(axis=1).reshape(-1, 1)
        return res


class LinearILR(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        tree = random_linkage(output_dim)  # pick random tree it doesn't really matter tbh
        basis = sparse_balance_basis(tree)[0].copy()
        indices = np.vstack((basis.row, basis.col))
        Psi = torch.sparse_coo_tensor(
            indices.copy(),
            basis.data.astype(np.float32).copy(),
            dtype=torch.double,
            requires_grad=False).coalesce()

        self.linear = nn.Linear(input_dim, output_dim)
        self.register_buffer('Psi', Psi)

    def forward(self, x):
        y = self.linear(x)
        logy = (Psi.t() @ y.t()).t()
        return F.softmax(logy, dim=1)

