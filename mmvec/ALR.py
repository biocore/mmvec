from numpy import float64
import pandas as pd

import torch
from torch import linalg
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Multinomial, Normal

from skbio import OrdinationResults


def structure_data(microbes, metabolites, holdout_num):
    if type(microbes) is not pd.core.frame.DataFrame:
        microbes = microbes.to_dataframe().T
    if type(metabolites) is not pd.core.frame.DataFrame:
        metabolites = metabolites.to_dataframe().T
   # idx = microbes.index.intersection(metabolites.index)
   # microbes = microbes.loc[idx]
   # metabolites = metabolites.loc[idx]
   #make sure none sum to zero
   #microbes = microbes.loc[:, microbes.sum(axis=0) > 0]
   # microbes = microbes.loc[microbes.sum(axis=1) > 0]

    microbe_idx = microbes.columns
    metabolite_idx = metabolites.columns

    microbes_train = torch.tensor(microbes[:-holdout_num].values,
                                  dtype=torch.float64)
    metabolites_train = torch.tensor(metabolites[:-holdout_num].values,
                                     dtype=torch.float64)

    microbes_test = torch.tensor(microbes[-holdout_num:].values,
                                 dtype=torch.float64)
    metabolites_test = torch.tensor(metabolites[-holdout_num:].values,
                                    dtype=torch.float64)
    microbe_count = microbes_train.shape[1]
    metabolite_count = metabolites_train.shape[1]

    microbes = torch.tensor(microbes.values, dtype=torch.int)
    metabolites = torch.tensor(metabolites.values, dtype=torch.int64)

    nnz = torch.count_nonzero(microbes).item()

    return (microbes_train, microbes_test, metabolites_train,
            metabolites_test, microbe_idx, metabolite_idx, microbe_count,
            metabolite_count,  nnz)


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
    def __init__(self, microbes, metabolites, latent_dim, sigma_u=1,
                 sigma_v=1, holdout_num=4):
        super().__init__()

        # Data setup
        (self.microbes_train, self.microbes_test, self.metabolites_train,
                self.metabolites_test, self.microbe_idx, self.metabolite_idx,
                self.num_microbes, self.num_metabolites,
                 self.nnz) = structure_data(
                    microbes, metabolites, holdout_num)

        self.sigma_u = sigma_u
        self.sigma_v = sigma_v
        self.latent_dim = latent_dim
        # TODO: intialize same way as linear bias
        self.encoder_bias = nn.parameter.Parameter(
                torch.randn((self.num_microbes, 1)))

        self.encoder = nn.Embedding(self.num_microbes, self.latent_dim)
        self.decoder = LinearALR(self.latent_dim, self.num_metabolites)


    def forward(self, X, Y):
        # Three likelihoods, the likelihood of each weight and the likelihood
        # of the data fitting in the way that we thought
        # LYs
        z = self.encoder(X)
        z = z + self.encoder_bias[X].reshape((*X.shape, 1))
        y_pred = self.decoder(z)

        predicted = torch.distributions.multinomial.Multinomial(total_count=0,
                                  validate_args=False,
                                  probs=y_pred)

        data_likelihood = predicted.log_prob(Y)

        l_y = data_likelihood.sum(0).mean()

        u_weights = self.encoder.weight
        l_u = Normal(0, self.sigma_u).log_prob(u_weights).sum()
        l_ubias = Normal(0, self.sigma_u).log_prob(self.encoder_bias).sum()

        v_weights = self.decoder.linear.weight
        l_v = Normal(0, self.sigma_v).log_prob(v_weights).sum()
        l_vbias = Normal(0, self.sigma_v).log_prob(self.decoder.linear.bias).sum()

        likelihood_sum = l_y + l_u + l_v + l_ubias + l_vbias
        return likelihood_sum

    def get_ordination(self, equalize_biplot=False):
        biplot = get_ordination_bare(self.ranks(), self.microbe_idx,
                self.metabolite_idx, equalize_biplot=False)
        return biplot

#        ranks = self.ranks()
#        ranks = ranks - ranks.mean(dim=0)
#
#        u, s_diag, v = linalg.svd(ranks, full_matrices=False)
#
#
#        # us torch.diag to go from vector to matrix with the vector on dia
#        if equalize_biplot:
#            #microbe_embed = u @ torch.sqrt(
#            #        torch.diag(s_diag)).detach().numpy()
#            microbe_embed = u @ torch.sqrt(
#                    torch.diag(s_diag))
#            microbe_embed = microbe_embed.detach().numpy()
#            #metabolite_embed = v.T @ torch.sqrt(s_diag).detach().numpy()
#            metabolite_embed = v.T @ torch.sqrt(s_diag)
#            metabolite_embed = metabolite_embed.detach().numpy()
#        else:
#            microbe_embed = u @ torch.diag(s_diag)
#            microbe_embed = microbe_embed.detach().numpy()
#            metabolite_embed = v.T
#            metabolite_embed = metabolite_embed.detach().numpy()
#
#        pc_ids = ['PC%d' % i for i in range(microbe_embed.shape[1])]
#
#
#        features = pd.DataFrame(
#                microbe_embed, columns=pc_ids, index=self.microbe_idx)
#
#        samples = pd.DataFrame(metabolite_embed, columns=pc_ids,
#                index=self.metabolite_idx)
#
#        short_method_name = 'mmvec biplot'
#        long_method_name = 'Multiomics mmvec biplot'
#        eigvals = pd.Series(s_diag, index=pc_ids)
#        proportion_explained = pd.Series(
#                torch.square(s_diag).detach().numpy() / torch.sum(
#                    torch.square(s_diag)).detach().numpy(),
#                index=pc_ids, dtype=float64)
#
#        biplot = OrdinationResults(
#            short_method_name, long_method_name, eigvals,
#            samples=samples, features=features,
#            proportion_explained=proportion_explained)
#
#        return biplot
#


    @property
    def u_bias(self):
        #ensure consistent access
        return self.encoder_bias.detach()

    @property
    def v_bias(self):
        #ensure consistent access
        return self.decoder.linear.bias.detach()

    @property
    def U(self):
        U = torch.cat(
            (torch.ones((self.encoder.weight.shape[0], 1)),
            self.u_bias,
            self.encoder.weight.detach()),
            dim=1)
        return U

    @property
    def V(self):
        V = torch.cat(
            (self.v_bias.unsqueeze(dim=0),
             torch.ones((1, self.decoder.linear.weight.shape[0] )),
             self.decoder.linear.weight.detach().T),
            dim=0)
        return V

    def ranks_dataframe(self):
        return pd.DataFrame(self.ranks(), index=self.microbe_idx,
                            columns=self.metabolite_idx)

    def ranks(self):
        return ranks_bare(self.U, self.V)
    #def ranks(self):
    #    # Adding the zeros is part of the inverse ALR.
    #    res = torch.cat((
    #            torch.zeros((self.num_microbes, 1)),
    #            self.U @ self.V
    #        ), dim=1)
    #    res = res - res.mean(axis=1).reshape(-1, 1)
    #    return res

    def microbe_relative_freq(self,microbes):
        return (microbes.T / microbes.sum(1)).T

    #def loss_fn(self, y_pred, observed):

    #    predicted = torch.distributions.multinomial.Multinomial(total_count=0,
    #                              validate_args=False,
    #                              probs=y_pred)
    #
    #    data_likelihood = predicted.log_prob(observed)
    #    l_y = data_likelihood.sum(1).mean()
### bare functions for testing/method creation

def ranks_bare(u, v):
    # Adding the zeros is part of the inverse ALR.
    res = torch.cat((
            torch.zeros((u.shape[0], 1)),
            u @ v
        ), dim=1)
    res = res - res.mean(axis=1).reshape(-1, 1)
    return res

def get_ordination_bare(ranks, microbe_index, metabolite_index,
                   equalize_biplot=False):

    ranks = ranks - ranks.mean(dim=0)

    u, s_diag, v = linalg.svd(ranks, full_matrices=False)


    # us torch.diag to go from vector to matrix with the vector on dia
    if equalize_biplot:
        microbe_embed = u @ torch.sqrt(
                torch.diag(s_diag))
        microbe_embed = microbe_embed.detach().numpy()
        metabolite_embed = v.T @ torch.sqrt(s_diag)
        metabolite_embed = metabolite_embed.detach().numpy()
    else:
        microbe_embed = u @ torch.diag(s_diag)
        microbe_embed = microbe_embed.detach().numpy()
        metabolite_embed = v.T
        metabolite_embed = metabolite_embed.detach().numpy()

    pc_ids = ['PC%d' % i for i in range(microbe_embed.shape[1])]


    features = pd.DataFrame(
            microbe_embed, columns=pc_ids, index=microbe_index)

    samples = pd.DataFrame(metabolite_embed, columns=pc_ids,
            index=metabolite_index)

    short_method_name = 'mmvec biplot'
    long_method_name = 'Multiomics mmvec biplot'
    eigvals = pd.Series(s_diag, index=pc_ids)
    proportion_explained = pd.Series(
            torch.square(s_diag).detach().numpy() / torch.sum(
                torch.square(s_diag)).detach().numpy(),
            index=pc_ids, dtype=float64)

    biplot = OrdinationResults(
        short_method_name, long_method_name, eigvals,
        samples=samples, features=features,
        proportion_explained=proportion_explained)

    return biplot

