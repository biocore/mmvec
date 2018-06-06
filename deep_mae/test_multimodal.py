import unittest
import numpy as np
import pandas as pd
from skbio.stats.composition import clr_inv as softmax
from skbio.stats.composition import closure
from sklearn.utils import check_random_state
from deep_mae.multimodal import onehot
from skbio.util import get_data_path
import numpy.testing as npt


def random_multimodal(num_microbes=20, num_metabolites=100, num_samples=100,
                      num_latent=5, low=-1, high=1,
                      microbe_total=10, metabolite_total=100,
                      uB=0, sigmaB=2, sigmaQ=0.1,
                      uU=0, sigmaU=1, uV=0, sigmaV=1,
                      seed=0):
    """
    Parameters
    ----------
    num_microbes : int
       Number of microbial species to simulate
    num_metabolites : int
       Number of molecules to simulate
    num_samples : int
       Number of samples to generate
    num_latent_microbes :
       Number of latent microbial dimensions
    num_latent_metabolites
       Number of latent metabolite dimensions
    num_latent_shared
       Number of dimensions in shared representation
    low : float
       Lower bound of gradient
    high : float
       Upper bound of gradient
    microbe_total : int
       Total number of microbial species
    metabolite_total : int
       Total number of metabolite species
    uB : float
       Mean of regression coefficient distribution
    sigmaB : float
       Standard deviation of regression coefficient distribution
    sigmaQ : float
       Standard deviation of error distribution
    uU : float
       Mean of microbial input projection coefficient distribution
    sigmaU : float
       Standard deviation of microbial input projection
       coefficient distribution
    uV : float
       Mean of metabolite input projection coefficient distribution
    sigmaU : float
       Standard deviation of metabolite input projection
       coefficient distribution
    seed : float
       Random seed

    Returns
    -------
    microbe_counts : pd.DataFrame
       Count table of microbial counts
    metabolite_counts : pd.DataFrame
       Count table of metabolite counts
    """
    state = check_random_state(seed)
    # only have two coefficients
    beta = state.normal(uB, sigmaB, size=(2, num_microbes))

    X = np.vstack((np.ones(num_samples),
                   np.linspace(low, high, num_samples))).T

    microbes = softmax(state.normal(X @ beta, sigmaQ))

    U = state.normal(
        uU, sigmaU, size=(num_microbes, num_latent))
    V = state.normal(
        uV, sigmaV, size=(num_latent, num_metabolites))

    probs = softmax(U @ V)
    microbe_counts = np.zeros((num_samples, num_microbes))
    metabolite_counts = np.zeros((num_samples, num_metabolites))
    n1 = microbe_total
    n2 = metabolite_total // n1
    for n in range(num_samples):
        otu = state.multinomial(n1, microbes[n, :])
        for i in range(num_microbes):
            ms = state.multinomial(otu[i] * n2, probs[i, :])
            metabolite_counts[n, :] += ms
        microbe_counts[n, :] += otu

    otu_ids = ['OTU_%d' % d for d in range(microbe_counts.shape[1])]
    ms_ids = ['metabolite_%d' % d for d in range(metabolite_counts.shape[1])]
    sample_ids = ['sample_%d' % d for d in range(metabolite_counts.shape[0])]

    microbe_counts = pd.DataFrame(
        microbe_counts, index=sample_ids, columns=otu_ids)
    metabolite_counts = pd.DataFrame(
        metabolite_counts, index=sample_ids, columns=ms_ids)

    return microbe_counts, metabolite_counts, X, beta, U, V


class TestMultimodal(unittest.TestCase):
    def setUp(self):
        seed = 0
        # build small simulation
        res = random_multimodal(
            uB=-5,
            num_microbes=2, num_metabolites=4, num_samples=10,
            num_latent=2, low=-1, high=1,
            microbe_total=10, metabolite_total=10,
            seed=seed
        )
        self.microbes, self.metabolites, self.X, self.B, self.U, self.V = res

    def test_onehot(self):
        otu_hits, ms_hits = onehot(self.microbes.values,
                                   closure(self.metabolites.values))
        npt.assert_allclose(ms_hits, closure(ms_hits))

        exp_ms_hits = np.loadtxt(get_data_path('ms_hits.txt'))
        exp_otu_hits = np.loadtxt(get_data_path('otu_hits.txt'))
        npt.assert_allclose(exp_ms_hits, ms_hits)
        npt.assert_allclose(exp_otu_hits, otu_hits)


    def test_build_model(self):
        epochs = 3
        batch_size = 100
        pass

    def test_cross_validate(self):
        pass


if __name__ == "__main__":
    unittest.main()
