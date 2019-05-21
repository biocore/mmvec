import numpy as np
import pandas as pd
from skbio.stats.composition import clr_inv, ilr_inv
from rhapsody.util import check_random_state


def bimodal(num_x, num_y, means=(-1, 1), sigma=0.1, seed=None):
    """ Creates a bimodal matrix distribution.

    num_x : int
        Number of rows in the resulting matrix
    num_y : int
        Number of columns in the resulting matrix.
    means : tuple of float
        Means for each normal
    sigma : float
        Standard deviation for each normal

    Returns
    -------
    Q : np.array
        The matrix of abundances.
    """
    state = check_random_state(seed)
    beta = state.normal(0, 1, size=(2, num_y))
    beta = np.sort(beta, axis=1)
    x = np.linspace(means[0], means[1], num_x)
    X = np.vstack((np.ones(num_x), x)).T
    Q = np.tanh(state.normal(X @ beta, sigma))
    return Q, x


def random_bimodal(num_microbes=20, num_metabolites=100, num_samples=100,
                   latent_dim=3, means=(-3, 3),
                   microbe_total=10, metabolite_total=100,
                   uU=0, sigmaU=1, uV=0, sigmaV=1,
                   eps=0.2, seed=0):
    """ Generates two random matrices that are conditionally linked

    Parameters
    ----------
    num_microbes : int
       Number of microbial species to simulate
    num_metabolites : int
       Number of molecules to simulate
    num_samples : int
       Number of samples to generate
    latent_dim :
       Number of latent dimensions
    means : tuple of float
        Means for each normal
    microbe_total : int
       Total number of microbial species
    metabolite_total : int
       Total number of metabolite species
    sigmaQ : float
       Standard deviation of error distribution
    uU : float
       Mean of microbial input projection coefficient distribution
    sigmaU : float
       Standard deviation of microbial input projection
       coefficient distribution
    uV : float
       Mean of metabolite output projection coefficient distribution
    sigmaV : float
       Standard deviation of metabolite output projection
       coefficient distribution
    eps : float
       Uncertainity for factor estimates
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
    microbes = bimodal(num_samples, num_microbes, means=means)[0]
    microbes = clr_inv(microbes)

    eUmain = bimodal(num_microbes, latent_dim, means=means)[0]
    eVmain = bimodal(num_metabolites, latent_dim, means=means)[0].T
    # center around zero
    eUmain = - eUmain.mean(axis=0) + eUmain - \
        eUmain.mean(axis=1).reshape(-1, 1)
    eVmain = - eVmain.mean(axis=0) + eVmain - \
        eVmain.mean(axis=1).reshape(-1, 1)

    eUbias = state.normal(
        uU, sigmaU, size=(num_microbes, 1))
    eVbias = state.normal(
        uV, sigmaV, size=(1, num_metabolites))

    microbe_counts = np.zeros((num_samples, num_microbes))
    metabolite_counts = np.zeros((num_samples, num_metabolites))
    n2 = metabolite_total // microbe_total
    for n in range(num_samples):
        u = state.normal(eUmain, eps)
        v = state.normal(eVmain, eps)
        ubias = state.normal(eUbias, eps)
        vbias = state.normal(eVbias, eps)
        u_ = np.hstack(
            (np.ones((num_microbes, 1)), ubias, u))
        v_ = np.vstack(
            (vbias, np.ones((1, num_metabolites)), v))
        probs = clr_inv(u_ @ v_)
        for _ in range(microbe_total):
            i = state.choice(np.arange(num_microbes), p=microbes[n, :])
            metabolite_counts[n, :] += state.multinomial(n2, probs[i, :])
        microbe_counts[n, :] += state.multinomial(microbe_total, microbes[n])
    return microbe_counts, metabolite_counts, eUmain, eVmain, eUbias, eVbias


def random_multimodal(num_microbes=20, num_metabolites=100, num_samples=100,
                      latent_dim=3, low=-1, high=1,
                      microbe_total=10, metabolite_total=100,
                      uB=0, sigmaB=2, sigmaQ=0.1,
                      uU=0, sigmaU=1, uV=0, sigmaV=1,
                      seed=0):
    """ Generic simulation for random matrix generation.

    Parameters
    ----------
    num_microbes : int
       Number of microbial species to simulate
    num_metabolites : int
       Number of molecules to simulate
    num_samples : int
       Number of samples to generate
    latent_dim :
       Number of latent dimensions
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
       Mean of metabolite output projection coefficient distribution
    sigmaV : float
       Standard deviation of metabolite output projection
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
    microbes = ilr_inv(state.multivariate_normal(
        mean=np.zeros(num_microbes-1), cov=np.diag([sigmaQ]*(num_microbes-1)),
        size=num_samples)
    )
    Umain = state.normal(
        uU, sigmaU, size=(num_microbes, latent_dim))
    Vmain = state.normal(
        uV, sigmaV, size=(latent_dim, num_metabolites-1))

    Ubias = state.normal(
        uU, sigmaU, size=(num_microbes, 1))
    Vbias = state.normal(
        uV, sigmaV, size=(1, num_metabolites-1))

    U_ = np.hstack(
        (np.ones((num_microbes, 1)), Ubias, Umain))
    V_ = np.vstack(
        (Vbias, np.ones((1, num_metabolites-1)), Vmain))

    phi = np.hstack((np.zeros((num_microbes, 1)), U_ @ V_))
    probs = clr_inv(phi)
    microbe_counts = np.zeros((num_samples, num_microbes))
    metabolite_counts = np.zeros((num_samples, num_metabolites))
    n1 = microbe_total
    n2 = metabolite_total // microbe_total
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

    return (microbe_counts, metabolite_counts, X, beta,
            Umain, Ubias, Vmain, Vbias)
