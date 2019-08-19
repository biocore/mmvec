import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from skbio.stats.composition import ilr_inv, closure
from skbio.stats.composition import clr_inv as softmax
from scipy.sparse import coo_matrix
from biom import Table


def random_multimodal(num_microbes=20, num_metabolites=100, num_samples=100,
                      latent_dim=3, low=-1, high=1,
                      microbe_total=10, metabolite_total=100,
                      uB=0, sigmaB=2, sigmaQ=0.1,
                      uU=0, sigmaU=1, uV=0, sigmaV=1,
                      seed=0):
    """ Simulate paired datasets

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
    microbe_counts : biom.Table
       Count table of microbial counts
    metabolite_counts : biom.Table
       Count table of metabolite counts
    """
    state = check_random_state(seed)
    # only have two coefficients
    beta = state.normal(uB, sigmaB, size=(2, num_microbes))

    X = np.vstack((np.ones(num_samples),
                   np.linspace(low, high, num_samples))).T
    microbes = ilr_inv(state.multivariate_normal(
        mean=np.zeros(
            num_microbes - 1), cov=np.diag([sigmaQ] * (num_microbes - 1)),
        size=num_samples)
    )
    Umain = state.normal(
        uU, sigmaU, size=(num_microbes, latent_dim))
    Vmain = state.normal(
        uV, sigmaV, size=(latent_dim, num_metabolites - 1))

    Ubias = state.normal(
        uU, sigmaU, size=(num_microbes, 1))
    Vbias = state.normal(
        uV, sigmaV, size=(1, num_metabolites - 1))

    U_ = np.hstack(
        (np.ones((num_microbes, 1)), Ubias, Umain))
    V_ = np.vstack(
        (Vbias, np.ones((1, num_metabolites - 1)), Vmain))

    phi = np.hstack((np.zeros((num_microbes, 1)), U_ @ V_))
    probs = softmax(phi)
    microbe_counts = np.zeros((num_samples, num_microbes))
    metabolite_counts = np.zeros((num_samples, num_metabolites))
    for n in range(num_samples):
        p = np.zeros(num_metabolites)
        pm = closure(microbes[n, :])
        for mt in range(microbe_total):
            otu = state.choice(np.arange(num_microbes), p=pm, size=1)
            microbe_counts[n, otu] += 1
            p += probs[otu, :].ravel()
        metabolite_counts[n, :] = state.multinomial(
            metabolite_total, closure(p))

    otu_ids = ['OTU_%d' % d for d in range(microbe_counts.shape[1])]
    ms_ids = ['metabolite_%d' % d for d in range(metabolite_counts.shape[1])]
    sample_ids = ['sample_%d' % d for d in range(metabolite_counts.shape[0])]

    # TODO: convert to biom Tables instead of DataFrames
    microbe_counts = Table(
        microbe_counts.T, otu_ids, sample_ids)
    metabolite_counts = Table(
        metabolite_counts.T, ms_ids, sample_ids)

    return (microbe_counts, metabolite_counts, X, beta,
            Umain, Ubias, Vmain, Vbias)


def onehot(microbes):
    """ One hot encoding for microbes.

    Parameters
    ----------
    microbes : np.array
       Table of microbe abundances (counts)

    Returns
    -------
    otu_hits : np.array
       One hot encodings of microbes
    sample_ids : np.array
       Sample ids
    """
    coo = coo_matrix(microbes)
    data = coo.data.astype(np.int64)
    otu_ids = coo.col
    sample_ids = coo.row
    otu_hits = np.repeat(otu_ids, data)
    sample_ids = np.repeat(sample_ids, data)

    return otu_hits.astype(np.int32), sample_ids


def rank_hits(ranks, k, pos=True):
    """ Creates an edge list based on rank matrix.

    Parameters
    ----------
    ranks : pd.DataFrame
       Matrix of ranks (aka conditional probabilities)
    k : int
       Number of nearest neighbors
    pos : bool
       Specifies either most associated or least associated.
       This is a proxy to positively correlated or negatively correlated.

    Returns
    -------
    edges : pd.DataFrame
       List of edges along with corresponding ranks.
    """
    axis = 1

    def sort_f(x):
        if pos:
            return [
                ranks.columns[i] for i in np.argsort(x)[-k:]
            ]
        else:
            return [
                ranks.columns[i] for i in np.argsort(x)[:k]
            ]

    idx = ranks.index
    topk = ranks.apply(sort_f, axis=axis).values
    topk = pd.DataFrame([x for x in topk], index=idx)
    top_hits = topk.reset_index()
    top_hits = top_hits.rename(columns={'index': 'src'})
    edges = pd.melt(
        top_hits, id_vars=['src'],
        var_name='rank',
        value_vars=list(range(k)),
        value_name='dest')

    # fill in actual ranks
    for i in edges.index:
        src = edges.loc[i, 'src']
        dest = edges.loc[i, 'dest']
        edges.loc[i, 'rank'] = ranks.loc[src, dest]
    edges['rank'] = edges['rank'].astype(np.float64)
    return edges


def format_params(vals, colnames, rownames,
                  embed_name, index_name='feature_id'):
    """ Reformats the model parameters in a readable format
    Parameters
    ----------
    vals : np.array
        Values of the model parameters
    colnames : array_like of str
        Column names corresponding to the features.
        These typically correspond to PC axis names.
    rownames : array_like of str
        Row names corresponding to the features.
        These typically correspond to microbe/metabolite names.
    embed_name: str
        Specifies which embedding is being formatted
    index_name : str
        Specifies the index name, since it'll be formatted
        into a qiime2 Metadata format
    Returns
    -------
    pd.DataFrame
        feature_id : str
            Feature names
        axis : str
            PC axis names
        embed_type: str
            Specifies which embedding is being formatted
        values : float
            Corresponding model parameters
    """
    df = pd.DataFrame(vals, columns=colnames, index=rownames)
    df = df.reset_index()
    df = df.rename(columns={'index': 'feature_id'})
    df = pd.melt(df, id_vars=['feature_id'],
                 var_name='axis', value_name='values')

    df['embed_type'] = embed_name

    return df[['feature_id', 'axis', 'embed_type', 'values']]


def alr2clr(x):
    if x.ndim > 1:
        y = np.hstack((np.zeros((x.shape[0], 1)), x))
        y = y - y.mean(axis=1).reshape(-1, 1)
    else:
        y = np.hstack((np.zeros(1), x))
        y = y - y.mean()

    return y
