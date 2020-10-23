import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from skbio.stats.composition import ilr_inv
from skbio.stats.composition import clr_inv as softmax


def random_multimodal(num_microbes=20, num_metabolites=100, num_samples=100,
                      latent_dim=3, low=-1, high=1,
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
    probs = softmax(phi)
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


def split_tables(otu_table, metabolite_table,
                 metadata=None, training_column=None, num_test=10,
                 min_samples=10):
    """ Splits otu and metabolite tables into training and testing datasets.

    Parameters
    ----------
    otu_table : biom.Table
       Table of microbe abundances
    metabolite_table : biom.Table
       Table of metabolite intensities
    metadata : pd.DataFrame
       DataFrame of sample metadata information.  This is primarily used
       to indicated training and testing samples
    training_column : str
       The column used to indicate training and testing samples.
       Samples labeled 'Train' are allocated to the training set.
       All other samples are placed in the testing dataset.
    num_test : int
       If metadata or training_column is not specified, then `num_test`
       indicates how many testing samples will be allocated for
       cross validation.
    min_samples : int
       The minimum number of samples a microbe needs to be observed in
       in order to not get filtered out
    Returns
    -------
    train_microbes : pd.DataFrame
       Training set of microbes
    test_microbes : pd.DataFrame
       Testing set of microbes
    train_metabolites : pd.DataFrame
       Training set of metabolites
    test_metabolites : pd.DataFrame
       Testing set of metabolites
    Notes
    -----
    There is an inefficient conversion from a sparse matrix to a
    dense matrix.  This may become a bottleneck later.
    """
    microbes_df = otu_table.to_dataframe().T
    metabolites_df = metabolite_table.to_dataframe().T

    microbes_df, metabolites_df = microbes_df.align(
        metabolites_df, axis=0, join='inner'
    )

    # filter out microbes that don't appear in many samples
    idx = (microbes_df > 0).sum(axis=0) >= min_samples
    microbes_df = microbes_df.loc[:, idx]
    if metadata is None or training_column is None:
        sample_ids = set(np.random.choice(microbes_df.index, size=num_test))
        sample_ids = np.array([(x in sample_ids) for x in microbes_df.index])
    else:
        if len(set(metadata[training_column]) & {'Train', 'Test'}) == 0:
            raise ValueError(
                "Training column must only specify `Train` and `Test` values"
            )
        idx = metadata.loc[metadata[training_column] != 'Train'].index
        sample_ids = set(idx)
        sample_ids = np.array([(x in sample_ids) for x in microbes_df.index])

    train_microbes = microbes_df.loc[~sample_ids]
    test_microbes = microbes_df.loc[sample_ids]
    train_metabolites = metabolites_df.loc[~sample_ids]
    test_metabolites = metabolites_df.loc[sample_ids]
    if len(train_microbes) == 0 or len(train_microbes.columns) == 0:
        raise ValueError('All of the training data has been filtered out. '
                         'Adjust the `--min-feature-count` accordingly.')
    return train_microbes, test_microbes, train_metabolites, test_metabolites


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


def embeddings2ranks(embeddings):
    """ Converts embeddings to ranks"""
    microbes = embeddings.loc[embeddings.embed_type == 'microbe']
    metabolites = embeddings.loc[embeddings.embed_type == 'metabolite']

    U = microbes.pivot(index='feature_id', columns='axis', values='values')
    V = metabolites.pivot(index='feature_id', columns='axis', values='values')
    pc_ids = sorted(list(set(U.columns) - {'bias'}))
    U['ones'] = 1
    V['ones'] = 1
    ranks = U[pc_ids + ['ones', 'bias']] @ V[pc_ids + ['bias', 'ones']].T
    # center each row
    ranks = ranks - ranks.mean(axis=1).values.reshape(-1, 1)
    return ranks


def alr2clr(x):
    if x.ndim > 1:
        y = np.hstack((np.zeros((x.shape[1], 1)), x))
        y = y - y.mean(axis=1).reshape(-1, 1)
    else:
        y = np.hstack((np.zeros(1), x))
        y = y - y.mean()

    return y
