import torch
import numpy as np
import pandas as pd
from skbio.stats.composition import ilr_inv
from skbio.stats.composition import clr_inv as softmax
from scipy.sparse import coo_matrix
import numbers


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Note
    ----
    This is directly from sklearn
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


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
    idx = (microbes_df > 0).sum(axis=0) > min_samples
    microbes_df = microbes_df.loc[:, idx]
    if metadata is None or training_column is None:
        sample_ids = set(np.random.choice(microbes_df.index, size=num_test))
        sample_ids = np.array([(x in sample_ids) for x in microbes_df.index])
    else:
        idx = metadata.loc[metadata[training_column] != 'Train'].index
        sample_ids = set(idx)
        sample_ids = np.array([(x in sample_ids) for x in microbes_df.index])

    train_microbes = microbes_df.loc[~sample_ids]
    test_microbes = microbes_df.loc[sample_ids]
    train_metabolites = metabolites_df.loc[~sample_ids]
    test_metabolites = metabolites_df.loc[sample_ids]

    return train_microbes, test_microbes, train_metabolites, test_metabolites


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
    coo = microbes.tocoo()
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



def get_batch(X, Y, i, subsample_size, batch_size):
    """ Retrieves minibatch

    Parameters
    ----------
    X : scipy.sparse.csr_matrix
        Input sparse matrix of abundances (samples x features)
    Y : np.array
        Output dense matrix of abundances (samples x features)
    i : int
        Sample index
    subsample_size : int
        Number of sequences to randomly draw per iteration
    batch_size : int
        Number of samples to load per minibatch

    TODO
    ----
    - It will be worth offloading this work to the torch.data.DataLoader class.
      One advantage is that the GPU work can be done in parallel
      to preparing the data for transfer.  This could yield at least
      a 2x speedup.
    """
    Xs = []
    Ys = []
    for n in range(batch_size):
        k = (i + n) % Y.shape[0]
        row = X.getrow(k)
        cnts = np.hstack([row.data[row.indptr[i]:row.indptr[i+1]]
                          for i in range(len(row.indptr)-1)])
        feats = np.hstack([row.indices[row.indptr[i]:row.indptr[i+1]]
                           for i in range(len(row.indptr)-1)])
        inp = np.random.choice(feats, p=cnts/np.sum(cnts), size=subsample_size)
        Xs.append(inp)
        Ys.append(Y[k])

    Xs = np.hstack(Xs)
    Ys = np.repeat(np.vstack(Ys), subsample_size, axis=0)
    return torch.from_numpy(Xs).long(), torch.from_numpy(Ys).float()


def format_params(mu, std, colnames, rownames,
                  embed_name, index_name='feature_id'):
    mudf = pd.DataFrame(mu, columns=colnames, index=rownames)
    mudf = mudf.reset_index()
    mudf = mudf.rename(columns={'index': 'feature_id'})
    mudf = pd.melt(mudf, id_vars=['feature_id'],
                   var_name='axis', value_name='mean')

    stddf = pd.DataFrame(std, columns=colnames, index=rownames)
    stddf = stddf.reset_index()
    stddf = stddf.rename(columns={'index': 'feature_id'})
    stddf = pd.melt(stddf, id_vars=['feature_id'],
                    var_name='axis', value_name='stddev')
    df = pd.merge(mudf, stddf, on=['feature_id', 'axis'])
    df['embed_type'] = embed_name

    return df[['feature_id', 'axis', 'embed_type', 'mean', 'stddev']]
