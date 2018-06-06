import os
import keras
import pandas as pd
import numpy as np
from biom import load_table, Table
from biom.util import biom_open
from skbio.stats.composition import clr, centralize, closure
from skbio.stats.composition import clr_inv as softmax
import matplotlib.pyplot as plt
from scipy.stats import entropy, spearmanr
from keras.optimizers import SGD, Adam
from keras.layers import (Input, Embedding, Dense,
                          Dropout, Flatten, Activation)
from keras.models import Model
from keras import regularizers
import click
from scipy.sparse import coo_matrix


@click.group()
def multimodal():
    pass

@multimodal.command()
@click.option('--otu-table-file', help='Input otu biom table')
@click.option('--metabolite-table-file', help='Input metabolite biom table')
@click.option('--num_test', default=10,
              help='Number of testing samples')
@click.option('--min_samples',
              help=('Minimum number of samples a feature needs to be '
                    'observed in before getting filtered out'),
              default=10)
@click.option('--output_dir', help='output directory')
def split(otu_table_file, metabolite_table_file, num_test,
          min_samples, output_dir):
    microbes = load_table(otu_table_file)
    metabolites = load_table(metabolite_table_file)

    microbes_df = pd.DataFrame(
        np.array(microbes.matrix_data.todense()).T,
        index=microbes.ids(axis='sample'),
        columns=microbes.ids(axis='observation'))

    metabolites_df = pd.DataFrame(
        np.array(metabolites.matrix_data.todense()).T,
        index=metabolites.ids(axis='sample'),
        columns=metabolites.ids(axis='observation'))

    microbes_df, metabolites_df = microbes_df.align(
        metabolites_df, axis=0, join='inner')

    # filter out microbes that don't appear in many samples
    microbes_df = microbes_df.loc[:, (microbes_df>0).sum(axis=0)>min_samples]

    sample_ids = set(np.random.choice(microbes_df.index, size=num_test))
    sample_ids = np.array([x in sample_ids for x in microbes_df.index])
    train_microbes = microbes_df.loc[~sample_ids]
    test_microbes = microbes_df.loc[sample_ids]
    train_metabolites = metabolites_df.loc[~sample_ids]
    test_metabolites = metabolites_df.loc[sample_ids]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    train_microbes = Table(train_microbes.values.T,
                           train_microbes.columns, train_microbes.index)
    test_microbes = Table(test_microbes.values.T,
                          test_microbes.columns, test_microbes.index)
    train_metabolites = Table(train_metabolites.values.T,
                              train_metabolites.columns, train_metabolites.index)
    test_metabolites = Table(test_metabolites.values.T,
                             test_metabolites.columns, test_metabolites.index)

    # output paths
    test_microbes_path = os.path.join(
        output_dir, 'test_' + os.path.basename(otu_table_file))
    train_microbes_path = os.path.join(
        output_dir, 'train_' + os.path.basename(otu_table_file))
    test_metabolites_path = os.path.join(
        output_dir, 'test_' + os.path.basename(metabolite_table_file))
    train_metabolites_path = os.path.join(
        output_dir, 'train_' + os.path.basename(metabolite_table_file))

    with biom_open(train_microbes_path, 'w') as f:
        train_microbes.to_hdf5(f, "train")
    with biom_open(test_microbes_path, 'w') as f:
        test_microbes.to_hdf5(f, "test")
    with biom_open(train_metabolites_path, 'w') as f:
        train_metabolites.to_hdf5(f, "train")
    with biom_open(test_metabolites_path, 'w') as f:
        test_metabolites.to_hdf5(f, "test")


def onehot(microbes, metabolites):
    """ One hot encoding for microbes.

    Parameters
    ----------
    microbes : np.array
       Table of microbe abundances (counts)
    metabolites : np.array
       Table of metabolite abundances (proportions)

    Returns
    -------
    otu_hits : np.array
       One hot encodings of microbes
    ms_hits : np.array
       Repeated copies of the metabolite abundances
    """
    coo = coo_matrix(microbes)
    data = coo.data.astype(np.int64)
    otu_ids = coo.col
    sample_ids = coo.row
    otu_hits = np.repeat(otu_ids, data)
    sample_ids = np.repeat(sample_ids, data)

    ms_hits = metabolites[sample_ids, :]
    return otu_hits.astype(np.int32), ms_hits


def build_model(microbes, metabolites,
                latent_dim=5, dropout_rate=0.5, lam=0,
                beta_1=0.999, beta_2=0.9999, clipnorm=10.):
    """ Building a model.

    Parameters
    ----------
    microbes : np.array
       Table of microbe abundances (counts)
    metabolites : np.array
       Table of metabolite abundances (proportions)
    """
    d1 = microbes.shape[1]
    d2 = metabolites.shape[1]

    otu_input = Input(shape=(1,), dtype='float32', name='otu_input')
    embedding = Embedding(input_dim=d1, output_dim=latent_dim,
                          input_length=1, name='otu_embedding')(otu_input)
    otu_embed = Flatten()(embedding)

    ms_in = Dense(d2, activation='linear', use_bias=False,
                  activity_regularizer=regularizers.l1(lam),
                  name='ms_in')(otu_embed)
    ms_drop = Dropout(dropout_rate, name='ms_drop')(ms_in)
    ms_output = Activation('softmax', name='ms_output')(ms_drop)

    model = Model(inputs=[otu_input], outputs=[ms_output])
    sgd = Adam(beta_1=beta_1, beta_2=beta_2, clipnorm=clipnorm)
    model.compile(optimizer=sgd,
                  loss={
                      'ms_output': 'kullback_leibler_divergence'
                  }
                 )
    return model


def cross_validation(model, microbes, metabolites, top_N=50):
    """ Running cross validation on test data.

    Parameters
    ----------
    model : keras.model
       Pre-trained model
    microbes : pd.DataFrame
       Microbe abundances (counts) on test dataset
    metabolites : pd.DataFrame
       Metabolite abundances (proportions) on test dataset
    top_N : int
       Number of top hits to evaluate

    Returns
    -------
    params : pd.Series
       List of cross-validation statistics
    """
    otu_hits, ms_hits = onehot(
        microbes.values, closure(metabolites.values))

    batch_size = ms_hits.shape[0]
    res = model.predict(otu_hits, batch_size)
    exp = ms_hits

    ms_r = []
    prec = []
    recall = []
    tps = fps = fns = tns = 0
    ids = set(range(len(metabolites.columns)))
    n, d = res.shape
    rank_stats = []
    for i in range(n):
        exp_names = np.argsort(exp[i, :])[-top_N:]
        res_names = np.argsort(res[i, :])[-top_N:]

        r = spearmanr(exp[i, exp_names],
                      res[i, exp_names])
        ms_r.append(r)

        hits  = set(res_names)
        truth = set(exp_names)

        tps += len(hits & truth)
        fns += len(truth - hits)
        fps += len(hits - truth)
        tns += len((ids - hits) & (ids - truth))

        p = len(hits & truth) / top_N
        r = len(hits & truth) / d
        prec.append(p)
        recall.append(r)
        rank_stats.append(r)

    ms_r = np.mean(ms_r)

    r = np.mean(recall)
    p = np.mean(prec)

    params = pd.Series({
        'TP': tps,
        'FP': fps,
        'FN': fns,
        'TN': tns,
        'precision': np.mean(prec),
        'recall': np.mean(recall),
        'f1_score': 2 * (p * r) / (p + r),
        'meanRK': np.mean(rank_stats)
    })

    return params


@multimodal.command()
@click.option('--otu-train-file',
              help='Input microbial abundances for training')
@click.option('--otu-test-file',
              help='Input microbial abundances for testing')
@click.option('--metabolite-train-file',
              help='Input metabolite abundances for training')
@click.option('--metabolite-test-file',
              help='Input metabolite abundances for testing')
@click.option('--epochs',
              help='Number of epochs to train', default=10)
@click.option('--batch_size',
              help='Size of mini-batch', default=32)
@click.option('--latent_dim',
              help=('Dimensionality of shared latent space. '
                    'This is analogous to the number of PC axes.'),
              default=3)
@click.option('--regularization',
              help=('Parameter regularization.  Helps with preventing overfitting.'
                    'Higher regularization forces more parameters to zero.'),
              default=10.)
@click.option('--dropout-rate',
              help=('Dropout regularization.  Helps with preventing overfitting.'
                    'This is the probability of dropping a parameter at a given iteration.'
                    'Values must be between (0, 1)'),
              default=0.5)
@click.option('--top-k',
              help=('Number of top hits to compare for cross-validation.'),
              default=10)
@click.option('--summary-dir',
              help='Summary directory')
@click.option('--results-file',
              help='Results file containing cross validation results.')
@click.option('--ranks-file',
              help='Ranks file containing microbe-metabolite rankings')
def autoencoder(otu_train_file, otu_test_file,
                metabolite_train_file, metabolite_test_file,
                epochs, batch_size, latent_dim,
                regularization, dropout_rate, top_k,
                summary_dir, results_file, ranks_file):

    lam = regularization

    train_microbes = load_table(otu_train_file)
    test_microbes = load_table(otu_test_file)
    train_metabolites = load_table(metabolite_train_file)
    test_metabolites = load_table(metabolite_test_file)

    microbes_df = pd.DataFrame(
        np.array(train_microbes.matrix_data.todense()).T,
        index=train_microbes.ids(axis='sample'),
        columns=train_microbes.ids(axis='observation'))

    metabolites_df = pd.DataFrame(
        np.array(train_metabolites.matrix_data.todense()).T,
        index=train_metabolites.ids(axis='sample'),
        columns=train_metabolites.ids(axis='observation'))

    # filter out low abundance microbes
    microbe_ids = microbes_df.columns
    metabolite_ids = metabolites_df.columns

    # normalize the microbe and metabolite counts to sum to 1
    microbes = closure(microbes_df)
    metabolites = closure(metabolites_df)
    params = []

    # sname = 'microbe_latent_dim_' + str(microbe_latent_dim) + \
    #        '_metabolite_latent_dim_' + str(metabolite_latent_dim) + \
    #        '_lam' + str(lam)

    otu_hits, ms_hits = onehot(
        microbes_df.values, closure(metabolites_df.values))
    model = build_model(
        microbes, metabolites,
        latent_dim=latent_dim, lam=lam,
        dropout_rate=dropout_rate
    )

    # tbCallBack = keras.callbacks.TensorBoard(
    #     log_dir=os.path.join(summary_dir + '/run_' + sname),
    #     histogram_freq=0,
    #     write_graph=True,
    #     write_images=True)

    model.fit(
        {
            'otu_input': otu_hits,
        },
        {
            'ms_output': ms_hits
        },
        verbose=1,
        #callbacks=[tbCallBack],
        epochs=epochs, batch_size=batch_size)

    microbes_df = pd.DataFrame(
        np.array(test_microbes.matrix_data.todense()).T,
        index=test_microbes.ids(axis='sample'),
        columns=test_microbes.ids(axis='observation'))

    metabolites_df = pd.DataFrame(
        np.array(test_metabolites.matrix_data.todense()).T,
        index=test_metabolites.ids(axis='sample'),
        columns=test_metabolites.ids(axis='observation'))

    otu_hits, ms_hits = onehot(
        microbes_df.values, closure(metabolites_df.values))

    # otu_output, ms_output = model.predict(
    #     [microbes, metabolites], batch_size=microbes.shape[0])
    weights = model.get_weights()
    U, V = weights[0], weights[1]

    ranks = U @ V
    ranks = pd.DataFrame(ranks, index=microbes_df.columns,
                         columns=metabolites_df.columns)
    params = cross_validation(
        model, microbes_df, metabolites_df, top_N=50)
    print(params)

    params.to_csv(results_file)
    ranks.to_csv(ranks_file)


@multimodal.command()
@click.option('--ranks-file',
              help='Ranks file containing microbe-metabolite rankings')
@click.option('--k-nearest-neighbors',
              help=('Number of nearest neighbors.'),
              default=3)
@click.option('--node-metadata',
              help='Node metadata for cytoscape.')
@click.option('--edge-metadata',
              help='Edge metadata for cytoscape.')
def network(ranks_file, k_nearest_neighbors, node_metadata, edge_metadata):
    ranks = pd.read_csv(ranks_file, index_col=0).T
    probs = ranks.apply(softmax, axis=1)
    top_hits = pd.DataFrame(
        {'ms_id': ranks.apply(np.argmin, axis=1),
         'rank': ranks.apply(np.min, axis=1)},
        index=ranks.index)
    k = k_nearest_neighbors
    otus = {x : i for i, x in enumerate(ranks.columns)}

    topk = ranks.apply(lambda x: [
        otus[ranks.columns[k]] for k in np.argsort(x)[-k:]],
                       axis=1).values
    topk = pd.DataFrame([x for x in topk], index=ranks.index)
    top_hits = pd.merge(
        top_hits, topk, left_index=True, right_index=True)
    top_hits = top_hits.reset_index()
    edges = pd.melt(
        top_hits, id_vars=['index'],
        value_vars=list(range(k)),
        value_name='otu_id')
    edges = edges.rename(columns={'index': 'ms_id'})
    edges = edges.rename(columns={'index': 'ms_id'})
    edges['edge_type'] = 'co_occur'
    # edges['ms_id'] = ['metabolite_%s' % x for x in edges.ms_id]
    # edges['otu_id'] = ['otu_%s' % x for x in edges.otu_id]
    edges = edges.set_index(['ms_id'])
    edges[['edge_type', 'otu_id']].to_csv(
        edge_metadata, sep='\t', header=False)

    otu_ids = set(edges.otu_id.values)
    ms_ids = set(edges.index)

    nodes = pd.DataFrame(columns=['id', 'node_type'])
    nodes['id'] = list(ms_ids) + list(otu_ids)
    nodes['node_type'] = ['metabolite'] * len(ms_ids) + ['OTU'] * len(otu_ids)
    nodes = nodes.set_index('id')
    nodes.to_csv(node_metadata, sep='\t')


if __name__ == '__main__':
    multimodal()
