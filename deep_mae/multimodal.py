import os
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
from biom import load_table, Table
from biom.util import biom_open
from skbio.stats.composition import clr, centralize, closure
from skbio.stats.composition import clr_inv as softmax
import matplotlib.pyplot as plt
from scipy.stats import entropy, spearmanr
import click
from scipy.sparse import coo_matrix
import tensorflow as tf
from tensorflow.contrib.distributions import Multinomial, Normal
import datetime


@click.group()
def multimodal():
    pass

@multimodal.command()
@click.option('--otu-table-file', help='Input otu biom table')
@click.option('--metabolite-table-file', help='Input metabolite biom table')
@click.option('--metadata-file', default=None, help='Sample metadata file')
@click.option('--metadata-column', default=None, help='Sample metadata category column')
@click.option('--num_test', default=10,
              help='Number of testing samples')
@click.option('--min_samples',
              help=('Minimum number of samples a feature needs to be '
                    'observed in before getting filtered out'),
              default=10)
@click.option('--output_dir', help='output directory')
def split(otu_table_file, metabolite_table_file,
          metadata_file, metadata_column, num_test,
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
    if metadata_file is None or metadata_column is None:
        sample_ids = set(np.random.choice(microbes_df.index, size=num_test))
        sample_ids = np.array([x in sample_ids for x in microbes_df.index])
    else:
        metadata = pd.read_table(metadata_file, index_col=0)
        sample_ids = set(metadata.loc[metadata[metadata_column]].index)
        sample_ids = np.array([(x in sample_ids) for x in microbes_df.index])


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


class Autoencoder(object):

    def __init__(self, session, num_samples,
                 d1, d2, u_mean=0, u_scale=1, v_mean=0, v_scale=1,
                 batch_size=50, latent_dim=3, dropout_rate=0.5,
                 learning_rate = 0.1, beta_1=0.999, beta_2=0.9999,
                 clipnorm=10., save_path=None):
        """ Build a tensorflow model

        Returns
        -------
        loss : tf.Tensor
           The log loss of the model.

        """
        p = latent_dim

        if save_path is None:
            basename = "logdir"
            suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            save_path = "_".join([basename, suffix])

        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.batch_size = batch_size
        self.clipnorm = clipnorm
        self.session = session

        self.total_count = tf.placeholder(
            tf.float32, [batch_size], name='total_count')
        self.Y_ph = tf.placeholder(
            tf.float32, [batch_size, d2], name='Y_ph')
        self.X_ph = tf.placeholder(
            tf.int32, [batch_size], name='X_ph')

        self.qU = tf.Variable(tf.random_normal([d1, p]), name='qU')
        self.qV = tf.Variable(tf.random_normal([p, d2-1]), name='qV')

        # regression coefficents distribution
        # U = Normal(loc=tf.zeros([d1, p]) + u_mean,
        #            scale=tf.ones([d1, p]) * u_scale,
        #            name='U')
        # V = Normal(loc=tf.zeros([p, d2-1]) + v_mean,
        #            scale=tf.ones([p, d2-1]) * v_scale,
        #            name='V')

        qU = tf.nn.dropout(self.qU, dropout_rate, name='qU_drop')
        qV = tf.nn.dropout(self.qV, dropout_rate, name='qV_drop')

        du = tf.gather(qU, self.X_ph, axis=0, name='du')
        dv = tf.concat([tf.zeros([batch_size, 1]),
                        du @ qV], axis=1, name='dv')

        Y = Multinomial(total_count=self.total_count, logits=dv, name='Y')

        norm = (num_samples / batch_size)
        #logprob_v =tf.reduce_sum(V.log_prob(self.qV), name='logprob_v')
        #logprob_u = tf.reduce_sum(U.log_prob(self.qU), name='logprob_u')
        logprob_y = tf.reduce_sum(Y.log_prob(self.Y_ph), name='logprob_y')
        # self.log_loss = - (logprob_y * norm + logprob_u + logprob_v)
        self.log_loss = - logprob_y * norm

        pred = tf.nn.log_softmax(dv) + \
               tf.reshape(tf.log(self.total_count), [-1, 1])
        err = tf.subtract(self.Y_ph, pred)
        self.cv = tf.sqrt(
            tf.reduce_mean(tf.reduce_mean(tf.multiply(err, err), axis=0)))

        tf.summary.scalar('loss', self.log_loss)
        tf.summary.scalar('cv', self.cv)
        tf.summary.histogram('qU', self.qU)
        tf.summary.histogram('qV', self.qV)
        self.merged = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(save_path, self.session.graph)
        with tf.name_scope('optimize'):
            optimizer = tf.train.AdamOptimizer(
                self.learning_rate, beta1=self.beta_1, beta2=self.beta_2)

            gradients, self.variables = zip(
                *optimizer.compute_gradients(self.log_loss))
            self.gradients, _ = tf.clip_by_global_norm(
                gradients, self.clipnorm)
            self.train = optimizer.apply_gradients(
                zip(self.gradients, self.variables))

        tf.global_variables_initializer().run()


    def fit(self, X, Y, epoch=10, batch_size=50, summary_interval=1000,
            testX=None, testY=None):
        """ Fits the model.

        Parameters
        ----------
       X : np.array
           Input table (likely OTUs).
        Y : np.array
           Output table (likely metabolites).

        Returns
        -------
        loss: float
            log likelihood loss.
        cv : float
            cross validation loss
        rU: np.array
            microbe latent parameters
        rV: np.array
            metabolite latent parameters
        """
        X_hits, sample_ids = onehot(X)
        iterations = epoch * X_hits.shape[0]
        cv = None

        for i in tqdm(range(0, iterations)):

            batch = np.random.randint(
                X_hits.shape[0], size=batch_size)
            batch_ids = sample_ids[batch]

            total = Y[batch_ids, :].sum(axis=1).astype(np.float32)

            if i % summary_interval == 0:
                if testX is not None and testY is not None:
                    cv = self.cross_validate(testX, testY)
                train_, summary, loss, rU, rV = self.session.run(
                    [self.train, self.merged, self.log_loss, self.qU, self.qV],
                    feed_dict={
                        self.X_ph: X_hits[batch],
                        self.Y_ph: Y[batch_ids, :],
                        self.total_count: total
                    }
                )
                self.writer.add_summary(summary, i)
            else:
                train_, loss, rU, rV = self.session.run(
                    [self.train, self.log_loss, self.qU, self.qV],
                    feed_dict={
                        self.X_ph: X_hits[batch],
                        self.Y_ph: Y[batch_ids, :],
                        self.total_count: total
                    }
                )
            # self.writer.add_summary(summary, i)

        self.U = rU
        self.V = rV

        return loss, cv


    def predict(self, X):
        """ Performs a prediction

        Parameters
        ----------
        X : np.array
           Input table (likely OTUs).

        Returns
        -------
        np.array :
           Predicted abundances.
        """
        X_hits, _ = onehot(X)

        d1 = X_hits.shape[0]
        r = self.U[X_hits] @ self.V
        res = softmax(np.hstack(
            (np.zeros((d1, 1)), r)))
        return res


    def cross_validate(self, X, Y):
        """
        Parameters
        ----------
        X : np.array
           Input table (likely OTUs).
        Y : np.array
           Output table (likely metabolites).

        Returns
        -------
        cv_loss: float
           Euclidean norm of the errors (i.e. the RMSE)

        """
        X_hits, sample_ids = onehot(X)

        total = Y[sample_ids, :].sum(axis=1).astype(np.float32)
        iterations = len(X_hits) // self.batch_size
        cv_losses = []
        for _ in range(iterations):
            batch = np.random.randint(
                X_hits.shape[0], size=self.batch_size)
            batch_ids = sample_ids[batch]
            total = Y[batch_ids, :].sum(axis=1).astype(np.float32)

            cv_loss, summary = self.session.run(
                [self.cv, self.merged],
                feed_dict={
                    self.X_ph: X_hits,
                    self.Y_ph: Y[sample_ids, :],
                    self.total_count: total
                }
            )
            cv_losses.append(cv_loss)
            self.writer.add_summary(summary, i)

        cv_loss = np.mean(cv_losses)
        return cv_loss


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
    rank_stats : pd.DataFrame
       List of OTUs along with their spearman predictive accuracy
    """
    # a little redundant
    otu_hits, sample_ids = onehot(microbes.values)
    batch_size = len(sample_ids)
    res = model.predict(microbes.values)
    exp = metabolites.values[sample_ids]

    ms_r = []
    prec = []
    recall = []
    tps = fps = fns = tns = 0
    ids = set(range(len(metabolites.columns)))

    n, d = res.shape
    rank_stats, rank_pvals = [], []
    tp_stats, fn_stats, fp_stats, tn_stats = [], [], [], []

    for i in range(n):

        exp_names = np.argsort(exp[i, :])[-top_N:]
        res_names = np.argsort(res[i, :])[-top_N:]
        result = spearmanr(exp[i, res_names],
                           res[i, res_names])
        r = result.correlation
        pval = result.pvalue

        if np.isnan(r):
            print(exp[i, exp_names])
            print(res[i, exp_names])

        rank_stats.append(r)
        rank_pvals.append(pval)

        hits  = set(res_names)
        truth = set(exp_names)

        tp_stats.append(len(hits & truth))
        fn_stats.append(len(truth - hits))
        fp_stats.append(len(hits - truth))
        tn_stats.append(len((ids - hits) & (ids - truth)))

        tps += len(hits & truth)
        fns += len(truth - hits)
        fps += len(hits - truth)
        tns += len((ids - hits) & (ids - truth))

        p = len(hits & truth) / top_N
        r = len(hits & truth) / d
        prec.append(p)
        recall.append(r)

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
    otu_names = [microbes.columns[o] for o in otu_hits]

    rank_stats = pd.DataFrame(
        {
            'spearman_r' : rank_stats,
            'pvalue': rank_pvals,
            'OTU': otu_names,
            'sample_ids': microbes.index[sample_ids],
            'TP': tp_stats,
            'FN': fn_stats,
            'FP': fp_stats,
            'TN': tn_stats
        }
    )

    rank_stats = rank_stats.groupby(by=['OTU', 'sample_ids']).mean()
    #rank_stats = rank_stats.drop_duplicates()

    return params, rank_stats


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
@click.option('--input_prior',
              help=('Width of normal prior for input embedding.  '
                    'Smaller values will regularize parameters towards zero. '
                    'Values must be greater than 0.'),
              default=1.)
@click.option('--output_prior',
              help=('Width of normal prior for input embedding.  '
                    'Smaller values will regularize parameters towards zero. '
                    'Values must be greater than 0.'),
              default=1.)
@click.option('--dropout-rate',
              help=('Dropout regularization.  Helps with preventing overfitting.'
                    'This is the probability of dropping a parameter at a given iteration.'
                    'Values must be between (0, 1)'),
              default=0.9)
@click.option('--top-k',
              help=('Number of top hits to compare for cross-validation.'),
              default=50)
@click.option('--learning-rate',
              help=('Gradient descent decay rate.'),
              default=1e-1)
@click.option('--clipnorm',
              help=('Gradient clipping size.'),
              default=10.)
@click.option('--summary-interval',
              help=('Number of iterations before a storing a summary.'),
              default=1000)
@click.option('--summary-dir', default='summarydir',
              help='Summary directory to save cross validation results.')
@click.option('--ranks-file',
              help='Ranks file containing microbe-metabolite rankings.')
def autoencoder(otu_train_file, otu_test_file,
                metabolite_train_file, metabolite_test_file,
                epochs, batch_size, latent_dim,
                input_prior, output_prior,
                dropout_rate, top_k,
                learning_rate, clipnorm,
                summary_interval, summary_dir, ranks_file):


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

    params = []

    sname = 'latent_dim_' + str(latent_dim) + \
           '_input_prior_%.2f' % input_prior + \
           '_output_prior_%.2f' % output_prior + \
           '_dropout_rate_%.2f' % dropout_rate
    sname = os.path.join(summary_dir, sname)

    n, d1 = microbes_df.shape
    n, d2 = metabolites_df.shape

    train_microbes_df = pd.DataFrame(
        np.array(train_microbes.matrix_data.todense()).T,
        index=train_microbes.ids(axis='sample'),
        columns=train_microbes.ids(axis='observation'))

    train_metabolites_df = pd.DataFrame(
        np.array(train_metabolites.matrix_data.todense()).T,
        index=train_metabolites.ids(axis='sample'),
        columns=train_metabolites.ids(axis='observation'))

    test_microbes_df = pd.DataFrame(
        np.array(test_microbes.matrix_data.todense()).T,
        index=test_microbes.ids(axis='sample'),
        columns=test_microbes.ids(axis='observation'))

    test_metabolites_df = pd.DataFrame(
        np.array(test_metabolites.matrix_data.todense()).T,
        index=test_metabolites.ids(axis='sample'),
        columns=test_metabolites.ids(axis='observation'))

    with tf.Graph().as_default(), tf.Session() as session:
        model = Autoencoder(
            session, n, d1, d2,
            latent_dim=latent_dim,
            u_scale=input_prior, v_scale=output_prior,
            learning_rate = learning_rate, beta_1=0.999, beta_2=0.9999,
            clipnorm=10., save_path=sname)

        loss, cv = model.fit(train_microbes_df.values,
                             train_metabolites_df.values,
                             epoch=epochs)

        U, V = model.U, model.V
        d1 = U.shape[0]

        ranks = clr(softmax(np.hstack((np.zeros((d1, 1)), U @ V))))
        ranks = pd.DataFrame(ranks, index=train_microbes_df.columns,
                             columns=train_metabolites_df.columns)

        params, rank_stats = cross_validation(
            model, test_microbes_df, test_metabolites_df, top_N=top_k)

        params.to_csv(os.path.join(summary_dir, 'model_results.csv'))
        rank_stats.to_csv(os.path.join(summary_dir, 'otu_cv_results.csv'))
        ranks.to_csv(ranks_file)


def rank_hits(ranks, k):
    """
    Parameters
    ----------
    ranks : pd.DataFrame
       Matrix of ranks
    k : int
       Number of nearest neighbors

    Returns
    -------
    edges : pd.DataFrame
       List of edges along with corresponding ranks.
    """
    axis = 1
    lookup = {x : i for i, x in enumerate(ranks.columns)}
    def sort_f(x):
        return [
            ranks.columns[i] for i in np.argsort(x)[-k:]
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
@click.option('--axis', default=0,
              help='Direction to draw edges. '
              'axis=0 guarantees that each row will have at least k edges'
              'axis=1 guarantees that each columns will have at least k edges')
def network(ranks_file, k_nearest_neighbors, node_metadata, edge_metadata, axis):
    ranks = pd.read_csv(ranks_file, index_col=0).T
    if axis == 0:
        edges = rank_hits(ranks, k_nearest_neighbors)
    else:
        edges = rank_hits(ranks.T, k_nearest_neighbors)
    edges['edge_type'] = 'co_occur'
    src = list(edges['src'].value_counts().index)
    dest = list(edges['dest'].value_counts().index)
    edges = edges.set_index(['src'])
    edges[['edge_type', 'dest']].to_csv(edge_metadata, sep='\t', header=False)

    nodes = pd.DataFrame(columns=['id', 'node_type'])
    nodes['id'] = list(src) + list(dest)
    nodes['node_type'] = ['src'] * len(src) + ['dest'] * len(dest)
    nodes = nodes.set_index('id')
    nodes.to_csv(node_metadata, sep='\t')


if __name__ == '__main__':
    multimodal()
