import os
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
from skbio.stats.composition import clr_inv as softmax
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.contrib.distributions import Multinomial, Normal
import datetime
from .util import onehot


class Autoencoder(object):

    def __init__(self, u_mean=0, u_scale=1, v_mean=0, v_scale=1,
                 batch_size=50, latent_dim=3, dropout_rate=0.5,
                 learning_rate=0.1, beta_1=0.999, beta_2=0.9999,
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
        self.p = p
        self.u_mean = u_mean
        self.u_scale = u_scale
        self.v_mean = v_mean
        self.v_scale = v_scale
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.clipnorm = clipnorm
        self.save_path = save_path

    def __call__(self, session, X, Y):
        """ Initialize the actual graph

        Parameters
        ----------
        session : tf.Session
            Tensorflow session
        X : sparse array in coo format
            Input OTU table, where rows are samples and columns are
            observations
        Y : np.array
            Output metabolite table
        """
        self.session = session
        self.nnz = len(X.data)
        self.d1 = X.shape[1]
        self.d2 = Y.shape[1]

        X_ph = tf.SparseTensor(
            indices=np.array([X.row,  X.col]).T,
            values=X.data,
            dense_shape=X.shape)
        Y_ph = tf.constant(Y, dtype=tf.float32)
        total_count = tf.reduce_sum(Y_ph, axis=1)
        batch_ids = tf.multinomial(tf.log(tf.reshape(X_ph.values, [1, -1])),
                                   self.batch_size)
        batch_ids = tf.squeeze(batch_ids)
        X_samples = tf.gather(X_ph.indices, 0, axis=1)
        X_obs = tf.gather(X_ph.indices, 1, axis=1)
        sample_ids = tf.gather(X_samples, batch_ids)

        Y_batch = tf.gather(Y_ph, sample_ids)
        X_batch = tf.gather(X_obs, batch_ids)

        self.qUmain = tf.Variable(
            tf.random_normal([self.d1, self.p]), name='qU')
        self.qUbias = tf.Variable(
            tf.random_normal([self.d1, 1]), name='qUbias')
        self.qVmain = tf.Variable(
            tf.random_normal([self.p, self.d2-1]), name='qV')
        self.qVbias = tf.Variable(
            tf.random_normal([1, self.d2-1]), name='qVbias')

        qU = tf.concat(
            [tf.ones([self.d1, 1]), self.qUbias, self.qUmain], axis=1)
        qV = tf.concat(
            [self.qVbias, tf.ones([1, self.d2-1]), self.qVmain], axis=0)

        # regression coefficents distribution
        Umain = Normal(loc=tf.zeros([self.d1, self.p]) + self.u_mean,
                       scale=tf.ones([self.d1, self.p]) * self.u_scale,
                       name='U')
        Ubias = Normal(loc=tf.zeros([self.d1, 1]) + self.u_mean,
                       scale=tf.ones([self.d1, 1]) * self.u_scale,
                       name='biasU')

        Vmain = Normal(loc=tf.zeros([self.p, self.d2-1]) + self.v_mean,
                       scale=tf.ones([self.p, self.d2-1]) * self.v_scale,
                       name='V')
        Vbias = Normal(loc=tf.zeros([1, self.d2-1]) + self.v_mean,
                       scale=tf.ones([1, self.d2-1]) * self.v_scale,
                       name='biasV')

        du = tf.gather(qU, X_batch, axis=0, name='du')
        dv = tf.concat([tf.zeros([self.batch_size, 1]),
                        du @ qV], axis=1, name='dv')

        tc = tf.gather(total_count, sample_ids)
        Y = Multinomial(total_count=tc, logits=dv, name='Y')
        num_samples = X.shape[0]
        norm = num_samples / self.batch_size
        logprob_vmain = tf.reduce_sum(
            Vmain.log_prob(self.qVmain), name='logprob_vmain')
        logprob_vbias = tf.reduce_sum(
            Vbias.log_prob(self.qVbias), name='logprob_vbias')
        logprob_umain = tf.reduce_sum(
            Umain.log_prob(self.qUmain), name='logprob_umain')
        logprob_ubias = tf.reduce_sum(
            Ubias.log_prob(self.qUbias), name='logprob_ubias')
        logprob_y = tf.reduce_sum(Y.log_prob(Y_batch), name='logprob_y')
        self.log_loss = - (
            logprob_y * norm +
            logprob_umain + logprob_ubias +
            logprob_vmain + logprob_vbias
        )

        pred = tf.nn.log_softmax(dv) + tf.reshape(tf.log(tc), [-1, 1])
        err = tf.subtract(Y_batch, pred)
        self.cv = tf.sqrt(
            tf.reduce_mean(tf.reduce_mean(tf.multiply(err, err), axis=0)))

        tf.summary.scalar('logloss', self.log_loss)
        tf.summary.scalar('cv_rmse', self.cv)
        tf.summary.histogram('qUmain', self.qUmain)
        tf.summary.histogram('qVmain', self.qVmain)

        tf.summary.histogram('qUbias', self.qUbias)
        tf.summary.histogram('qVbias', self.qVbias)
        self.merged = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(self.save_path, self.session.graph)
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

    def fit(self, epoch=10, summary_interval=1000, checkpoint_interval=3600,
            testX=None, testY=None):
        """ Fits the model.

        Parameters
        ----------
        epoch : int
           Number of epochs to train
        summary_interval : int
           Number of seconds until a summary is recorded
        checkpoint_interval : int
           Number of seconds until a checkpoint is recorded

        Returns
        -------
        loss: float
            log likelihood loss.
        cv : float
            cross validation loss
        """
        iterations = epoch * self.nnz // self.batch_size

        cv = None
        last_checkpoint_time = 0
        last_summary_time = 0
        saver = tf.train.Saver()
        now = time.time()
        for i in tqdm(range(0, iterations)):

            if now - last_summary_time > summary_interval:
                if testX is not None and testY is not None:
                    cv = self.cross_validate(testX, testY)
                res = self.session.run(
                    [self.train, self.merged, self.log_loss,
                     self.qUmain, self.qUbias,
                     self.qVmain, self.qVbias]
                )
                train_, summary, loss, rU, rUb, rV, rVb = res
                self.writer.add_summary(summary, i)
                last_summary_time = now
            else:
                res = self.session.run(
                    [self.train, self.log_loss,
                     self.qUmain, self.qUbias,
                     self.qVmain, self.qVbias]
                )
                train_, loss, rU, rUb, rV, rVb = res

            # checkpoint model
            now = time.time()
            if now - last_checkpoint_time > checkpoint_interval:
                saver.save(self.session,
                           os.path.join(self.save_path, "model.ckpt"),
                           global_step=i)
                last_checkpoint_time = now

        self.U = rU
        self.V = rV
        self.Ubias = rUb
        self.Vbias = rVb

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
        U_ = np.hstack(
            (np.ones((self.U.shape[0], 1)), self.Ubias, self.U))
        V_ = np.vstack(
            (self.Vbias, np.ones((1, self.V.shape[1])), self.V))
        r = U_[X_hits] @ V_
        res = softmax(np.hstack(
            (np.zeros((d1, 1)), r)))
        return res

    def cross_validate(self, X, Y):
        pass


def cross_validation(model, microbes, metabolites, top_N=50):
    """ Running cross validation on test data.

    Parameters
    ----------
    model : Autoencoder
       Pre-trained tensorflow model
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
    res = model.predict(microbes.values)
    exp = metabolites.values[sample_ids]

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

        hits = set(res_names)
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
            'spearman_r': rank_stats,
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

    return params, rank_stats
