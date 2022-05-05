import os
import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Multinomial, Normal
import datetime


class MMvec(object):

    def __init__(self, u_mean=0, u_scale=1, v_mean=0, v_scale=1,
                 batch_size=50, latent_dim=3,
                 learning_rate=0.1, beta_1=0.8, beta_2=0.9,
                 clipnorm=10., device_name='/cpu:0', save_path=None):
        """ Build a tensorflow model for microbe-metabolite vectors

        Returns
        -------
        loss : tf.Tensor
           The log loss of the model.

        Notes
        -----
        To enable a GPU, set the device to '/device:GPU:x'
        where x is 0 or greater
        """
        p = latent_dim
        self.device_name = device_name
        if save_path is None:
            basename = "logdir"
            suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            save_path = "_".join([basename, suffix])

        self.p = p
        self.u_mean = u_mean
        self.u_scale = u_scale
        self.v_mean = v_mean
        self.v_scale = v_scale
        self.batch_size = batch_size
        self.latent_dim = latent_dim

        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.clipnorm = clipnorm
        self.save_path = save_path

    def __call__(self, session, trainX, trainY, testX, testY):
        """ Initialize the actual graph

        Parameters
        ----------
        session : tf.Session
            Tensorflow session
        trainX : sparse array in coo format
            Test input OTU table, where rows are samples and columns are
            observations
        trainY : np.array
            Test output metabolite table
        testX : sparse array in coo format
            Test input OTU table, where rows are samples and columns are
            observations.  This is mainly for cross validation.
        testY : np.array
            Test output metabolite table.  This is mainly for cross validation.
        """
        self.session = session
        self.nnz = len(trainX.data)
        self.d1 = trainX.shape[1]
        self.d2 = trainY.shape[1]
        self.cv_size = len(testX.data)

        # keep the multinomial sampling on the cpu
        # https://github.com/tensorflow/tensorflow/issues/18058
        with tf.device('/cpu:0'):
            X_ph = tf.SparseTensor(
                indices=np.array([trainX.row,  trainX.col]).T,
                values=trainX.data,
                dense_shape=trainX.shape)
            Y_ph = tf.constant(trainY, dtype=tf.float32)

            X_holdout = tf.SparseTensor(
                indices=np.array([testX.row,  testX.col]).T,
                values=testX.data,
                dense_shape=testX.shape)
            Y_holdout = tf.constant(testY, dtype=tf.float32)

            total_count = tf.reduce_sum(Y_ph, axis=1)
            batch_ids = tf.multinomial(
                tf.log(tf.reshape(X_ph.values, [1, -1])),
                self.batch_size)
            batch_ids = tf.squeeze(batch_ids)
            X_samples = tf.gather(X_ph.indices, 0, axis=1)
            X_obs = tf.gather(X_ph.indices, 1, axis=1)
            sample_ids = tf.gather(X_samples, batch_ids)

            Y_batch = tf.gather(Y_ph, sample_ids)
            X_batch = tf.gather(X_obs, batch_ids)

        with tf.device(self.device_name):
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
            num_samples = trainX.shape[0]
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

        # keep the multinomial sampling on the cpu
        # https://github.com/tensorflow/tensorflow/issues/18058
        with tf.device('/cpu:0'):
            # cross validation
            with tf.name_scope('accuracy'):
                cv_batch_ids = tf.multinomial(
                    tf.log(tf.reshape(X_holdout.values, [1, -1])),
                    self.cv_size)
                cv_batch_ids = tf.squeeze(cv_batch_ids)
                X_cv_samples = tf.gather(X_holdout.indices, 0, axis=1)
                X_cv = tf.gather(X_holdout.indices, 1, axis=1)
                cv_sample_ids = tf.gather(X_cv_samples, cv_batch_ids)

                Y_cvbatch = tf.gather(Y_holdout, cv_sample_ids)
                X_cvbatch = tf.gather(X_cv, cv_batch_ids)
                holdout_count = tf.reduce_sum(Y_cvbatch, axis=1)
                cv_du = tf.gather(qU, X_cvbatch, axis=0, name='cv_du')
                pred = tf.reshape(
                    holdout_count, [-1, 1]) * tf.nn.softmax(
                        tf.concat([tf.zeros([
                            self.cv_size, 1]),
                                   cv_du @ qV], axis=1, name='pred')
                    )

                self.cv = tf.reduce_mean(
                    tf.squeeze(tf.abs(pred - Y_cvbatch))
                )

        # keep all summaries on the cpu
        with tf.device('/cpu:0'):
            tf.summary.scalar('logloss', self.log_loss)
            tf.summary.scalar('cv_rmse', self.cv)
            tf.summary.histogram('qUmain', self.qUmain)
            tf.summary.histogram('qVmain', self.qVmain)
            tf.summary.histogram('qUbias', self.qUbias)
            tf.summary.histogram('qVbias', self.qVbias)
            self.merged = tf.summary.merge_all()

            self.writer = tf.summary.FileWriter(
                self.save_path, self.session.graph)

        with tf.device(self.device_name):
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

    def ranks(self):
        modelU = np.hstack(
            (np.ones((self.U.shape[0], 1)), self.Ubias, self.U))
        modelV = np.vstack(
            (self.Vbias, np.ones((1, self.V.shape[1])), self.V))

        res = np.hstack((np.zeros((self.U.shape[0], 1)), modelU @ modelV))
        res = res - res.mean(axis=1).reshape(-1, 1)
        return res

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
        losses, cvs = [], []
        cv = None
        last_checkpoint_time = 0
        last_summary_time = 0
        saver = tf.train.Saver()
        now = time.time()
        for i in tqdm(range(0, iterations)):
            if now - last_summary_time > summary_interval:

                res = self.session.run(
                    [self.train, self.merged, self.log_loss, self.cv,
                     self.qUmain, self.qUbias,
                     self.qVmain, self.qVbias]
                )
                train_, summary, loss, cv, rU, rUb, rV, rVb = res
                self.writer.add_summary(summary, i)
                last_summary_time = now
            else:
                res = self.session.run(
                    [self.train, self.log_loss,
                     self.qUmain, self.qUbias,
                     self.qVmain, self.qVbias]
                )
                train_, loss, rU, rUb, rV, rVb = res
                losses.append(loss)
                cvs.append(cv)
                cv = None

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

        return losses, cvs
