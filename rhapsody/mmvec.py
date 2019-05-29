import os
import time
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from rhapsody.batch import get_batch
from rhapsody.layers import GaussianEmbedding, GaussianDecoder
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multinomial import Multinomial


class MMvec(nn.Module):
    def __init__(self, num_samples, num_microbes, num_metabolites,
                 microbe_total, latent_dim, batch_size=10,
                 subsample_size=100):
        super(MMvec, self).__init__()
        self.num_microbes = num_microbes
        self.num_metabolites = num_metabolites
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.subsample_size = subsample_size
        self.microbe_total = microbe_total
        self.encoder = GaussianEmbedding(in_features=num_microbes,
                                         out_features=latent_dim)
        self.decoder = GaussianDecoder(in_features=latent_dim,
                                       out_features=num_metabolites)

    def forward(self, x):
        code = self.encoder(x)
        log_probs = self.decoder(code)
        return log_probs

    def loss(self, pred, obs):
        """ Computes the loss function to be minimized. """
        kld = self.encoder.divergence() + self.decoder.divergence()
        n = self.microbe_total * self.num_samples
        likelihood = n * torch.mean(Multinomial(logits=pred).log_prob(obs))
        metabolite_total = torch.sum(obs, 1).view(-1, 1)
        err = torch.mean(
            torch.abs(F.softmax(pred, dim=1) * metabolite_total - obs))
        elbo = kld + likelihood
        return -elbo, kld, likelihood, err

    def validate(self, inp, out):
        """ Computes cross-validation scores on holdout train/test set.

        Here, the mean absolute error is computed, which can be interpreted
        as the average number of counts that were incorrectly predicted.
        """
        logprobs = self.forward(inp)
        n = torch.sum(out, 1)
        probs = torch.nn.functional.softmax(logprobs, 1)
        pred = n.view(-1, 1) * probs

        # computes mean absolute error.
        mae = torch.mean(torch.abs(out - pred))
        return mae

    @staticmethod
    def load(model_file):
        """ Initializes weights based on model parameters file.

        Parameters
        ----------
        model_file : str
           Path to MMvec model parameters.

        Returns
        -------
        MMvec model
        """
        params = pd.read_csv(model_file, index_col=0)
        Uparam = params.loc[params.embed_type=='microbe']
        Vparam = params.loc[params.embed_type=='metabolite']
        Umean = pd.pivot(Uparam, index='feature_id',
                         columns='axis', values='mean')
        Vmean = pd.pivot(Vparam, index='feature_id',
                         columns='axis', values='mean')
        Ustd = pd.pivot(Uparam, index='feature_id',
                        columns='axis', values='stddev')
        Vstd = pd.pivot(Vparam, index='feature_id',
                        columns='axis', values='stddev')
        pc_ids = sorted(list(set(Umean.columns) - {'bias'}))
        latent_dim = len(pc_ids)
        num_microbes = Umean.shape[0]
        num_metabolites = Vmean.shape[0]
        # Note that we are hardcoding num_samples and microbe_total to -1,
        # since it is not relevant for preloading weights.
        # TODO: We will want to somehow make this more appear to the user
        # at some point
        model = MMvec(-1, num_microbes, num_metabolites, -1, latent_dim)
        # Load encoder embedding
        model.encoder.embedding.weight.data.copy_(
            torch.from_numpy(Umean[pc_ids].values)
        )
        model.encoder.bias.weight.data.copy_(
            torch.from_numpy(Umean['bias'].values.reshape(-1, 1))
        )
        model.encoder.embedding_var.weight.data.copy_(
            torch.from_numpy(2*np.log(Ustd[pc_ids].values))
        )
        model.encoder.bias_var.weight.data.copy_(
            torch.from_numpy(2*np.log(Ustd['bias'].values.reshape(-1, 1)))
        )
        # Load decoder weights
        model.decoder.mean.weight.data.copy_(
            torch.from_numpy(Vmean[pc_ids].values)
        )
        model.decoder.mean.bias.data.copy_(
            torch.from_numpy(Vmean['bias'].values)
        )
        model.decoder.var.weight.data.copy_(
            torch.from_numpy(2*np.log(Vstd[pc_ids].values))
        )
        model.decoder.var.bias.data.copy_(
            torch.from_numpy(2*np.log(Vstd['bias'].values))
        )

        return model, list(Umean.index), list(Vmean.index)

    def fit(self, trainX, trainY, testX, testY,
            epochs=10, learning_rate=0.1, beta1=0.9, beta2=0.99,
            summary_interval=5, checkpoint_interval=60, device='cpu',
            save_path=None):
        """ Trains a co-occurrence model to predict Y from X.

        Parameters
        ----------
        model : rhapsody.mmvec.MMvec
            Model to train
        trainX : scipy.sparse.csr
            Input training data (samples x features)
        trainY : np.array
            Output training data (samples x features)
        testX : scipy.sparse.csr
            Input testing data (samples x features)
        testY : np.array
            Output testing data (samples x features)
        epochs : int
            Number of training iterations over the entire dataset
        batch_size : int
            Number of samples to train per iteration
        beta2 : float
            Second momentum constant for ADAM gradient descent Values
            can only be between (0, 1). Values close to 1 indicate
            sparse updates.
        summary_interval : int
            The number of seconds until a summary is written.
        checkpoint_interval : int
            The number of seconds until a checkpoint is saved.

        TODO
        ----
        Eventually, this will need to be refactored to include the DataLoader class
        Also, we may want to make the save directory a parameter in this method.

        TODO: set the microbe_total and num_samples in this fit method
        """
        if save_path is None:
            basename = "logdir"
            suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            save_path = "_".join([basename, suffix])

        losses, klds, likes, errs = [], [], [], []
        last_checkpoint_time = 0
        last_summary_time = 0
        num_samples = trainY.shape[0]
        optimizer = optim.Adam(self.parameters(), betas=(beta1, beta2),
                               lr=learning_rate)
        writer = SummaryWriter(save_path)
        for ep in tqdm(range(0, epochs)):

            self.train()
            for i in range(0, self.num_samples, self.batch_size):
                now = time.time()
                optimizer.zero_grad()

                inp, out = get_batch(trainX, trainY, i % self.num_samples,
                                     self.subsample_size, self.batch_size)
                inp = inp.to(device)
                out = out.to(device)
                pred = self.forward(inp)
                loss, kld, like, err = self.loss(pred, out)
                losses.append(loss.item())
                klds.append(kld.item())
                likes.append(like.item())
                errs.append(err.item())

                loss.backward()
                optimizer.step()

                # save summary
                if now - last_summary_time > summary_interval:
                    test_in, test_out = get_batch(testX, testY, i % num_samples,
                                         self.subsample_size, self.batch_size)
                    test_in = test_in.to(device=device)
                    test_out = test_out.to(device=device)

                    cv_mae = self.validate(test_in, test_out)
                    iteration = i + ep*num_samples
                    writer.add_scalar('elbo', loss, iteration)
                    writer.add_scalar('KL_divergence', kld, iteration)
                    writer.add_scalar('log_likelihood', like, iteration)
                    writer.add_scalar('cv_mean_count_err', cv_mae, iteration)
                    writer.add_scalar('train_mean_count_err', err, iteration)
                    writer.add_embedding(
                        self.encoder.embedding.weight.detach(),
                        global_step=iteration, tag='U')
                    # note that these are in alr coordinates
                    writer.add_embedding(
                        self.decoder.mean.weight.detach(),
                        global_step=iteration, tag='V')
                    last_summary_time = now

                # checkpoint self
                now = time.time()
                if now - last_checkpoint_time > checkpoint_interval:
                    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
                    torch.save(self.state_dict(),
                               os.path.join(save_path,
                                            'checkpoint_' + suffix))
                    last_checkpoint_time = now

                optimizer.step()

        return losses, klds, likes, errs
