import os
import time
import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
from rhapsody.batch import get_batch
from rhapsody.layers import GaussianEmbedding, GaussianDecoder

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multinomial import Multinomial


def train_cooccurrence(model, trainX, trainY, testX, testY,
                       epochs=10, learning_rate=0.1, beta1=0.9, beta2=0.99,
                       summary_interval=5, checkpoint_interval=60, device='cpu'):
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
    """
    last_checkpoint_time = 0
    last_summary_time = 0
    num_samples = trainY.shape[0]
    optimizer = optim.Adam(model.parameters(), betas=(beta1, beta2),
                           lr=learning_rate)
    writer = SummaryWriter(model.save_path)
    for ep in tqdm(range(0, epochs)):

        model.train()
        for i in range(0, model.num_samples, model.batch_size):
            now = time.time()
            optimizer.zero_grad()

            inp, out = get_batch(trainX, trainY, i % model.num_samples,
                                 model.subsample_size, model.batch_size)

            pred = model.forward(inp)
            loss, kld, like, err = model.loss(pred, out)
            loss.backward()
            optimizer.step()

            # save summary
            if now - last_summary_time > summary_interval:
                test_in, test_out = get_batch(testX, testY, i % num_samples,
                                     model.subsample_size, model.batch_size)
                test_in = test_in.to(device=device)
                test_out = test_out.to(device=device)

                cv_mae = model.validate(test_in, test_out)
                iteration = i + ep*num_samples
                writer.add_scalar('elbo', loss, iteration)
                writer.add_scalar('KL_divergence', kld, iteration)
                writer.add_scalar('log_likelihood', like, iteration)
                writer.add_scalar('cv_mean_count_err', cv_mae, iteration)
                writer.add_scalar('train_mean_count_err', err, iteration)
                writer.add_embedding(
                    model.encoder.embedding.weight.detach(),
                    global_step=iteration, tag='U')
                # note that these are in alr coordinates
                writer.add_embedding(
                    model.decoder.mean.weight.detach(),
                    global_step=iteration, tag='V')
                last_summary_time = now

            # checkpoint model
            now = time.time()
            if now - last_checkpoint_time > checkpoint_interval:
                suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
                torch.save(model.state_dict(),
                           os.path.join(model.save_path,
                                        'checkpoint_' + suffix))
                last_checkpoint_time = now

            optimizer.step()

    return model
