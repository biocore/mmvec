from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multinomial import Multinomial
from torch.distributions.normal import Normal
from torch.optim.lr_scheduler import StepLR
from rhapsody.layers import VecEmbedding, VecLinear
from rhapsody.batch import get_batch


class MMvec(torch.nn.Module):
    def __init__(self, num_samples, num_microbes, num_metabolites, microbe_total,
                 latent_dim, batch_size=10, subsample_size=100,
                 in_prior=1, out_prior=1, device='cpu'):
        super(MMvec, self).__init__()
        self.num_microbes = num_microbes
        self.num_metabolites = num_metabolites
        self.num_samples = num_samples
        self.device = device
        self.batch_size = batch_size
        self.subsample_size = subsample_size
        self.microbe_total = microbe_total
        self.in_prior = 1
        self.out_prior = 1
        self.latent_dim = latent_dim

        # TODO: enable max norm in embedding to account for scale identifiability
        self.encoder = VecEmbedding(num_microbes, latent_dim)
        self.decoder = VecLinear(latent_dim, num_metabolites)

    def forward(self, x):
        code = self.encoder(x)
        log_probs = self.decoder(code)
        #zeros = torch.zeros(self.batch_size * self.subsample_size, 1)
        #log_probs = torch.cat((zeros, alrs), dim=1)
        return log_probs

    def loss(self, pred, obs):
        """ Computes the loss function to be minimized. """
        n = self.microbe_total * self.num_samples
        likelihood = n * torch.mean(Multinomial(logits=pred).log_prob(obs))
        encoder_prior = self.encoder.log_prob(self.in_prior)
        decoder_prior = self.decoder.log_prob(self.out_prior)
        prior = encoder_prior + decoder_prior
        return -(likelihood + prior)

    def fit(self, train_dataloader, test_dataloader, epochs=1000,
            learning_rate=1e-3, beta1=0.9, beta2=0.99):
        """ Fit the model

        Parameters
        ----------
        train_dataloader: torch.data_utils.DataLoader
            Torch DataLoader iterator for training samples
        test_dataloader: torch.data_utils.DataLoader
            Torch DataLoader iterator for testing samples
        epochs : int
            Number of epochs to train model
        learning_rate : float
            The initial learning rate
        beta1 : float
            First ADAM momentum constant
        beta2 : float
            Second ADAM momentum constant

        Returns
        -------
        losses : list of float
            Log likelihood of model
        errs : list of float
            Cross validation error between model and testing dataset
        """
        losses = []
        klds = []
        likes = []
        errs = []
        # custom make scheduler for alternating
        baseline = 1e-8
        lrs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        with tqdm(total=epochs*len(lrs)*2) as pbar:
            for lr in lrs:
                # setup optimizer for alternating optimization
                for (b, l) in [[baseline, lr], [lr, baseline]]:
                    optimizer = optim.Adamax(
                        [
                            {'params': self.encoder.parameters(), 'lr': b},
                            {'params': self.decoder.parameters(), 'lr': l}
                        ],
                        betas=(beta1, beta2))
                    for _ in range(0, epochs):
                        # Train
                        self.train()
                        for inp, out in train_dataloader:
                            inp = inp.to(self.device)
                            out = out.to(self.device)
                            optimizer.zero_grad()

                            pred = self.forward(inp)
                            loss = self.loss(pred, out)
                            loss.backward()
                            losses.append(loss.item())
                            optimizer.step()

                        # Validation
                        mean_err = []
                        for inp, out in test_dataloader:
                            inp = inp.to(self.device)
                            out = out.to(self.device)
                            mt = torch.sum(out, 1).view(-1, 1)
                            err = torch.mean(
                                torch.abs(
                                    F.softmax(pred, dim=1) * mt - out
                                )
                            )
                            mean_err.append(err.item())
                        errs.append(np.mean(mean_err))
                        pbar.update(1)

        return losses, errs

    def ranks(self):
        U = self.encoder.embedding.weight
        Ub = self.encoder.bias.weight
        V = self.decoder.weight_
        Vb = self.decoder.bias_
        res = Ub.view(-1, 1) + (U @ torch.t(V)) + Vb
        res = res - res.mean(1).view(-1, 1)
        return res

    def embeddings(self, rowids, columnids):
        U = self.encoder.embedding.weight
        Ub = self.encoder.bias.weight
        V = self.decoder.weight
        Vb = self.decoder.bias

        pc_ids = ['PC%d' % i for i in range(latent_dim)]
        df = pd.concat(
            (
                format_params(U, pc_ids, rowids, 'microbe'),
                format_params(V.T, pc_ids, columnids, 'metabolite'),
                format_params(Ub, ['bias'], rowids, 'microbe'),
                format_params(Vb, ['bias'], columnids, 'metabolite')
            ), axis=0)

        return df

    def ordination(self, rowids, columnids):

        pc_ids = ['PC%d' % i for i in range(U.shape[1])]
        res = self.ranks()
        res = res - res.mean(axis=0).view(-1, 1)
        u, s, v = svds(res.detach().numpy(), k=latent_dim)
        microbe_embed = u @ np.diag(s)
        metabolite_embed = v.T

        pc_ids = ['PC%d' % i for i in range(latent_dim)]
        features = pd.DataFrame(
            microbe_embed, columns=pc_ids,
            index=rowids)
        samples = pd.DataFrame(
            metabolite_embed, columns=pc_ids,
            index=columnids)
        short_method_name = 'mmvec biplot'
        long_method_name = 'Multiomics mmvec biplot'
        eigvals = pd.Series(s, index=pc_ids)
        proportion_explained = pd.Series(s**2 / np.sum(s**2), index=pc_ids)
        biplot = OrdinationResults(
            short_method_name, long_method_name, eigvals,
            samples=samples, features=features,
            proportion_explained=proportion_explained)
