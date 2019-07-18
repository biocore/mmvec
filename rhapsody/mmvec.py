from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multinomial import Multinomial
from torch.distributions.normal import Normal
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
        prior = self.encoder.log_prob(self.in_prior) + \
            self.decoder.log_prob(self.out_prior)
        return -(likelihood + prior)

    def fit(self, trainX, trainY, testX, testY, epochs=1000,
            learning_rate=1e-3, beta1=0.9, beta2=0.99):
        losses = []
        klds = []
        likes = []
        errs = []
        optimizer = optim.Adam(self.parameters(), betas=(beta1, beta2),
                               lr=learning_rate)
        for ep in tqdm(range(0, epochs)):

            self.train()
            for i in range(0, self.num_samples, self.batch_size):
                optimizer.zero_grad()

                inp, out = get_batch(trainX, trainY, i % self.num_samples,
                                     self.subsample_size, self.batch_size)

                pred = self.forward(inp)
                loss = self.loss(pred, out)
                metabolite_total = torch.sum(out, 1).view(-1, 1)
                err = torch.mean(torch.abs(F.softmax(pred, dim=1) * metabolite_total - out))
                loss.backward()

                errs.append(err.item())
                losses.append(loss.item())

                optimizer.step()

        return losses, errs

    def ranks(self):
        U = self.encoder.embedding.weight
        Ub = self.encoder.bias.weight
        V = self.decoder.weight
        Vb = self.decoder.bias

        return Ub.view(-1, 1) + (U @ torch.t(V)) + Vb
