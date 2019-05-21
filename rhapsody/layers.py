import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.multinomial import Multinomial
from torch.nn.utils import clip_grad_norm


class GaussianLayer(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(GaussianLayer, self).__init__()

    def reparameterize(self, mu, logvar):
        """ Samples from the posterior distribution via
        reparameterization gradients"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def _divergence(self, mu, logvar):
        return 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class GaussianEmbedding(GaussianLayer):
    def __init__(self, in_features, out_features):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(GaussianEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(in_features, out_features)
        self.bias = torch.nn.Embedding(in_features, 1)

        self.embedding_var = torch.nn.Embedding(in_features, out_features)
        self.bias_var = torch.nn.Embedding(in_features, 1)
        self.in_features = in_features
        self.out_features = out_features

    def divergence(self):
        """ Computes the KL divergence between posterior and prior. """
        w = self._divergence(self.embedding.weight, self.embedding_var.weight)
        b = self._divergence(self.bias.weight, self.bias_var.weight)
        return w + b

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        embeds = self.reparameterize(
            self.embedding(x),
            self.embedding_var(x)
        )
        biases = self.reparameterize(
            self.bias(x),
            self.bias_var(x)
        )
        pred = embeds + biases
        return pred


class GaussianDecoder(GaussianLayer):
    def __init__(self, in_features, out_features):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(GaussianDecoder, self).__init__()
        self.mean = torch.nn.Linear(in_features, out_features)
        self.var = torch.nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

    def divergence(self):
        """ Computes the KL divergence between posterior and prior. """
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        w = self._divergence(self.mean.weight, self.var.weight)
        b = self._divergence(self.mean.bias, self.var.bias)
        return w + b

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        pred = self.reparameterize(
            self.mean(x),
            self.var(x)
        )
        return pred
