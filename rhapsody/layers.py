# see http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.nn.utils import clip_grad_norm


class VecEmbedding(torch.nn.Module):
    def __init__(self, in_features, out_features):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(VecEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(in_features, out_features)
        self.bias = torch.nn.Embedding(in_features, 1)

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        embeds = self.embedding(x)
        biases = self.bias(x)
        pred = embeds + biases
        return pred

    def log_prob(self, sigma):
        """ This is for MAP regularization """
        mu = torch.zeros_like(self.embedding.weight)
        sigma = torch.ones_like(self.embedding.weight)
        w = Normal(mu, sigma).log_prob(self.embedding.weight)

        muB = torch.zeros_like(self.bias.weight)
        sigmaB = torch.ones_like(self.bias.weight)
        b = Normal(muB, sigmaB).log_prob(self.bias.weight)
        return torch.sum(w) + torch.sum(b)


class VecLinear(torch.nn.Linear):

    def __init__(self, in_features, out_features):
        """
        In the constructor we just inherit
        """
        super(VecLinear, self).__init__(in_features, out_features)

    def log_prob(self, sigma):
        """ This is for MAP regularization """
        mu = torch.zeros_like(self.weight)
        sigma = torch.ones_like(self.weight)
        w = Normal(mu, sigma).log_prob(self.weight)

        muB = torch.zeros_like(self.bias)
        sigmaB = torch.ones_like(self.bias)
        b = Normal(muB, sigmaB).log_prob(self.bias)
        return torch.sum(w) + torch.sum(b)
