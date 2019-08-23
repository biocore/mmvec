# see http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal


class VecEmbedding(torch.nn.Module):
    def __init__(self, in_features, out_features):
        """
        In the constructor we instantiate two nn.Linear modules and assign
        them as member variables.
        """
        super(VecEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(in_features, out_features)
        self.bias = torch.nn.Embedding(in_features, 1)

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and
        we must return a Tensor of output data. We can use Modules
        defined in the constructor as well as arbitrary operators on Tensors.
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
        super(VecLinear, self).__init__(in_features, out_features - 1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must
        return a Tensor of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Tensors.
        """
        y = F.linear(x, self.weight, self.bias)
        z = torch.zeros(y.shape[0], 1, device=self.weight.device)
        return torch.cat((z, y), 1)

    @property
    def weight_(self):
        z = torch.zeros(1, self.weight.shape[1], device=self.weight.device)
        return torch.cat((z, self.weight))

    @property
    def bias_(self):
        z = torch.zeros(1, device=self.bias.device)
        return torch.cat((z, self.bias))

    def log_prob(self, sigma):
        """ This is for MAP regularization """
        mu = torch.zeros_like(self.weight)
        sigma = torch.ones_like(self.weight)
        w = Normal(mu, sigma).log_prob(self.weight)

        muB = torch.zeros_like(self.bias)
        sigmaB = torch.ones_like(self.bias)
        b = Normal(muB, sigmaB).log_prob(self.bias)
        return torch.sum(w) + torch.sum(b)
