# see http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.multinomial import Multinomial
from torch.nn.utils import clip_grad_norm
from rhapsody.layers import GaussianDecoder, GaussianEmbedding
from rhapsody.sim import bimodal
from rhapsody.util import onehot
import unittest

import numpy.testing as npt


# See below for tips on how to unittest deep learning frameworks
# https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
class TestGaussianEmbedding(unittest.TestCase):

    def setUp(self):
        self.num_samples = 100
        self.num_microbes = 10
        self.input, self.output = bimodal(self.num_samples, self.num_microbes)

    def test_gaussian_embedding_forward(self):
        x, ids = onehot(self.input)
        y = self.output[ids]
        model = GaussianEmbedding(self.num_microbes, 1)
        pred = model.forward(x)
        npt.assert_allclose(pred.shape, y.shape)

    def test_gaussian_embedding_loss(self):
        # test for improvement in loss
        pass


class TestGaussianDecoder(unittest.TestCase):

    def setUp(self):
        num_samples = 100
        num_microbes = 10
        self.input, self.output = bimodal(num_samples, num_microbes)

    def test_gaussian_decoder_forward(self):
        model = GaussianDecoder(self.num_microbes, 1)
        pred = model.forward(x)
        npt.assert_allclose(pred.shape, y.shape)

    def test_gaussian_decoder_loss(self):
        # test for improvement in loss
        pass


if __name__ == "__main__":
    unittest.main()
