import unittest
from numpy.testing._private.utils import assert_equal
import torch
from mmvec.ALR import ranks_bare

class TestBareFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.u = torch.rand(4, 5)
        self.v = torch.rand(5, 3)
        return super().setUp()

    def test_ranks(self):
        ranks = ranks_bare(self.u, self.v)
        print(ranks)

        assert_equal(ranks.shape[0], self.u.shape[0])
        assert_equal(ranks.shape[1], (self.v.shape[1] + 1))
