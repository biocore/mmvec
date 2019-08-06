import unittest
import numpy as np
from biom import Table
from rhapsody.dataset import PairedDataset
import numpy.testing as npt


class TestPairedDataset(unittest.TestCase):

    def setUp(self):
        d1 = 20
        d2 = 30
        n = 10
        np.random.seed(0)
        self.oids = list(map(lambda x: 'o%d' % x, np.arange(d1)))
        self.mids = list(map(lambda x: 'o%d' % x, np.arange(d2)))
        self.sids = list(map(lambda x: 's%d' % x, np.arange(n)))
        self.microbes = Table(
            np.random.rand(d1, n),
            self.oids, self.sids
        )
        self.metabolites = Table(
            np.random.rand(d2, n),
            self.mids, self.sids[::-1]
        )

    def test_constructor(self):
        # make sure that the samples are aligned
        dataset = PairedDataset(self.microbes, self.metabolites)
        self.assertListEqual(list(dataset.microbes.ids(axis='sample')),
                             self.sids)
        self.assertListEqual(list(dataset.metabolites.ids(axis='sample')),
                             self.sids)


if __name__ == "__main__":
    unittest.main()
