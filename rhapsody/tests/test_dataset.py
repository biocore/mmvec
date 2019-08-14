import unittest
import numpy as np
import pandas as pd
from biom import Table
from rhapsody.dataset import PairedDataset
from rhapsody.dataset import split_tables
import numpy.testing as npt


class TestPairedDataset(unittest.TestCase):

    def setUp(self):
        d1 = 200
        d2 = 30
        n = 1000
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

    def test_get_item(self):
        dataset = PairedDataset(self.microbes, self.metabolites)
        for i in range(self.microbes.shape[1]):
            dataset[i]


class TestSplitTables(unittest.TestCase):

    def setUp(self):

        omat = np.array([
            [104, 10, 2, 0, 0],
            [4, 100, 20, 0, 0],
            [0, 1, 0, 0, 4],
            [4, 0, 21, 0, 2],
            [40, 0, 2, 1, 39],
            [0, 0, 32, 10, 3],
            [59, 1, 0, 0, 3]
        ])
        mmat = np.array([
            [104, 1, 31, 0, 8],
            [4, 100, 20, 0, 0],
            [0, 8, 0, 0, 4],
            [0, 0, 2, 1, 2],
            [0, 0, 20, 10, 3],
            [0, 8, 0, 0, 4],
            [0, 0, 2, 10, 3],
            [0, 0, 320, 139, 3],
            [59, 9, 0, 0, 33]
        ]) * 10e6

        oids = list(map(lambda x: 'o'+str(x), np.arange(omat.shape[0])))
        mids = list(map(lambda x: 'm'+str(x), np.arange(mmat.shape[0])))
        sids = list(map(lambda x: 'm'+str(x), np.arange(mmat.shape[1])))

        self.otu_table = Table(omat, oids, sids)
        self.metabolite_table = Table(mmat, mids, sids)

        self.metadata = pd.DataFrame(
            {
                'testing': ['Train', 'Test', 'Train', 'Test', 'Train'],
                'bad': [True, False, True, False, True]
            }, index=sids
        )

    def test_split_tables_train_column(self):

        train, test = split_tables(self.otu_table, self.metabolite_table,
                                   metadata=self.metadata,
                                   training_column='testing',
                                   num_test=10, min_samples=0)

        npt.assert_allclose(train.microbes.shape, np.array([7, 3]))
        npt.assert_allclose(test.microbes.shape, np.array([7, 2]))
        npt.assert_allclose(train.metabolites.shape, np.array([9, 3]))
        npt.assert_allclose(test.metabolites.shape, np.array([9, 2]))

    def test_split_tables_bad_column(self):
        with self.assertRaises(Exception):
            split_tables(self.otu_table, self.metabolite_table,
                         metadata=self.metadata, training_column='bad',
                         num_test=10, min_samples=0)

    def test_split_tables_random(self):
        train, test = split_tables(self.otu_table, self.metabolite_table,
                                   num_test=2, min_samples=0)

        npt.assert_allclose(train.microbes.shape, np.array([7, 3]))
        npt.assert_allclose(test.microbes.shape, np.array([7, 2]))
        npt.assert_allclose(train.metabolites.shape, np.array([9, 3]))
        npt.assert_allclose(test.metabolites.shape, np.array([9, 2]))

    def test_split_tables_random_filter(self):
        train, test = split_tables(self.otu_table, self.metabolite_table,
                                   num_test=2, min_samples=2)

        npt.assert_allclose(train.microbes.shape, np.array([7, 3]))
        npt.assert_allclose(test.microbes.shape, np.array([7, 2]))
        npt.assert_allclose(train.metabolites.shape, np.array([9, 3]))
        npt.assert_allclose(test.metabolites.shape, np.array([9, 2]))


if __name__ == "__main__":
    unittest.main()
