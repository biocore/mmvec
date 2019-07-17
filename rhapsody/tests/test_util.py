import unittest
import numpy as np
import pandas as pd
from biom import Table
from rhapsody.util import (onehot, rank_hits,
                           random_multimodal, split_tables,
                           embeddings2ranks)
from skbio.util import get_data_path
import numpy.testing as npt
import pandas.util.testing as pdt


class TestOnehot(unittest.TestCase):
    def setUp(self):
        seed = 0
        # build small simulation
        res = random_multimodal(
            uB=-5,
            num_microbes=2, num_metabolites=4, num_samples=10,
            latent_dim=2, low=-1, high=1,
            microbe_total=10, metabolite_total=10,
            seed=seed
        )
        (self.microbes, self.metabolites, self.X, self.B,
         self.U, self.Ubias, self.V, self.Vbias) = res

    def test_onehot(self):
        otu_hits, _ = onehot(self.microbes.values)
        np.savetxt(get_data_path('otu_hits.txt'), otu_hits)
        exp_otu_hits = np.loadtxt(get_data_path('otu_hits.txt'))
        npt.assert_allclose(exp_otu_hits, otu_hits)

    def test_onehot_simple(self):
        seed = 0
        # build small simulation
        res = random_multimodal(
            uB=-5,
            num_microbes=2, num_metabolites=2, num_samples=3,
            latent_dim=1, low=-1, high=1,
            microbe_total=3, metabolite_total=3,
            seed=seed
        )

        (microbes, metabolites, X, B, U, V, Ubias, Vbias) = res
        otu_hits, sample_ids = onehot(microbes.values)
        exp_otu_hits = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])

        exp_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        npt.assert_allclose(exp_ids, sample_ids)
        npt.assert_allclose(exp_otu_hits, otu_hits)


class TestRankHits(unittest.TestCase):

    def test_rank_hits(self):
        ranks = pd.DataFrame(
            [
                [1., 4., 1., 5., 7.],
                [2., 6., 9., 2., 8.],
                [2., 2., 6., 8., 4.]
            ],
            index=['OTU_1', 'OTU_2', 'OTU_3'],
            columns=['MS_1', 'MS_2', 'MS_3', 'MS_4', 'MS_5']
        )
        res = rank_hits(ranks, k=2)
        exp = pd.DataFrame(
            [
                ['OTU_1', 5., 'MS_4'],
                ['OTU_2', 8., 'MS_5'],
                ['OTU_3', 6., 'MS_3'],
                ['OTU_1', 7., 'MS_5'],
                ['OTU_2', 9., 'MS_3'],
                ['OTU_3', 8., 'MS_4']
            ], columns=['src', 'rank', 'dest'],
        )

        pdt.assert_frame_equal(res, exp)


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

        res = split_tables(self.otu_table, self.metabolite_table,
                           metadata=self.metadata, training_column='testing',
                           num_test=10, min_samples=0)

        (train_microbes, test_microbes,
         train_metabolites, test_metabolites) = res

        npt.assert_allclose(train_microbes.shape, np.array([3, 7]))
        npt.assert_allclose(test_microbes.shape, np.array([2, 7]))
        npt.assert_allclose(train_metabolites.shape, np.array([3, 9]))
        npt.assert_allclose(test_metabolites.shape, np.array([2, 9]))

    def test_split_tables_bad_column(self):
        with self.assertRaises(Exception):
            split_tables(self.otu_table, self.metabolite_table,
                         metadata=self.metadata, training_column='bad',
                         num_test=10, min_samples=0)

    def test_split_tables_random(self):
        res = split_tables(self.otu_table, self.metabolite_table,
                           num_test=2, min_samples=0)

        (train_microbes, test_microbes,
         train_metabolites, test_metabolites) = res
        npt.assert_allclose(train_microbes.shape, np.array([3, 7]))
        npt.assert_allclose(test_microbes.shape, np.array([2, 7]))
        npt.assert_allclose(train_metabolites.shape, np.array([3, 9]))
        npt.assert_allclose(test_metabolites.shape, np.array([2, 9]))

    def test_split_tables_random_filter(self):
        res = split_tables(self.otu_table, self.metabolite_table,
                           num_test=2, min_samples=2)

        (train_microbes, test_microbes,
         train_metabolites, test_metabolites) = res
        npt.assert_allclose(train_microbes.shape, np.array([3, 6]))
        npt.assert_allclose(test_microbes.shape, np.array([2, 6]))
        npt.assert_allclose(train_metabolites.shape, np.array([3, 9]))
        npt.assert_allclose(test_metabolites.shape, np.array([2, 9]))


if __name__ == "__main__":
    unittest.main()
