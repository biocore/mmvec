import numpy as np
import math
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import IterableDataset


class PairedDataset(Dataset):
    """Paired microbe/metabolite dataset."""

    def __init__(self, microbes, metabolites):
        """
        Parameters
        ----------
        microbe_file : biom.Table
            Biom table of microbe abundances
        metabolite_file : biom.Table
            Biom table of metabolite abundances
        sample_metadata : pd.DataFrame
            Sample metadata
        """
        self.microbes = microbes
        self.metabolites = metabolites

        # match samples
        common_samples = set(self.microbes.ids(axis='sample')) & \
            set(self.metabolites.ids(axis='sample'))
        filter_fn = lambda v, i, m: i in common_samples
        self.microbes = self.microbes.filter(filter_fn, axis='sample')
        self.metabolites = self.metabolites.filter(filter_fn, axis='sample')
        self.microbes = self.microbes.sort()
        self.metabolites = self.metabolites.sort()
        self._microbes = self.microbes.matrix_data.T
        self._metabolites = self.metabolites.matrix_data.T

    def __len__(self):
        return len(self.microbes.ids())

    def __getitem__(self, idx):
        row = self._microbes.getrow(idx)
        cnts = np.hstack([row.data[row.indptr[i]:row.indptr[i + 1]]
                          for i in range(len(row.indptr) - 1)])
        feats = np.hstack([row.indices[row.indptr[i]:row.indptr[i + 1]]
                           for i in range(len(row.indptr) - 1)])
        microbe_seq = np.random.choice(feats, p=cnts / np.sum(cnts), size=1)

        row = self._metabolites.getrow(idx)
        metabolite_cnts = row.todense()
        s = microbe_seq.item()
        m = torch.from_numpy(np.ravel(metabolite_cnts)).float()
        return s, m


class IterablePairedDataset(IterableDataset):
    """Paired microbe/metabolite dataset iterable."""

    def __init__(self, microbes, metabolites, subsample_size=100):
        """
        Parameters
        ----------
        microbe_file : biom.Table
            Biom table of microbe abundances
        metabolite_file : biom.Table
            Biom table of metabolite abundances
        subsample_size : int
            Number of sequences to subsample per iteration from a sample.
        """
        self.microbes = microbes
        self.metabolites = metabolites

        # match samples
        common_samples = set(self.microbes.ids(axis='sample')) & \
            set(self.metabolites.ids(axis='sample'))
        filter_fn = lambda v, i, m: i in common_samples
        self.microbes = self.microbes.filter(filter_fn, axis='sample')
        self.metabolites = self.metabolites.filter(filter_fn, axis='sample')
        self.microbes = self.microbes.sort()
        self.metabolites = self.metabolites.sort()
        self._microbes = self.microbes.matrix_data.T
        self._metabolites = self.metabolites.matrix_data.T

        self.subsample_size = subsample_size

    def subsample(self, start, end):
        idx = np.random.randint(start, end)
        row = self._microbes.getrow(idx)
        cnts = np.hstack([row.data[row.indptr[i]:row.indptr[i + 1]]
                          for i in range(len(row.indptr) - 1)])
        feats = np.hstack([row.indices[row.indptr[i]:row.indptr[i + 1]]
                           for i in range(len(row.indptr) - 1)])
        microbe_seq = np.random.choice(feats, p=cnts / np.sum(cnts),
                                       size=self.subsample_size)

        row = self._metabolites.getrow(idx)
        metabolite_cnts = row.todense()
        return microbe_seq, metabolite_cnts

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        start = 0
        end = len(self.microbes.ids())
        if worker_info is None:  # single-process data loading
            for _ in range(end):
                s, m = self.subsample(start, end)
                m = torch.from_numpy(np.ravel(m)).float()
                for sj in s:
                    yield sj.item(), m
        else:
            # setup bounds
            t = (end - start)
            w = float(worker_info.num_workers)
            per_worker = int(math.ceil(t / w))
            worker_id = worker_info.id
            iter_start = start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, end)
            for _ in range(iter_start, iter_end):
                s, m = self.subsample(iter_start, iter_end)
                m = torch.from_numpy(np.ravel(m)).float()
                for sj in s:
                    yield sj.item(), m

    def __getitem__(self, idx):
        row = self._microbes.getrow(idx)
        cnts = np.hstack([row.data[row.indptr[i]:row.indptr[i + 1]]
                          for i in range(len(row.indptr) - 1)])
        feats = np.hstack([row.indices[row.indptr[i]:row.indptr[i + 1]]
                           for i in range(len(row.indptr) - 1)])
        microbe_seq = np.random.choice(feats, p=cnts / np.sum(cnts), size=1)

        row = self._metabolites.getrow(idx)
        metabolite_cnts = row.todense()
        s = microbe_seq.item()
        m = torch.from_numpy(np.ravel(metabolite_cnts)).float()
        return s, m


def split_tables(otu_table, metabolite_table,
                 metadata=None, training_column=None, num_test=10,
                 min_samples=10, iterable=False):
    """ Splits otu and metabolite tables into training and testing datasets.

    Parameters
    ----------
    otu_table : biom.Table
       Table of microbe abundances
    metabolite_table : biom.Table
       Table of metabolite intensities
    metadata : pd.DataFrame
       DataFrame of sample metadata information.  This is primarily used
       to indicated training and testing samples
    training_column : str
       The column used to indicate training and testing samples.
       Samples labeled 'Train' are allocated to the training set.
       All other samples are placed in the testing dataset.
    num_test : int
       If metadata or training_column is not specified, then `num_test`
       indicates how many testing samples will be allocated for
       cross validation.
    min_samples : int
       The minimum number of samples a microbe needs to be observed in
       in order to not get filtered out

    Returns
    -------
    train_datset: PairedDataset
       Torch dataset object composed of paired microbe/metabolite data
       for training.
    test_datset: PairedDataset
       Torch dataset object composed of paired microbe/metabolite data
       for testing/cross-validation.

    Notes
    -----
    There is an inefficient conversion from a sparse matrix to a
    dense matrix.  This may become a bottleneck later.
    """

    if metadata is None or training_column is None:
        sample_ids = np.random.permutation(otu_table.ids())
        train_ids = set(sample_ids[:-num_test])
        test_ids = set(sample_ids[-num_test:])
    else:
        if len(set(metadata[training_column]) & {'Train', 'Test'}) == 0:
            raise ValueError(
                "Training column must only specify `Train` and `Test`"
                "values"
            )
        train_ids = set(
            metadata.loc[metadata[training_column] == 'Train'].index)
        test_ids = set(
            metadata.loc[metadata[training_column] == 'Test'].index)

    train_filter = lambda v, i, m: i in train_ids
    test_filter = lambda v, i, m: i in test_ids
    train_microbes = otu_table.filter(train_filter, inplace=False)
    test_microbes = otu_table.filter(test_filter, inplace=False)

    train_metabolites = metabolite_table.filter(
        train_filter, inplace=False)
    test_metabolites = metabolite_table.filter(
        test_filter, inplace=False)

    if iterable:
        train_dataset = IterablePairedDataset(
            train_microbes, train_metabolites)
        test_dataset = IterablePairedDataset(
            test_microbes, test_metabolites)
    else:
        train_dataset = PairedDataset(
            train_microbes, train_metabolites)
        test_dataset = PairedDataset(
            test_microbes, test_metabolites)

    return train_dataset, test_dataset
