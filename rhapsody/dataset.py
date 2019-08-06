import numpy as np
from biom import load_table
from torch.utils.data import Dataset
from multiprocessing import Pool


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
        self.metabolites = self.microbes.filter(filter_fn, axis='sample')
        self.microbes = self.microbes.sort()
        self.metabolites = self.metabolites.sort()

    def __len__(self):
        return len(self.microbes.ids())

    def __getitem__(self, idx):
        row = self.microbes.getrow(idx)
        microbe_cnts = np.hstack([row.data[row.indptr[i]:row.indptr[i+1]]
                                  for i in range(len(row.indptr)-1)])
        microbe_feats = np.hstack([row.indices[row.indptr[i]:row.indptr[i+1]]
                                   for i in range(len(row.indptr)-1)])
        microbe_seq = np.random.choice(feats, p=cnts/np.sum(cnts), size=1)

        row = self.metabolites.getrow(idx)
        metabolite_cnts = row.todense()
        return microbe_seq, metabolite_cnts

def split_tables(otu_table, metabolite_table,
                 metadata=None, training_column=None, num_test=10,
                 min_samples=10):
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
            raise ValueError("Training column must only specify `Train` and `Test`"
                             "values")
        train_ids = set(metadata.loc[metadata[training_column] == 'Train'].index)
        test_ids = set(metadata.loc[metadata[training_column] == 'Test'].index)

    def train_filter(v, i, m): return (i in train_ids)
    def test_filter(v, i, m): return (i in test_ids)
    train_microbes = otu_table.filter(train_filter)
    test_microbes = otu_table.filter(test_filter)

    train_metabolites = metabolite_table.filter(train_filter)
    test_metabolites = metabolite_table.filter(test_filter)

    train_dataset = PairedDataset(train_microbes, train_metabolites)
    test_dataset = PairedDataset(test_microbes, test_metabolites)

    return train_dataset, test_dataset
