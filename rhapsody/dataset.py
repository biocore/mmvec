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
