import unittest
import pandas as pd
from qiime2 import Artifact, CategoricalMetadataColumn
from qiime2.plugins import mmvec


# these tests just make sure the visualizer runs; nuts + bolts are tested in
# the main package.
class TestHeatmap(unittest.TestCase):

    def setUp(self):
        _ranks = pd.DataFrame([[4.1, 1.3, 2.1], [0.1, 0.3, 0.2],
                               [2.2, 4.3, 3.2], [-6.3, -4.4, 2.1]],
                              index=pd.Index([c for c in 'ABCD'], name='id'),
                              columns=['m1', 'm2', 'm3'])
        self.ranks = Artifact.import_data('FeatureData[Conditional]', _ranks)
        self.taxa = CategoricalMetadataColumn(pd.Series([
            'k__Bacteria; p__Proteobacteria; c__Deltaproteobacteria; '
            'o__Desulfobacterales; f__Desulfobulbaceae; g__; s__',
            'k__Bacteria; p__Cyanobacteria; c__Chloroplast; o__Streptophyta',
            'k__Bacteria; p__Proteobacteria; c__Alphaproteobacteria; '
            'o__Rickettsiales; f__mitochondria; g__Lardizabala; s__biternata',
            'k__Archaea; p__Euryarchaeota; c__Methanomicrobia; '
            'o__Methanosarcinales; f__Methanosarcinaceae; g__Methanosarcina'],
            index=pd.Index([c for c in 'ABCD'], name='feature-id'),
            name='Taxon'))
        self.metabolites = CategoricalMetadataColumn(pd.Series([
            'amino acid', 'carbohydrate', 'drug metabolism'],
            index=pd.Index(['m1', 'm2', 'm3'], name='feature-id'),
            name='Super Pathway'))

    def test_heatmap_default(self):
        mmvec.actions.heatmap(self.ranks, self.taxa, self.metabolites)

    def test_heatmap_no_metadata(self):
        mmvec.actions.heatmap(self.ranks)

    def test_heatmap_one_metadata(self):
        mmvec.actions.heatmap(self.ranks, self.taxa, None)

    def test_heatmap_no_taxonomy_parsing(self):
        mmvec.actions.heatmap(self.ranks, self.taxa, None, level=-1)

    def test_heatmap_plot_axis_labels(self):
        mmvec.actions.heatmap(self.ranks, x_labels=True, y_labels=True)


if __name__ == "__main__":
    unittest.main()
