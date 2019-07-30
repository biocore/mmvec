import unittest
import pandas as pd
from rhapsody.heatmap import (
    _parse_taxonomy_strings, _parse_heatmap_metadata_annotations)
import pandas.util.testing as pdt


class TestParseTaxonomyStrings(unittest.TestCase):

    def setUp(self):
        self.taxa = pd.Series([
            'k__Bacteria; p__Proteobacteria; c__Deltaproteobacteria; '
            'o__Desulfobacterales; f__Desulfobulbaceae; g__; s__',
            'k__Bacteria; p__Cyanobacteria; c__Chloroplast; o__Streptophyta',
            'k__Bacteria; p__Proteobacteria; c__Alphaproteobacteria; '
            'o__Rickettsiales; f__mitochondria; g__Lardizabala; s__biternata',
            'k__Archaea; p__Euryarchaeota; c__Methanomicrobia; '
            'o__Methanosarcinales; f__Methanosarcinaceae; g__Methanosarcina',
            'k__Bacteria; p__Proteobacteria; c__Alphaproteobacteria; '
            'o__Rickettsiales; f__mitochondria; g__Pavlova; s__lutheri',
            'k__Archaea; p__[Parvarchaeota]; c__[Parvarchaea]; o__WCHD3-30',
            'k__Bacteria; p__Proteobacteria; c__Alphaproteobacteria; '
            'o__Sphingomonadales; f__Sphingomonadaceae'],
            index=pd.Index([c for c in 'ABCDEFG'], name='feature-id'),
            name='Taxon')
        self.exp = pd.Series(
            ['s__', 'o__Streptophyta', 's__biternata', 'g__Methanosarcina',
             's__lutheri', 'o__WCHD3-30', 'f__Sphingomonadaceae'],
            index=pd.Index([c for c in 'ABCDEFG'], name='feature-id'),
            name='Taxon')

    def test_parse_taxonomy_strings(self):
        exp = pd.Series(['p__Proteobacteria', 'p__Cyanobacteria',
                         'p__Proteobacteria', 'p__Euryarchaeota',
                         'p__Proteobacteria', 'p__[Parvarchaeota]',
                         'p__Proteobacteria'],
                        index=pd.Index([c for c in 'ABCDEFG'],
                        name='feature-id'), name='Taxon')
        obs = _parse_taxonomy_strings(self.taxa, level=2)
        pdt.assert_series_equal(exp, obs)

    def test_parse_taxonomy_strings_baserank(self):
        exp = pd.Series(['k__Bacteria', 'k__Bacteria', 'k__Bacteria',
                         'k__Archaea', 'k__Bacteria', 'k__Archaea',
                         'k__Bacteria'],
                        index=pd.Index([c for c in 'ABCDEFG'],
                        name='feature-id'), name='Taxon')
        obs = _parse_taxonomy_strings(self.taxa, level=1)
        pdt.assert_series_equal(exp, obs)

    def test_parse_taxonomy_strings_toprank(self):
        # expect top rank even if level is higher than depth of top rank
        obs = _parse_taxonomy_strings(self.taxa, level=7)
        pdt.assert_series_equal(self.exp, obs)

    def test_parse_taxonomy_strings_rank_out_of_range_is_top(self):
        # expect top rank even if level is higher than depth of top rank
        obs = _parse_taxonomy_strings(self.taxa, level=9)
        pdt.assert_series_equal(self.exp, obs)


class TestHeatmapAnnotation(unittest.TestCase):

    def setUp(self):
        self.taxonomy = pd.Series(
            ['k__Bacteria', 'k__Archaea', 'k__Bacteria', 'k__Archaea'],
            index=pd.Index([c for c in 'ABCD'], name='id'), name='Taxon')

    def test_parse_heatmap_metadata_annotations_colorhelix(self):
        exp_cols = pd.Series(
            [[0.8377187772618228, 0.7593149036488329, 0.9153517040128891],
             [0.2539759281991313, 0.3490084835469758, 0.14482988411775732],
             [0.8377187772618228, 0.7593149036488329, 0.9153517040128891],
             [0.2539759281991313, 0.3490084835469758, 0.14482988411775732]],
            index=pd.Index([c for c in 'ABCD'], name='id'), name='Taxon')
        exp_classes = {'k__Archaea': [0.2539759281991313, 0.3490084835469758,
                                      0.14482988411775732],
                       'k__Bacteria': [0.8377187772618228, 0.7593149036488329,
                                       0.9153517040128891]}
        cols, classes = _parse_heatmap_metadata_annotations(
            self.taxonomy, 'colorhelix')
        pdt.assert_series_equal(exp_cols, cols)
        self.assertDictEqual(exp_classes, classes)

    def test_parse_heatmap_metadata_annotations_magma(self):
        exp_cols = pd.Series(
            [(0.944006, 0.377643, 0.365136), (0.445163, 0.122724, 0.506901),
             (0.944006, 0.377643, 0.365136), (0.445163, 0.122724, 0.506901)],
            index=pd.Index([c for c in 'ABCD'], name='id'), name='Taxon')
        exp_classes = {'k__Archaea': (0.445163, 0.122724, 0.506901),
                       'k__Bacteria': (0.944006, 0.377643, 0.365136)}
        cols, classes = _parse_heatmap_metadata_annotations(
            self.taxonomy, 'magma')
        pdt.assert_series_equal(exp_cols, cols)
        self.assertDictEqual(exp_classes, classes)
