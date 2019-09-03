# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import importlib
import qiime2.plugin
import qiime2.sdk
from mmvec import __version__, _heatmap_choices, _cmaps
from qiime2.plugin import (Str, Properties, Int, Float, Metadata, Bool,
                           MetadataColumn, Categorical, Range, Choices)
from q2_types.feature_table import FeatureTable, Frequency
from q2_types.feature_data import FeatureData
from q2_types.ordination import PCoAResults

from mmvec.q2 import (
    Conditional, ConditionalFormat, ConditionalDirFmt,
    paired_omics, heatmap
)

plugin = qiime2.plugin.Plugin(
    name='mmvec',
    version=__version__,
    website="https://github.com/biocore/mmvec",
    short_description='Plugin for performing microbe-metabolite '
                      'co-occurence analysis.',
    description='This is a QIIME 2 plugin supporting microbe-metabolite '
                'co-occurence analysis using mmvec.',
    package='mmvec')

plugin.methods.register_function(
    function=paired_omics,
    inputs={'microbes': FeatureTable[Frequency],
            'metabolites': FeatureTable[Frequency]},
    parameters={
        'metadata': Metadata,
        'training_column': Str,
        'num_testing_examples': Int,
        'min_feature_count': Int,
        'epochs': Int,
        'batch_size': Int,
        'latent_dim': Int,
        'input_prior': Float,
        'output_prior': Float,
        'learning_rate': Float,
        'summary_interval': Int
    },
    outputs=[
        ('conditionals', FeatureData[Conditional]),
        ('conditional_biplot', PCoAResults % Properties('biplot'))
    ],
    input_descriptions={
        'microbes': 'Input table of microbial counts.',
        'metabolites': 'Input table of metabolite intensities.',
    },
    output_descriptions={
        'conditionals': 'Mean-centered Conditional log-probabilities.',
        'conditional_biplot': 'Biplot of microbe-metabolite vectors.',
    },
    parameter_descriptions={
        'metadata': 'Sample metadata table with covariates of interest.',
        'training_column': "The metadata column specifying which "
                           "samples are for training/testing. "
                           "Entries must be marked `Train` for training "
                           "examples and `Test` for testing examples. ",
        'num_testing_examples': "The number of random examples to select "
                                "if `training_column` isn't specified.",
        'epochs': 'The total number of iterations over the entire dataset.',
        'batch_size': 'The number of samples to be evaluated per '
                      'training iteration.',
        'input_prior': 'Width of normal prior for the microbial '
                       'coefficients. Smaller values will regularize '
                       'parameters towards zero. Values must be greater '
                       'than 0.',
        'output_prior': 'Width of normal prior for the metabolite '
                        'coefficients. Smaller values will regularize '
                        'parameters towards zero. Values must be greater '
                        'than 0.',
        'learning_rate': 'Gradient descent decay rate.'
    },
    name='Microbe metabolite vectors',
    description="Performs bi-loglinear multinomial regression and calculates "
                "the conditional probability ranks of metabolite "
                "co-occurence given the microbe presence.",
    citations=[]
)

plugin.visualizers.register_function(
    function=heatmap,
    inputs={'ranks': FeatureData[Conditional]},
    parameters={
        'microbe_metadata': MetadataColumn[Categorical],
        'metabolite_metadata': MetadataColumn[Categorical],
        'method': Str % Choices(_heatmap_choices['method']),
        'metric': Str % Choices(_heatmap_choices['metric']),
        'color_palette': Str % Choices(_cmaps['heatmap']),
        'margin_palette': Str % Choices(_cmaps['margins']),
        'x_labels': Bool,
        'y_labels': Bool,
        'level': Int % Range(-1, None),
    },
    input_descriptions={'ranks': 'Conditional probabilities.'},
    parameter_descriptions={
        'microbe_metadata': 'Optional microbe metadata for annotating plots.',
        'metabolite_metadata': 'Optional metabolite metadata for annotating '
                               'plots.',
        'method': 'Hierarchical clustering method used in clustermap.',
        'metric': 'Distance metric used in clustermap.',
        'color_palette': 'Color palette for clustermap.',
        'margin_palette': 'Name of color palette to use for annotating '
                          'metadata along margin(s) of clustermap.',
        'x_labels': 'Plot x-axis (metabolite) labels?',
        'y_labels': 'Plot y-axis (microbe) labels?',
        'level': 'taxonomic level for annotating clustermap. Set to -1 if not '
                 'parsing semicolon-delimited taxonomies or wish to print '
                 'entire annotation.',
    },
    name='Conditional probability heatmap',
    description="Generate heatmap depicting mmvec conditional probabilities.",
    citations=[]
)


plugin.register_formats(ConditionalFormat, ConditionalDirFmt)
plugin.register_semantic_types(Conditional)
plugin.register_semantic_type_to_format(
    FeatureData[Conditional], ConditionalDirFmt)

importlib.import_module('mmvec.q2._transformer')
