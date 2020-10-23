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
                           MetadataColumn, Categorical, Range, Choices, List)
from q2_types.feature_table import FeatureTable, Frequency
from q2_types.feature_data import FeatureData
from q2_types.sample_data import SampleData
from q2_types.ordination import PCoAResults
from mmvec.q2 import (
    Conditional, ConditionalFormat, ConditionalDirFmt,
    MMvecStats, MMvecStatsFormat, MMvecStatsDirFmt,
    paired_omics, heatmap, paired_heatmap, summarize_single, summarize_paired
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
        'equalize_biplot': Bool,
        'summary_interval': Int
    },
    outputs=[
        ('conditionals', FeatureData[Conditional]),
        ('conditional_biplot', PCoAResults % Properties('biplot')),
        ('model_stats', SampleData[MMvecStats]),
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
        'equalize_biplot': 'Biplot arrows and points are on the same scale.',
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
        'row_center': Bool,
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
        'row_center': 'Center conditional probability table '
                      'around average row.'
    },
    name='Conditional probability heatmap',
    description="Generate heatmap depicting mmvec conditional probabilities.",
    citations=[]
)

plugin.visualizers.register_function(
    function=paired_heatmap,
    inputs={'ranks': FeatureData[Conditional],
            'microbes_table': FeatureTable[Frequency],
            'metabolites_table': FeatureTable[Frequency]},
    parameters={
        'microbe_metadata': MetadataColumn[Categorical],
        'features': List[Str],
        'top_k_microbes': Int % Range(0, None),
        'color_palette': Str % Choices(_cmaps['heatmap']),
        'normalize': Str % Choices(['log10', 'z_score_col', 'z_score_row',
                                    'rel_row', 'rel_col', 'None']),
        'top_k_metabolites': Int % Range(1, None) | Str % Choices(['all']),
        'keep_top_samples': Bool,
        'level': Int % Range(-1, None),
        'row_center': Bool,
    },
    input_descriptions={'ranks': 'Conditional probabilities.',
                        'microbes_table': 'Microbial feature abundances.',
                        'metabolites_table': 'Metabolite feature abundances.'},
    parameter_descriptions={
        'microbe_metadata': 'Optional microbe metadata for annotating plots.',
        'features': 'Microbial feature IDs to display in heatmap. Use this '
                    'parameter to include named feature IDs in the heatmap. '
                    'Can be used in conjunction with top_k_microbes, in which '
                    'case named features will be displayed first, then top '
                    'microbial features in order of log conditional '
                    'probability maximum values.',
        'top_k_microbes': 'Select top k microbes (those with the highest '
                          'relative abundances) to display on the heatmap. '
                          'Set to "all" to display all metabolites.',
        'color_palette': 'Color palette for clustermap.',
        'normalize': 'Optionally normalize heatmap values by columns or rows.',
        'top_k_metabolites': 'Select top k metabolites associated with each '
                             'of the chosen features to display on heatmap.',
        'keep_top_samples': 'Display only samples in which at least one of '
                            'the selected microbes is the most abundant '
                            'feature.',
        'level': 'taxonomic level for annotating clustermap. Set to -1 if not '
                 'parsing semicolon-delimited taxonomies or wish to print '
                 'entire annotation.',
        'row_center': 'Center conditional probability table '
                      'around average row.'
    },
    name='Paired feature abundance heatmaps',
    description="Generate paired heatmaps that depict microbial and "
                "metabolite feature abundances. The left panel displays the "
                "abundance of each selected microbial feature in each sample. "
                "The right panel displays the abundances of the top k "
                "metabolites most highly correlated with these microbes in "
                "each sample. The y-axis (sample axis) is shared between each "
                "panel.",
    citations=[]
)


plugin.visualizers.register_function(
    function=summarize_single,
    inputs={
        'model_stats': SampleData[MMvecStats]
    },
    parameters={},
    input_descriptions={
        'model_stats': (
            "Summary information produced by running "
            "`qiime mmvec paired-omics`."
        )
    },
    parameter_descriptions={
    },
    name='MMvec summary statistics',
    description=(
        "Visualize the convergence statistics from running "
        "`qiime mmvec paired-omics`regression, giving insight "
        "into how the model fit to your data."
    )
)

plugin.visualizers.register_function(
    function=summarize_paired,
    inputs={
        'model_stats': SampleData[MMvecStats],
        'baseline_stats': SampleData[MMvecStats]
    },
    parameters={},
    input_descriptions={

        'model_stats': (
            "Summary information for the reference model, produced by running "
            "`qiime mmvec paired-omics`."
        ),
        'baseline_stats': (
            "Summary information for the baseline model, produced by running "
            "`qiime mmvec paired-omics`."
        )

    },
    parameter_descriptions={
    },
    name='Paired regression summary statistics',
    description=(
        "Visualize the convergence statistics from two MMvec models, "
        "giving insight into how the models fit to your data. "
        "The produced visualization includes a 'pseudo-Q-squared' value."
    )
)

# Register types
plugin.register_formats(MMvecStatsFormat, MMvecStatsDirFmt)
plugin.register_semantic_types(MMvecStats)
plugin.register_semantic_type_to_format(
    SampleData[MMvecStats], MMvecStatsDirFmt)

plugin.register_formats(ConditionalFormat, ConditionalDirFmt)
plugin.register_semantic_types(Conditional)
plugin.register_semantic_type_to_format(
    FeatureData[Conditional], ConditionalDirFmt)

importlib.import_module('mmvec.q2._transformer')
