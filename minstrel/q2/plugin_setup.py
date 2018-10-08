# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import qiime2.plugin
import qiime2.sdk
from minstrel import __version__
from ._method import autoencoder
from qiime2.plugin import (Str, Properties, Int, Float,  Metadata)
from q2_types.feature_table import FeatureTable, Composition, Frequency
from q2_types.ordination import PCoAResults


# citations = qiime2.plugin.Citations.load(
#             'citations.bib', package='minstrel')

plugin = qiime2.plugin.Plugin(
    name='minstrel',
    version=__version__,
    website="https://github.com/mortonjt/minstrel",
    # citations=[citations['morton2017balance']],
    short_description=('Plugin for performing microbe-metabolite '
                       'co-occurence analysis.'),
    description=('This is a QIIME 2 plugin supporting microbe-metabolite '
                 'co-occurence analysis using multimodal autoencoders.'),
    package='minstrel')

plugin.methods.register_function(
    function=autoencoder,
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
        ('conditional_ranks',
         FeatureTable[Composition % Properties('coefficients')])
    ],
    input_descriptions={
        'microbes': 'Input table of microbial counts.',
        'metabolites': 'Input table of metabolite intensities.',
    },
    parameter_descriptions={
        'metadata': 'Sample metadata table with covariates of interest.',
        'training_column': ("The metadata column specifying which "
                            "samples are for training/testing. "
                            "Entries must be marked `True` for training examples "
                            "and `False` for testing examples. "),
        'num_testing_examples': ("The number of random examples to select "
                                 "if `training_column` isn't specified"),
        'epochs': ('The number of total number of iterations '
                  'over the entire dataset'),
        'batch_size': ('The number of samples to be evaluated per '
                       'training iteration'),
        'input_prior': ('Width of normal prior for the microbial '
                        'coefficients .Smaller values will regularize '
                        'parameters towards zero. Values must be greater '
                        'than 0.'),
        'output_prior': ('Width of normal prior for the metabolite '
                         'coefficients. Smaller values will regularize '
                         'parameters towards zero. Values must be greater '
                         'than 0.'),
        'learning_rate': ('Gradient descent decay rate.'),

    },
    name='Multimodal autoencoder',
    description=("Performs bi-loglinear multinomial regression and calculates "
                 "the conditional probability ranks of metabolite co-occurence "
                 "given the microbe presence."),
    citations=[]
)
