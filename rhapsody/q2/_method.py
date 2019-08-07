import biom
import pandas as pd
import numpy as np
import tensorflow as tf
from skbio import OrdinationResults
from skbio.stats.composition import clr, clr_inv
from qiime2.plugin import Metadata
from rhapsody.util import split_tables
from rhapsody.mmvec import run_mmvec
from scipy.sparse import coo_matrix


def mmvec(microbes: biom.Table,
          metabolites: biom.Table,
          metadata: Metadata = None,
          training_column: str = None,
          num_testing_examples: int = 5,
          min_feature_count: int = 10,
          epochs: int = 100,
          batch_size: int = 50,
          latent_dim: int = 3,
          input_prior: float = 1,
          output_prior: float = 1,
          num_workers: int = 1,
          learning_rate: float = 0.001,
          arm_the_gpu: bool = False,
          summary_interval: int = 60) -> (pd.DataFrame, OrdinationResults):

    if metadata is not None:
        metadata = metadata.to_dataframe()

    _, ranks, ordination = run_mmvec(
        microbes, metabolites, metadata,
        training_column, num_testing_examples,
        min_feature_count, epochs,
        batch_size, latent_dim,
        input_prior, output_prior,
        num_workers, learning_rate, arm_the_gpu,
        summary_interval, checkpoint_interval=-1)

    return ranks, ordination
