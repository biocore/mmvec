import biom
import pandas as pd
from skbio import OrdinationResults
from qiime2.plugin import Metadata
from mmvec.mmvec import run_mmvec


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
          arm_the_gpu: bool = False) -> (pd.DataFrame, OrdinationResults):

    if metadata is not None:
        metadata = metadata.to_dataframe()

    # note that there are no checkpoints
    _, ranks, ordination = run_mmvec(
        microbes=microbes, metabolites=metabolites,
        metadata=metadata,
        training_column=training_column,
        num_testing_examples=num_testing_examples,
        min_feature_count=min_feature_count,
        epochs=epochs,
        batch_size=batch_size,
        latent_dim=latent_dim,
        input_prior=input_prior,
        output_prior=output_prior,
        beta1=0.9, beta2=0.99,
        num_workers=num_workers,
        learning_rate=learning_rate,
        arm_the_gpu=arm_the_gpu,
        checkpoint_interval=-1)

    return ranks, ordination
