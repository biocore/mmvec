import biom
import pandas as pd
import numpy as np
from skbio import OrdinationResults
import qiime2
from qiime2.plugin import Metadata
from mmvec.train import mmvec_training_loop
from mmvec.ALR import MMvecALR
from mmvec.util import split_tables
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds


def paired_omics(microbes: biom.Table,
                 metabolites: biom.Table,
                 metadata: qiime2.Metadata = None,
                 training_column: str = None,
                 num_testing_examples: int = 5,
                 min_feature_count: int = 10,
                 epochs: int = 100,
                 batch_size: int = 50,
                 latent_dim: int = 3,
                 input_prior: float = 1,
                 output_prior: float = 1,
                 learning_rate: float = 1e-3,
                 equalize_biplot: float = False,
                 arm_the_gpu: bool = False,
                 summary_interval: int = 60) -> (
                         pd.DataFrame, OrdinationResults, qiime2.Metadata
                         ):

    if metadata is not None:
        metadata = metadata.to_dataframe()

    #TODO refactor for pytorch!
    if arm_the_gpu:
        # pick out the first GPU
        device_name = '/device:GPU:0'
    else:
        device_name = '/cpu:0'

    # Note: there are a couple of biom -> pandas conversions taking
    # place here.  This is currently done on purpose, since we
    # haven't figured out how to handle sparse matrix multiplication
    # in the context of this algorithm.  That is a future consideration.
    res = split_tables(
        microbes, metabolites,
        metadata=metadata, training_column=training_column,
        num_test=num_testing_examples,
        min_samples=min_feature_count)

    (train_microbes_df, test_microbes_df,
     train_metabolites_df, test_metabolites_df) = res

    train_microbes_coo = coo_matrix(train_microbes_df.values)
    test_microbes_coo = coo_matrix(test_microbes_df.values)

    model = MMvecALR(
        microbes=microbes,
        metabolites= metabolites,
        latent_dim=latent_dim,
        sigma_u=input_prior, sigma_v=output_prior,
        )

    convergence_stats = pd.DataFrame.from_records(mmvec_training_loop(model=model,
        learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
        summary_interval=summary_interval)
        ,
        columns=['iteration','loss', 'cross-validation'])


    convergence_stats.astype({'loss': 'float', 'cross-validation':
        'float'}, copy=False)

    convergence_stats.set_index("iteration", inplace=True)
    convergence_stats.index.name="id"

    biplot = model.get_ordination()
    ranks = model.ranks_dataframe()
    return ranks, biplot, qiime2.Metadata(convergence_stats)
