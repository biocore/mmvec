import biom
import pandas as pd
import numpy as np
import tensorflow as tf
from skbio.stats.composition import clr, clr_inv
from qiime2.plugin import Metadata
from rhapsody.multimodal import Autoencoder
from rhapsody.util import split_tables
from scipy.sparse import coo_matrix


def autoencoder(microbes: biom.Table,
                metabolites: biom.Table,
                metadata: Metadata=None,
                training_column: str=None,
                num_testing_examples: int=5,
                min_feature_count: int=10,
                epochs: int=100,
                batch_size: int=50,
                latent_dim: int=3,
                input_prior: float=1,
                output_prior: float=1,
                learning_rate: float=0.001,
                summary_interval: int=60) -> pd.DataFrame:

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

    with tf.Graph().as_default(), tf.Session() as session:
        model = Autoencoder(
            latent_dim=latent_dim,
            u_scale=input_prior, v_scale=output_prior,
            learning_rate=learning_rate)
        model(session,
              train_microbes_coo, train_metabolites_df.values,
              test_microbes_coo, test_metabolites_df.values)

        loss, cv = model.fit(epoch=epochs, summary_interval=summary_interval)

        U, V = model.U, model.V

        U_ = np.hstack(
            (np.ones((model.U.shape[0], 1)),
             model.Ubias.reshape(-1, 1), U)
        )
        V_ = np.vstack(
            (model.Vbias.reshape(1, -1),
             np.ones((1, model.V.shape[1])), V)
        )

        ranks = pd.DataFrame(
            clr(clr_inv(np.hstack(
                (np.zeros((model.U.shape[0], 1)), U_ @ V_)))),
            index=train_microbes_df.columns,
            columns=train_metabolites_df.columns)

        return ranks
