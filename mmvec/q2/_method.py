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
                 metadata: Metadata = None,
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

    #with tf.Graph().as_default(), tf.Session() as session:
    model = MMvecALR(
        microbes=microbes,
        metabolites= metabolites,
        latent_dim=latent_dim,
        sigma_u=input_prior, sigma_v=output_prior,
        )

    mmvec_training_loop(model=model, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, summary_interval=summary_interval)
    ranks = model.ranks_dataframe()
    #ranks = pd.DataFrame(model.ranks(), index=train_microbes_df.columns,
    #                     columns=train_metabolites_df.columns)
    if latent_dim > 0:
        u, s, v = svds(ranks - ranks.mean(axis=0), k=latent_dim)
    else:
        # fake it until you make it
        u, s, v = svds(ranks - ranks.mean(axis=0), k=1)

    ranks = ranks.T
    ranks.index.name = 'featureid'
    s = s[::-1]
    u = u[:, ::-1]
    v = v[::-1, :]
    if equalize_biplot:
        microbe_embed = u @ np.sqrt(np.diag(s))
        metabolite_embed = v.T @ np.sqrt(np.diag(s))
    else:
        microbe_embed = u @ np.diag(s)
        metabolite_embed = v.T

    pc_ids = ['PC%d' % i for i in range(microbe_embed.shape[1])]
    features = pd.DataFrame(
        microbe_embed, columns=pc_ids,
        index=train_microbes_df.columns)
    samples = pd.DataFrame(
        metabolite_embed, columns=pc_ids,
        index=train_metabolites_df.columns)
    short_method_name = 'mmvec biplot'
    long_method_name = 'Multiomics mmvec biplot'
    eigvals = pd.Series(s, index=pc_ids)
    proportion_explained = pd.Series(s**2 / np.sum(s**2), index=pc_ids)
    biplot = OrdinationResults(
        short_method_name, long_method_name, eigvals,
        samples=samples, features=features,
        proportion_explained=proportion_explained)

    its = np.arange(len(loss))
    convergence_stats = pd.DataFrame(
        {
            'loss': loss,
            'cross-validation': cv,
            'iteration': its
        }
    )

    convergence_stats.index.name = 'id'
    convergence_stats.index = convergence_stats.index.astype(np.str)

    c = convergence_stats['loss'].astype(np.float)
    convergence_stats['loss'] = c

    c = convergence_stats['cross-validation'].astype(np.float)
    convergence_stats['cross-validation'] = c

    c = convergence_stats['iteration'].astype(np.int)
    convergence_stats['iteration'] = c

    return ranks, biplot, qiime2.Metadata(convergence_stats)
