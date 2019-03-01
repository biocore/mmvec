import os
import biom
import qiime2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from songbird.q2 import _summarize


def summarize_single(output_dir: str,  feature_table: biom.Table,
                     mmvec_stats: qiime2.Metadata):
    n = feature_table.shape[1]
    _summarize(output_dir, n, regression_stats.to_dataframe())


def summarize_paired(output_dir: str, feature_table: biom.Table,
                     regression_stats: qiime2.Metadata,
                     mmvec_stats: qiime2.Metadata):
    n = feature_table.shape[1]
    _summarize(output_dir, n,
               regression_stats.to_dataframe(),
               baseline_stats.to_dataframe())
