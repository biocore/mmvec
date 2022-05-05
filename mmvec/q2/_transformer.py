import qiime2
import pandas as pd

from mmvec.q2 import ConditionalFormat, MMvecStatsFormat
from mmvec.q2.plugin_setup import plugin


@plugin.register_transformer
def _1(ff: ConditionalFormat) -> pd.DataFrame:
    df = pd.read_csv(str(ff), sep='\t', comment='#', skip_blank_lines=True,
                     header=0, index_col=0)
    return df


@plugin.register_transformer
def _2(df: pd.DataFrame) -> ConditionalFormat:
    ff = ConditionalFormat()
    df.to_csv(str(ff), sep='\t', header=True, index=True)
    return ff


@plugin.register_transformer
def _3(ff: ConditionalFormat) -> qiime2.Metadata:
    return qiime2.Metadata.load(str(ff))


@plugin.register_transformer
def _4(obj: qiime2.Metadata) -> MMvecStatsFormat:
    ff = MMvecStatsFormat()
    obj.save(str(ff))
    return ff


@plugin.register_transformer
def _5(ff: MMvecStatsFormat) -> qiime2.Metadata:
    return qiime2.Metadata.load(str(ff))
