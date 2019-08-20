import pandas as pd

from mmvec.q2 import ConditionalFormat
from mmvec.q2.plugin_setup import plugin


@plugin.register_transformer
def _1(ff: ConditionalFormat) -> pd.DataFrame:
    df = pd.read_csv(str(ff), sep='\t', comment='#', skip_blank_lines=True,
                     header=True, dtype=object)
    return df


@plugin.register_transformer
def _2(df: pd.DataFrame) -> ConditionalFormat:
    ff = ConditionalFormat()
    df.to_csv(str(ff), sep='\t', header=True, index=True)
    return ff
