import pandas as pd

from rhapsody.q2 import ConditionalFormat
from rhapsody.q2.plugin_setup import plugin


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


# posterior types
@plugin.register_transformer
def _22(ff: EmbeddingFormat) -> pd.DataFrame:
    return qiime2.Metadata.load(str(ff)).to_dataframe()


@plugin.register_transformer
def _23(ff: EmbeddingFormat) -> qiime2.Metadata:
    return qiime2.Metadata.load(str(ff))


@plugin.register_transformer
def _24(data: pd.DataFrame) -> EmbeddingFormat:
    ff = EmbeddingFormat()
    qiime2.Metadata(data).save(str(ff))
    return ff
