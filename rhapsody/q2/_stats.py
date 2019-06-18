from qiime2.plugin import SemanticType, model
from q2_types.feature_data import FeatureData



Conditional = SemanticType('Conditional',
                           variant_of=FeatureData.field['type'])


class ConditionalFormat(model.TextFileFormat):
    def validate(*args):
        pass


ConditionalDirFmt = model.SingleFileDirectoryFormat(
    'ConditionalDirFmt', 'conditionals.tsv', ConditionalFormat)


Embedding = SemanticType('Embedding',
                         variant_of=FeatureData.field['type'])


class EmbeddingFormat(model.TextFileFormat):
    def validate(self, *args):
        try:
            md = qiime2.Metadata.load(str(self))
        except qiime2.metadata.MetadataFileError as md_exc:
            raise ValidationError(md_exc) from md_exc

        if md.column_count <= 3:
            raise ValueError(
                    'Embedding format must contain more than 3 columns'
            )
        md = md.to_dataframe()
        # Must have at least 3 columns, featureid, embed_type and axis
        required_cols = {'featureid', 'embed_type', 'axis'}
        if required_cols & set(md.columns) != required_cols:
            raise ValueError(
                    ('Embedding format must contain columns for '
                     '`featureid`, `embed_type` and `axis`')
            )

        other_cols = list(set(md.columns) - required_cols)
        remaining_md = md[other_cols]
        types = remaining_md.dtypes
        for t in types:
            if np.issubdtype(np.number, t):
                raise ValueError(
                        ('Embedding types must contain '
                         'continuously valued quantities')
                )


EmbeddingDirFmt = model.SingleFileDirectoryFormat(
    'EmbeddingDirFmt', 'embedding.tsv', EmbeddingFormat)
