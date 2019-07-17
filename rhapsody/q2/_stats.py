from qiime2.plugin import SemanticType, model
from q2_types.feature_data import FeatureData


Conditional = SemanticType('Conditional',
                           variant_of=FeatureData.field['type'])


class ConditionalFormat(model.TextFileFormat):
    def validate(*args):
        pass


ConditionalDirFmt = model.SingleFileDirectoryFormat(
    'ConditionalDirFmt', 'conditionals.tsv', ConditionalFormat)
