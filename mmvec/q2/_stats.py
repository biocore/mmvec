from qiime2.plugin import SemanticType, model
from q2_types.feature_data import FeatureData
from q2_types.sample_data import SampleData


Conditional = SemanticType('Conditional',
                           variant_of=FeatureData.field['type'])


class ConditionalFormat(model.TextFileFormat):
    def validate(*args):
        pass


ConditionalDirFmt = model.SingleFileDirectoryFormat(
    'ConditionalDirFmt', 'conditionals.tsv', ConditionalFormat)


# songbird stats summarizing loss and cv error
MMvecStats = SemanticType('MMvecStats',
                          variant_of=SampleData.field['type'])


class MMvecStatsFormat(model.TextFileFormat):
    def validate(*args):
        pass


MMvecStatsDirFmt = model.SingleFileDirectoryFormat(
    'MMvecStatsDirFmt', 'stats.tsv', MMvecStatsFormat)
