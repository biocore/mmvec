qiime tools import --input-path otus_nt.biom --output-path otus_nt.qza --type FeatureTable[Frequency]
qiime tools import --input-path lcms_nt.biom --output-path lcms_nt.qza --type FeatureTable[Frequency]

qiime mmvec paired-omics \
      --i-microbes otus_nt.qza \
      --i-metabolites lcms_nt.qza  \
      --p-epochs 100 \
      --p-learning-rate 1e-3 \
      --o-conditionals ranks.qza \
      --o-conditional-biplot biplot.qza \
      --verbose

qiime emperor biplot \
      --i-biplot biplot.qza \
      --m-sample-metadata-file metabolite-metadata.txt \
      --m-feature-metadata-file microbe-metadata.txt \
      --p-number-of-features 50 \
      --o-visualization emperor.qzv \
      --p-ignore-missing-samples
