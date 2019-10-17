mmvec paired-omics\
	 --microbe-file microbes.biom \
	 --metabolite-file metabolites.biom \
	 --num-testing-examples 1 \
	 --min-feature-count 0 \
	 --latent-dim 1 \
	 --learning-rate 1e-3 \
	 --epochs 3000

qiime tools import --input-path microbes.biom --output-path microbes.biom.qza --type FeatureTable[Frequency]
qiime tools import --input-path metabolites.biom --output-path metabolites.biom.qza --type FeatureTable[Frequency]

qiime mmvec paired-omics \
      --i-microbes microbes.biom.qza \
      --i-metabolites metabolites.biom.qza  \
      --p-epochs 100 \
      --p-learning-rate 1e-3 \
      --o-conditionals ranks.qza \
      --o-conditional-biplot biplot.qza \
      --verbose
