# rhapsody
Neural networks for estimating microbe-metabolite co-occurence probabilities.

# Installation
```
conda create -n mae python=3.5 tensorflow numpy scipy pandas scikit-bio tqdm pip
conda install -n mae biom-format -c conda-forge
source activate mae
pip install h5py git+https://github.com/mortonjt/rhapsody.git
```

If you are getting errors, it is likely because you have garbage channels under your .condarc.  Make sure to delete your .condarc -- you shouldn't need it.

# Getting started

To get started you can run a quick example as follows.  This will generate
microbe-metabolite conditional probabilities that are accurate up to rank.

```
rhapsody mmvec \
	--otu-file data/otu.biom \
	--metabolite-file data/ms.biom \
	--summary-dir summary \
	--results-file cv-results.csv \
	--ranks-file ranks.csv
```

While this is running, you can open up another session and run `tensorboard --logdir .` for diagnosis.

See the following url for a more complete tutorial with real datasets.

https://github.com/knightlab-analyses/multiomic-cooccurences

More information can found under `rhapsody --help`

# Qiime2 plugin

If you want to make this qiime2 compatible, install this in your
qiime2 conda environment and run the following

```
qiime dev refresh-cache
```

This should allow your q2 environment to recognize rhapsody.  To test run
the qiime2 plugin, run the following commands

```
qiime tools import \
	--input-path data/otu.biom \
	--output-path otu.qza \
	--type FeatureTable[Frequency]

qiime tools import \
	--input-path data/ms.biom \
	--output-path ms.qza \
	--type FeatureTable[Frequency]

qiime rhapsody mmvec \
	--i-microbes otu.qza \
	--i-metabolites ms.qza \
	--o-conditional-ranks ranks.qza
```

More information can found under `qiime rhapsody --help`
