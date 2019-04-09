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
	--input-path data/otus.biom \
	--output-path otus.qza \
	--type FeatureTable[Frequency]

qiime tools import \
	--input-path data/ms.biom \
	--output-path ms.qza \
	--type FeatureTable[Frequency]
```

Then you can run mmvec
```
qiime rhapsody mmvec \
	--i-microbes otus.qza \
	--i-metabolites ms.qza \
	--output-dir results
```
It is worth your time to investigate the logs that are deposited using Tensorboard.
Tensorboard can be run via
```
tensorboard --logdir .
```
And the results, if working properly will look something like this
![tensorboard](https://github.com/biocore/rhapsody/raw/master/images/summary.png "Tensorboard")

You may need to tinker with the parameters to get readable tensorflow results, namely `--p-summary-interval`,
`--epochs` and `--batch-size`.  Both `--p-epochs` and `--p-batch-size` contribute to determining how long the algorithm will run, namely

**Number of iterations = `--p-epoch #` multiplied by the `--p-batch-size` parameter**

A description of these two graphs is outlined in the FAQs below.


Then you can run

```
qiime emperor biplot \
	--i-biplot results/conditional_biplot.qza \
	--m-sample-metadata-file data/metabolite-metadata.txt \
	--m-feature-metadata-file data/microbe-metadata.txt \
	--o-visualization emperor.qzv --verbose

```

More information behind the parameters can found under `qiime rhapsody --help`
