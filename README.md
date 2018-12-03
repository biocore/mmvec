# minstrel
Multi-modal autoencoder

# Installation
```
conda create -n mae python=3.5 tensorflow numpy scipy pandas scikit-bio tqdm pip
conda install -n mae biom-format -c conda-forge
source activate mae
pip install h5py git+https://github.com/mortonjt/minstrel.git
```

# Getting started

To get started you can run a quick example as follows.  This will generate
microbe-metabolite conditional probabilities that are accurate up to rank.

```
minstrel autoencoder \
	--otu-file data/otu.biom \
	--metabolite-file data/ms.biom \
	--summary-dir summary \
	--results-file cv-results.csv \
	--ranks-file ranks.csv
```

We can use the results from the autoencoder to build a
network as follows
```
multimodal.py network \
	--ranks-file ranks.csv \
	--node-metadata cytoscape-nodes.txt \
	--edge-metadata cytoscape-edges.sif
```

This information can be directly feed into cytoscape.

More information can found under `minstrel --help`

# Qiime2 plugin

If you want to make this qiime2 compatible, install this in your
qiime2 conda environment and run the following

```
qiime dev refresh-cache
```

This should allow your q2 environment to recognize minstrel.  To test run
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

qiime minstrel autoencoder \
	--i-microbes otu.qza \
	--i-metabolites ms.qza \
	--o-conditional-ranks ranks.qza
```

More information can found under `qiime minstrel --help`
