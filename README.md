# deep-mae
Deep multi-modal autoencoder

# Installation
```
conda create -n mae tensorflow keras numpy scipy pandas scikit-bio biom-format
source activate mae
pip install git+https://github.com/mortonjt/deep-mae.git
```

# Getting started

First split up your data into training and testing sets.
You can also use this to filter out features observed in few samples.
The number of examples that you would like to cross-validate against
can be specified by `num_test` as follows
```
multimodal.py split \
	--otu-table-file data/otu.biom \
	--metabolite-table-file data/ms.biom \
	--num_test 20 \
	--min_samples 10 \
	--output_dir split_data
```

Once the data is splitted, you can then run the model and perform
cross-validation.
```
multimodal.py autoencoder \
	--otu-train-file split_data/train_otu.biom \
	--metabolite-train-file split_data/train_ms.biom \
	--otu-test-file split_data/test_otu.biom \
	--metabolite-test-file split_data/test_ms.biom \
	--summary-dir summary \
	--results-file cv-results.csv \
	--ranks-file ranks.csv
```

Finally we can use the results from the autoencoder to build a
network as follows
```
multimodal.py network \
	--ranks-file ranks.csv \
	--node-metadata cytoscape-nodes.txt \
	--edge-metadata cytoscape-edges.sif
```

This information can be directly feed into cytoscape.
