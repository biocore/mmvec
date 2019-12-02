[![Build Status](https://travis-ci.org/biocore/mmvec.svg?branch=master)](https://travis-ci.org/biocore/mmvec)

# mmvec
Neural networks for estimating microbe-metabolite interactions through their co-occurence probabilities.

![](https://github.com/biocore/mmvec/raw/master/img/mmvec.png "mmvec")

# Installation

MMvec can be installed via pypi as follows

```
pip install mmvec
```

If you are planning on using GPUs, be sure to `pip install tensorflow-gpu`.

MMvec can also be installed via conda as follows

```
conda install mmvec -c conda-forge
```

Note that this option may not work in cluster environments, it maybe workwhile to pip install within a virtual environment.  It is possible to pip install mmvec within a conda environment, including qiime2 conda environments.  However, pip and conda are known to have compatibility issues, so proceed with caution.

# Getting started

To get started you can run a quick example as follows.  This will learn microbe-metabolite vectors (mmvec)
which can be used to estimate microbe-metabolite conditional probabilities that are accurate up to rank.

```
mmvec paired-omics \
	--microbe-file examples/cf/otus_nt.biom \
	--metabolite-file examples/cf/lcms_nt.biom \
	--summary-dir summary
```

While this is running, you can open up another session and run `tensorboard --logdir .` for diagnosis, see FAQs below for more details.

If you investigate the summary folder, you will notice that there are a number of files deposited.

See the following url for a more complete tutorial with real datasets.

https://github.com/knightlab-analyses/multiomic-cooccurences

More information can found under `mmvec --help`

# Qiime2 plugin

If you want to run this in a qiime environment, install this in your
qiime2 conda environment (see qiime2 installation instructions [here](https://qiime2.org/)) and run the following

```
pip install git+https://github.com/biocore/mmvec.git
qiime dev refresh-cache
```

This should allow your q2 environment to recognize mmvec. Before we test
the qiime2 plugin, run the following commands to import an example dataset

```
qiime tools import \
	--input-path data/otus_nt.biom \
	--output-path otus_nt.qza \
	--type FeatureTable[Frequency]

qiime tools import \
	--input-path data/lcms_nt.biom \
	--output-path lcms_nt.qza \
	--type FeatureTable[Frequency]
```

Then you can run mmvec
```
qiime mmvec paired-omics \
	--i-microbes otus_nt.qza \
	--i-metabolites lcms_nt.qza \
	--o-conditionals ranks.qza \
	--o-conditional-biplot biplot.qza
```

In the results, there are two files, namely `results/conditional_biplot.qza` and `results/conditionals.qza`. The conditional biplot is a biplot representation the
conditional probability matrix so that you can visualize these microbe-metabolite interactions in an exploratory manner.  This can be directly visualized in
Emperor as shown below.  We also have the estimated conditional probability matrix given in `results/conditionals.qza`,
which an be unzip to yield a tab-delimited table via `unzip results/conditionals`. Each row can be ranked,
so the top most occurring metabolites for a given microbe can be obtained by identifying the highest co-occurrence probabilities for each microbe.

These log conditional probabilities can also be viewed directly with `qiime metadata tabulate`.  This can be
created as follows

```
qiime metadata tabulate \
	--m-input-file results/conditionals.qza \
	--o-visualization conditionals-viz.qzv
```


Then you can run the following to generate a emperor biplot.

```
qiime emperor biplot \
	--i-biplot conditional_biplot.qza \
	--m-sample-metadata-file data/metabolite-metadata.txt \
	--m-feature-metadata-file data/microbe-metadata.txt \
	--o-visualization emperor.qzv

```

The resulting biplot should look like something as follows

![biplot](https://github.com/biocore/mmvec/raw/master/img/biplot.png "Biplot")

Here, the metabolite represent points and the arrows represent microbes.  The points close together are indicative of metabolites that
frequently co-occur with each other.  Furthermore, arrows that have a small angle between them are indicative of microbes that co-occur with each other.
Arrows that point in the same direction as the metabolites are indicative of microbe-metabolite co-occurrences.  In the biplot above, the red arrows
correspond to Pseudomonas aeruginosa, and the red points correspond to Rhamnolipids that are likely produced by Pseudomonas aeruginosa.

Another way to examine these associations is to build heatmaps of the log
conditional probabilities between observations, using the `heatmap` action:

```
qiime mmvec heatmap \
  --i-ranks ranks.qza \
  --m-microbe-metadata-file taxonomy.tsv \
  --m-microbe-metadata-column Taxon \
  --m-metabolite-metadata-file metabolite-metadata.txt \
  --m-metabolite-metadata-column Compound_Source \
  --p-level 5 \
  --o-visualization ranks-heatmap.qzv
```

This action generates a clustered heatmap displaying the log conditional
probabilities between microbes and metabolites. Larger positive log conditional
probabilities indicate a stronger likelihood of co-occurrence. Low and negative
values indicate no relationship, not necessarily a negative correlation. Rows
(microbial features) can be annotated according to feature metadata, as shown
in this example; we provide a taxonomic classification file and the semicolon-
delimited taxonomic rank (`level`) that should be displayed in the color-coded
margin annotation. Set `level` to `-1` to display the full annotation
(including of non-delimited feature metadata). Separate parameters are
available to annotate the x-axis (metabolites) in a similar fashion. Row and
column clustering can be adjusted using the `method` and `metric` parameters.
This action will generate a heatmap that looks similar to this:

![heatmap](https://github.com/biocore/mmvec/raw/master/img/heatmap.png "Heatmap")

Biplots and heatmaps give a great overview of co-occurrence associations, but
do not provide information about the abundances of these co-occurring features
in each sample. This can be done with the `paired-heatmap` action:

```
qiime mmvec paired-heatmap \
  --i-ranks ranks.qza \
  --i-microbes-table otus_nt.qza \
  --i-metabolites-table lcms_nt.qza \
  --m-microbe-metadata-file taxonomy.tsv \
  --m-microbe-metadata-column Taxon \
  --p-features TACGAAGGGTGCAAGCGTTAATCGGAATTACTGGGCGTAAAGCGCGCGTAGGTGGTTCAGCAAGTTGGATGTGAAATCCCCGGGCTCAACCTGGGAACTGCATCCAAAACTACTGAGCTAGAGTACGGTAGAGGGTGGTGGAATTTCCTG \
  --p-features TACGTAGGTCCCGAGCGTTGTCCGGATTTATTGGGCGTAAAGCGAGCGCAGGCGGTTAGATAAGTCTGAAGTTAAAGGCTGTGGCTTAACCATAGTAGGCTTTGGAAACTGTTTAACTTGAGTGCAAGAGGGGAGAGTGGAATTCCATGT \
  --p-top-k-microbes 0 \
  --p-normalize rel_row \
  --p-top-k-metabolites 100 \
  --p-level 6 \
  --o-visualization paired-heatmap-top2.qzv
```

This action generates paired heatmaps that are aligned on the y-axis (sample
IDs): the left panel displays the abundances of each selected microbial feature
in each sample, and the right panel displays the abundances of the top k
metabolite features associated with each of these microbes in each sample.
Microbes can be selected automatically using the `top-k-microbes` parameter
(which selects the microbes with the top k highest relative abundances) or they
can be selected by name using the `features` parameter (if using the QIIME 2
plugin command-line interface as shown in this example, multiple features are
selected by passing this parameter multiple times, e.g., `--p-features feature1
--p-features feature2`; for python interfaces, pass a list of features:
`features=[feature1, feature2]`). As with the `heatmap` action, microbial
features can be annotated by passing in `microbe-metadata` and specifying a
taxonomic `level` to display. The output looks something like this:

![paired-heatmap](https://github.com/biocore/mmvec/raw/master/img/paired-heatmap.png "Paired Heatmap")


More information behind the actions and parameters can found under `qiime mmvec --help`

# FAQs

**Q**: Looks like there are two different commands, a standalone script and a qiime2 interface.  Which one should I use?!?

**A**:  It'll depend on how deep in the weeds you'll want to get.  For most intents and purposes, the qiime2 interface will more practical for most analyses.  There are 3 major reasons why the standalone scripts are more preferable to the qiime2 interface, namely

1. Customized acceleration : If you want to bring down your runtime from a few days to a few hours, you may need to compile Tensorflow to handle hardware specific instructions (i.e. GPUs / SIMD instructions).  It probably is possible to enable GPU compatiability within a conda environment with some effort, but since conda packages binaries, SIMD instructions will not work out of the box.

2. Checkpoints : If you are not sure how long your analysis should run, the standalone script can allow you record checkpoints, which can allow you to recover your model parameters.  This enables you to investigate your model while the model is training.

3. More model parameters : The standalone script will return the bias parameters learned for each dataset (i.e. microbe and metabolite abundances).  These are stored under the summary directory (specified by `--summary`) under the names `embeddings.csv`. This file will hold the coordinates for the microbes and metabolites, along with biases.  There are 4 columns in this file, namely `feature_id`, `axis`, `embed_type` and `values`.  `feature_id` is the name of the feature, whether it be a microbe name or a metabolite feature id.  `axis` corresponds to the name of the axis, which either corresponds to a PC axis or bias.  `embed_type` denotes if the coordinate corresponds to a microbe or metabolite.  `values` is the coordinate value for the given `axis`, `embed_type` and `feature_id`.  This can be useful for accessing the raw parameters and building custom biplots / ranks visualizations - this also has the advantage of requiring much less memory to manipulate.

It is also important to note that you don't have to explicitly chose - it is very doable to run the standalone version first, then import those output files into qiime2.  Importing can be done as follows

```
qiime tools import --input-path <your ranks file> --output-path conditionals.qza --type FeatureData[Conditional]

qiime tools import --input-path <your ordination file> --output-path ordination.qza --type 'PCoAResults % ("biplot")'
```

**Q** : You mentioned that you can use GPUs.  How can you do that??

**A** : This can be done by running `pip install tensorflow-gpu` in your environment.  See details [here](https://www.tensorflow.org/install/gpu).

At the moment, these capabilities are only available for the standalone CLI due to complications of installation.  See the `--arm-the-gpu` option in the standalone interface.

**Q** : Neural networks scare me - don't they overfit the crap out of your data?

**A** : Here, we are using shallow neural networks (so only two layers).  This falls under the same regime as PCA and SVD.  But just as you can overfit PCA/SVD, you can also overfit mmvec.  Which is why we have Tensorboard enabled for diagnostics. You can visualize the `cv_rmse` to gauge if there is overfitting -- if your run is strictly decreasing, then that is a sign that you are probably not overfitting.  But this is not necessarily indicative that you have reach the optimal -- you want to check to see if `logloss` has reached a plateau as shown above.

**Q** : I'm confused, what is Tensorboard?

**A** : Tensorboard is a diagnostic tool that runs in a web browser - note that this is only explicitly supported in the standalone version of mmvec. To open tensorboard, make sure you’re in the mmvec environment and cd into the folder you are running the script above from. Then run:

```
tensorboard --logdir .
```

Returning line will look something like:

```
TensorBoard 1.9.0 at http://Lisas-MacBook-Pro-2.local:6006 (Press CTRL+C to quit)
```
Open the website (highlighted in red) in a browser. (Hint; if that doesn’t work try putting only the port number (here it is 6006), adding localhost, localhost:6006). Leave this tab alone. Now any mmvec output directories that you add to the folder that tensorflow is running in will be added to the webpage.


If working properly, it will look something like this
![tensorboard](https://github.com/biocore/mmvec/raw/master/img/tensorboard.png "Tensorboard")

FIRST graph in Tensorflow; 'Prediction accuracy'. Labelled `cv_rmse`

This is a graph of the prediction accuracy of the model; the model will try to guess the metabolite intensitiy values for the testing samples that were set aside in the script above, using only the microbe counts in the testing samples. Then it looks at the real values and sees how close it was.

The second graph is the `likelihood` - if your `likelihood` values are plateaued, that is a sign that you have converged and reached at a local minima.

The x-axis is the number of iterations (meaning times the model is training across the entire dataset). Every time you iterate across the training samples, you also run the test samples and the averaged results are being plotted on the y-axis.


The y-axis is the average number of counts off for each feature. The model is predicting the sequence counts for each feature in the samples that were set aside for testing. So in the graph above it means that, on average, the model is off by ~0.75 intensity units, which is low. However, this is ABSOLUTE error not relative error (unfortunately we don't know how to compute relative errors because of the sparsity in these datasets).

You can also compare multiple runs with different parameters to see which run performed the best. Useful parameters to note are `--epochs` and `--batch-size`.  If you are committed to fine-tuning parameters, be sure to look at the `training-column` example make the testing samples consistent across runs.


**Q** : What's up with the `--training-column` argument?

**A** : That is used for cross-validation if you have a specific reproducibility question that you are interested in answering. It can also make it easier to compare cross validation results across runs. If this is specified, only samples labeled "Train" under this column will be used for building the model and samples labeled "Test" will be used for cross validation. In other words the model will attempt to predict the microbe abundances for the "Test" samples. The resulting prediction accuracy is used to evaluate the generalizability of the model in order to determine if the model is overfitting or not. If this argument is not specified, then 10 random samples will be chosen for the test dataset. If you want to specify more random samples to allocate for cross-validation, the `num-random-test-examples` argument can be specified.


**Q** : What sort of parameters should I focus on when picking a good model?

**A** : There are 3 different parameters to focus on, `input-prior`, `output-prior` and `latent-dim`

The `--input-prior`  and `--output-prior` options specifies the width of the prior distribution of the coefficients, where the `--input-prior` is typically specific to microbes and the `--output-prior` is specific to metabolites.
For a prior of 1, this means 99% of entries in the embeddings (typically given in the `U.txt` and `V.txt` files are within -3 and +3 (log fold change). The higher differential-prior is, the more parameters can have bigger changes, so you want to keep this relatively small. If you see overfitting (accuracy and fit increasing over iterations in tensorboard) you may consider reducing the `--input-prior` and `--output-prior` in order to reduce the parameter space.

Another parameter worth thinking about is `--latent-dim`, which controls the number of dimensions used to approximate the conditional probability matrix.  This also specifies the dimensions of the microbe/metabolite embeddings `U.txt` and `V.txt`.  The more dimensions this has, the more accurate the embeddings can be -- but the higher the chance of overfitting there is.  The rule of thumb to follow is in order to fit these models, you need at least 10 times as many samples as there are latent dimensions (this is following a similar rule of thumb for fitting straight lines).  So if you have 100 samples, you should definitely not have a latent dimension of more than 10.  Furthermore, you can still overfit certain microbes and metabolites.  For example, you are fitting a model with those 100 samples and just 1 latent dimension, you can still easily overfit microbes and metabolites that appear in less than 10 samples -- so even fitting models with just 1 latent dimension will require some microbes and metabolites that appear in less than 10 samples to be filtered out.


**Q** : What does a good model fit look like??

**A** : Again the numbers vary greatly by dataset. But you want to see the both the `logloss` and `cv_rmse` curves decaying, and plateau as close to zero as possible.

**Q** : How long should I expect this program to run?

**A** : Both `epochs` and `batch-size` contribute to determining how long the algorithm will run, namely

**Number of iterations = `epoch #` multiplied by the ( Total # of microbial reads / `batch-size` parameter)**

This also depends on if your program will converge. The `learning-rate` specifies the resolution (smaller step size = smaller resolution, but may take longer to converge). You will need to consult with Tensorboard to make sure that your model fit is sane. See this paper for more details on gradient descent: https://arxiv.org/abs/1609.04747

The default learning rate is **1e-1**, however note that the learning rates used in the paper were ***1e-5***.
If you want accurate results, make sure to have a small learning rate.

If you are running this on a CPU, 16 cores, a run that reaches convergence should take about 1 day.
If you have a GPU - you maybe able to get this down to a few hours.  However, some finetuning of the `batch-size` parameter maybe required -- instead of having a small `batch-size` < 100, you'll want to bump up the `batch-size` to between 1000 and 10000 to fully leverage the speedups available on the GPU.

Credits to Lisa Marotz ([@lisa55asil](https://github.com/lisa55asil)),  Yoshiki Vazquez-Baeza ([@ElDeveloper](https://github.com/ElDeveloper)), Julia Gauglitz ([@jgauglitz](https://github.com/jgauglitz)) and Nickolas Bokulich ([@nbokulich](https://github.com/nbokulich)) for their README contributions.
