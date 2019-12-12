# mmvec changelog

## Version 1.0.3 (2019-12-12)
# Enhancements
 - Tensorflow is now pinned to any version below 2.0 in [#112](https://github.com/biocore/mmvec/pull/112)
 - Learning rate defaults have been fixed to `1e-5` in [#110](https://github.com/biocore/mmvec/pull/110)

# Bug fixes
 - Inputs are now expected to be metabolites x microbes in heatmaps [#100](https://github.com/biocore/mmvec/pull/100)

## Version 1.0.2 (2019-10-18)
# Bug fixes
 - Inputs are now expected to be metabolites x microbes in heatmaps [#100](https://github.com/biocore/mmvec/pull/100)

## Version 1.0.1 (2019-10-17)
# Enhancements
 - Ranks are transposed and viewable in qiime metadata tabulate [#99](https://github.com/biocore/mmvec/pull/99)

# Bug fixes
 - Ranks are now calculated consistently between q2 and standalone cli [#99](https://github.com/biocore/mmvec/pull/99)

## Version 1.0.0 (2019-09-30)
# Enhancements
 - Paired heatmaps are available [#89](https://github.com/biocore/mmvec/pull/89)
 - Heatmap tutorials are available [#90](https://github.com/biocore/mmvec/pull/90)

# Bug fixes
 - The ordering of the eigenvalues are now reversed [#92](https://github.com/biocore/mmvec/pull/92)
 - The qiime2 assets setup is corrected [#91](https://github.com/biocore/mmvec/pull/91)

## Version 0.6.0 (2019-09-05)

# Enhancements
 - Ranks from CLI can now be imported into qiime2 [#84](https://github.com/biocore/mmvec/pull/84)
 - Ranks can be visualized as heatmaps [#69](https://github.com/biocore/mmvec/pull/69)

# Bug fixes
 - ConditionalFormat has been fixed [#68](https://github.com/biocore/mmvec/pull/68)

## Version 0.4.0 (2019-07-22)

# Enhancements
 - Simpler standalone CLI interface - now all outputs have named rows and columns. [#61](https://github.com/biocore/mmvec/pull/61)

# Bug fixes
 - The ranks file is no longer empty. [#61](https://github.com/biocore/mmvec/pull/61)

## Version 0.3.0 (2019-06-20)

Initial beta release.

# Bug fixes
 - Biplots are now being properly centered in the qiime2 interface [#58](https://github.com/biocore/mmvec/pull/58)


## Version 0.2.0 (2019-04-22)

Initial alpha release. MMvec API, standalone command line interface and qiime2 interface should be stable.
