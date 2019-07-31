import seaborn as sns
import matplotlib.pyplot as plt


_heatmap_choices = {
    'metric': {'braycurtis', 'canberra', 'chebyshev', 'cityblock',
               'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
               'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
               'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
               'sokalsneath', 'sqeuclidean', 'yule'},
    'method': {'single', 'complete', 'average', 'weighted', 'centroid',
               'median', 'ward'}}

_cmaps = {
    'heatmap': [
        'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
        'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
        'viridis', 'plasma', 'inferno', 'magma', 'cividis'],
    'margins': [
        'cubehelix', 'Pastel1', 'Pastel2', 'Paired',
        'Accent', 'Dark2', 'Set1', 'Set2', 'Set3',
        'tab10', 'tab20', 'tab20b', 'tab20c',
        'viridis', 'plasma', 'inferno', 'magma', 'cividis']}


def ranks_heatmap(ranks, microbe_metadata=None, metabolite_metadata=None,
                  method='average', metric='euclidean',
                  color_palette='seismic', margin_palette='cubehelix',
                  x_labels=False, y_labels=False, level=3, threshold=3):
    '''
    Generate clustermap of microbe X metabolite conditional probabilities.
Parameters
----------
    ranks: pd.DataFrame of conditional probabilities.
        Microbes (rows) X metabolites (columns).
    microbe_metadata: pd.Series of microbe metadata for annotating plots
    metabolite_metadata: pd.Series of metabolite metadata for annotating plots
    method: str
        Hierarchical clustering method used in clustermap.
    metric: str
        Hierarchical clustering distance metric used in clustermap.
    color_palette: str
        Color palette for clustermap.
    margin_palette: str
        Name of color palette to use for annotating metadata
        along margin(s) of clustermap.
    x_labels: bool
        Plot x-axis (metabolite) labels?
    y_labels: bool
        Plot y-axis (microbe) labels?
    level: int
        taxonomic level for annotating clustermap. Set to -1 if not parsing
        semicolon-delimited taxonomies or wish to print entire annotation.
    threshold: int
        Minimum absolute value of conditional probabilities to plot.
        metabolites/taxa that do not have at least one CP above the
        threshold will be removed.

    Returns clustermap
    '''
    # filter microbes
    abs_vals = ranks.abs()
    ranks = ranks[abs_vals.max(axis=1) > threshold]
    # filter metabolites
    ranks = ranks[ranks.columns[(abs_vals > threshold).any()]]

    # subset microbe metadata based on rows/columns
    if microbe_metadata is not None:
        microbe_metadata = microbe_metadata.reindex(ranks.index)
        # parse semicolon-delimited taxonomy
        if level > -1:
            microbe_metadata = _parse_taxonomy_strings(microbe_metadata, level)
        # map metadata categories to row colors
        row_colors, row_class_colors = _parse_heatmap_metadata_annotations(
            microbe_metadata, margin_palette)
    else:
        row_colors = None

    # subset metabolite metadata based on rows/columns
    if metabolite_metadata is not None:
        metabolite_metadata = metabolite_metadata.reindex(ranks.columns)
        # map metadata categories to column colors
        col_colors, col_class_colors = _parse_heatmap_metadata_annotations(
            metabolite_metadata, margin_palette)
    else:
        col_colors = None

    # Generate heatmap
    hotmap = sns.clustermap(ranks, cmap=color_palette, center=0,
                            col_colors=col_colors, row_colors=row_colors,
                            figsize=(12, 12), method=method, metric=metric,
                            cbar_kws={'label': 'Conditional\nProbability'})

    # add legends
    if col_colors is not None:
        for label in col_class_colors.keys():
            hotmap.ax_col_dendrogram.bar(
                0, 0, color=col_class_colors[label], label=label, linewidth=0)
        hotmap.ax_col_dendrogram.legend(
            title=metabolite_metadata.name, ncol=5, bbox_to_anchor=(0.9, 0.95),
            bbox_transform=plt.gcf().transFigure)
    if row_colors is not None:
        for label in row_class_colors.keys():
            hotmap.ax_row_dendrogram.bar(
                0, 0, color=row_class_colors[label], label=label, linewidth=0)
        hotmap.ax_row_dendrogram.legend(
            title=microbe_metadata.name, ncol=1, bbox_to_anchor=(0.1, 0.5),
            bbox_transform=plt.gcf().transFigure)

    # toggle axis labels
    if not x_labels:
        hotmap.ax_heatmap.set_xticklabels('')
    if not y_labels:
        hotmap.ax_heatmap.set_yticklabels('')

    return hotmap


def _parse_heatmap_metadata_annotations(metadata_column, margin_palette):
    '''
    Transform feature or sample metadata into color vector for annotating
    margin of clustermap.
Parameters
----------
    metadata_column: pd.Series of metadata for annotating plots
    margin_palette: str
        Name of color palette to use for annotating metadata
        along margin(s) of clustermap.
Returns
-------
    Returns vector of colors for annotating clustermap and dict mapping colors
    to classes.
    '''
    # Create a categorical palette to identify md col
    metadata_column = metadata_column.astype(str)
    col_names = sorted(metadata_column.unique())

    # Select Color palette
    if margin_palette == 'colorhelix':
        col_palette = sns.cubehelix_palette(
            len(col_names), start=2, rot=3, dark=0.3, light=0.8, reverse=True)
    else:
        col_palette = sns.color_palette(margin_palette, len(col_names))
    class_colors = dict(zip(col_names, col_palette))

    # Convert the palette to vectors that will be drawn on the matrix margin
    col_colors = metadata_column.map(class_colors)

    return col_colors, class_colors


def _parse_taxonomy_strings(taxonomy_series, level):
    '''
    taxonomy_series: pd.Series of semicolon-delimited taxonomy strings
    level: int
        taxonomic level for annotating clustermap.
 Returns
 -------
    Returns a pd.Series of taxonomy names at specified level,
        or terminal annotation
    '''
    return taxonomy_series.apply(lambda x: x.split(';')[:level][-1].strip())
