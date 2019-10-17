from os.path import join
import pandas as pd
import qiime2
import biom
import pkg_resources
import q2templates
from mmvec.heatmap import ranks_heatmap, paired_heatmaps


TEMPLATES = pkg_resources.resource_filename('mmvec.q2', 'assets')


def heatmap(output_dir: str,
            ranks: pd.DataFrame,
            microbe_metadata: qiime2.CategoricalMetadataColumn = None,
            metabolite_metadata: qiime2.CategoricalMetadataColumn = None,
            method: str = 'average',
            metric: str = 'euclidean',
            color_palette: str = 'seismic',
            margin_palette: str = 'cubehelix',
            x_labels: bool = False,
            y_labels: bool = False,
            level: int = -1) -> None:
    if microbe_metadata is not None:
        microbe_metadata = microbe_metadata.to_series()
    if metabolite_metadata is not None:
        metabolite_metadata = metabolite_metadata.to_series()

    hotmap = ranks_heatmap(ranks.T, microbe_metadata, metabolite_metadata,
                           method, metric, color_palette, margin_palette,
                           x_labels, y_labels, level)

    hotmap.savefig(join(output_dir, 'heatmap.pdf'), bbox_inches='tight')
    hotmap.savefig(join(output_dir, 'heatmap.png'), bbox_inches='tight')

    index = join(TEMPLATES, 'index.html')
    q2templates.render(index, output_dir, context={
        'title': 'Rank Heatmap',
        'pdf_fp': 'heatmap.pdf',
        'png_fp': 'heatmap.png'})


def paired_heatmap(output_dir: str,
                   ranks: pd.DataFrame,
                   microbes_table: biom.Table,
                   metabolites_table: biom.Table,
                   features: str = None,
                   top_k_microbes: int = 2,
                   keep_top_samples: bool = True,
                   microbe_metadata: qiime2.CategoricalMetadataColumn = None,
                   normalize: str = 'log10',
                   color_palette: str = 'magma',
                   top_k_metabolites: int = 50,
                   level: int = -1) -> None:
    if microbe_metadata is not None:
        microbe_metadata = microbe_metadata.to_series()
    select_microbes, select_metabolites, hotmaps = paired_heatmaps(
        ranks.T, microbes_table, metabolites_table, microbe_metadata, features,
        top_k_microbes, top_k_metabolites, keep_top_samples, level, normalize,
        color_palette)

    hotmaps.savefig(join(output_dir, 'heatmap.pdf'), bbox_inches='tight')
    hotmaps.savefig(join(output_dir, 'heatmap.png'), bbox_inches='tight')
    select_microbes.to_csv(join(output_dir, 'select_microbes.tsv'), sep='\t')
    select_metabolites.to_csv(
        join(output_dir, 'select_metabolites.tsv'), sep='\t')

    index = join(TEMPLATES, 'index.html')
    q2templates.render(index, output_dir, context={
        'title': 'Paired Feature Abundance Heatmaps',
        'pdf_fp': 'heatmap.pdf',
        'png_fp': 'heatmap.png',
        'table1_fp': 'select_microbes.tsv',
        'download1_text': 'Download microbe abundances as TSV',
        'table2_fp': 'select_metabolites.tsv',
        'download2_text': 'Download top k metabolite abundances as TSV'})
