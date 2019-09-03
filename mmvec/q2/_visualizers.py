from os.path import join
import pandas as pd
import qiime2
import pkg_resources
import q2templates
from mmvec.heatmap import ranks_heatmap


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

    hotmap = ranks_heatmap(ranks, microbe_metadata, metabolite_metadata,
                           method, metric, color_palette, margin_palette,
                           x_labels, y_labels, level)

    hotmap.savefig(join(output_dir, 'heatmap.pdf'), bbox_inches='tight')
    hotmap.savefig(join(output_dir, 'heatmap.png'), bbox_inches='tight')

    index = join(TEMPLATES, 'index.html')
    q2templates.render(index, output_dir, context={
        'title': 'Rank Heatmap',
        'pdf_fp': 'heatmap.pdf',
        'png_fp': 'heatmap.png'})
