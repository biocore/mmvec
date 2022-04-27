from .heatmap import _heatmap_choices, _cmaps
from .ALR import MMvecALR 
from .ILR import MMvecILR
from .train import mmvec_training_loop

__version__ = "1.0.6"

__all__ = ['_heatmap_choices', '_cmaps', 'MMvecALR', 'MMvecILR',
           'mmvec_training_loop']
