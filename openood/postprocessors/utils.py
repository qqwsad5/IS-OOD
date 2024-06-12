from openood.utils import Config

from .ash_postprocessor import ASHPostprocessor
from .base_postprocessor import BasePostprocessor
from .dice_postprocessor import DICEPostprocessor
from .gradnorm_postprocessor import GradNormPostprocessor
from .knn_postprocessor import KNNPostprocessor
from .mds_postprocessor import MDSPostprocessor
from .odin_postprocessor import ODINPostprocessor
from .rankfeat_postprocessor import RankFeatPostprocessor


def get_postprocessor(config: Config):
    postprocessors = {
        'ash': ASHPostprocessor,
        'msp': BasePostprocessor,
        'odin': ODINPostprocessor,
        'mds': MDSPostprocessor,
        'gradnorm': GradNormPostprocessor,
        'knn': KNNPostprocessor,
        'dice': DICEPostprocessor,
        'rankfeat': RankFeatPostprocessor,
    }

    return postprocessors[config.postprocessor.name](config)
