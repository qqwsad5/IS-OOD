import os
import urllib.request

from openood.postprocessors import (
    ASHPostprocessor, BasePostprocessor, DICEPostprocessor, KNNPostprocessor, GradNormPostprocessor, RankFeatPostprocessor, ODINPostprocessor, MDSPostprocessor)
from openood.utils.config import Config, merge_configs

postprocessors = {
    'ash': ASHPostprocessor,
    'msp': BasePostprocessor,
    'odin': ODINPostprocessor,
    'gradnorm': GradNormPostprocessor,
    'knn': KNNPostprocessor,
    'dice': DICEPostprocessor,
    'rankfeat': RankFeatPostprocessor,
    'mds': MDSPostprocessor,
}

link_prefix = 'https://raw.githubusercontent.com/Jingkang50/OpenOOD/main/configs/postprocessors/'


def get_postprocessor(config_root: str, postprocessor_name: str, id_data_name: str):
    postprocessor_config_path = os.path.join(config_root, 'postprocessors',
                                             f'{postprocessor_name}.yml')
    if not os.path.exists(postprocessor_config_path):
        os.makedirs(os.path.dirname(postprocessor_config_path), exist_ok=True)
        urllib.request.urlretrieve(link_prefix + f'{postprocessor_name}.yml',
                                   postprocessor_config_path)

    config = Config(postprocessor_config_path)
    config = merge_configs(config, Config(**{'dataset': {'name': id_data_name}}))
    postprocessor = postprocessors[postprocessor_name](config)
    # postprocessor.APS_mode = config.postprocessor.APS_mode
    postprocessor.APS_mode = False
    postprocessor.hyperparam_search_done = False
    return postprocessor
