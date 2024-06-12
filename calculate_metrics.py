import os
import argparse
import torch

from openood.networks import ResNet50, ASHNet
from datasets import get_id_ood_dataloader_level_split
from evaluate import eval_ood
from postprocessor import get_postprocessor


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt-path', default='./pretrained_weight/resnet50_imagenet1k_v1.pth')
parser.add_argument('--save-csv', default=True)
parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('--postprocessor', default='msp', choices=['msp','knn','dice','rankfeat','odin','gradnorm','ash', 'mds'])
args = parser.parse_args()

# load the model
net = ResNet50()
ckpt = torch.load(args.ckpt_path, map_location='cpu')
net.load_state_dict(ckpt)

postprocessor_name = args.postprocessor

# wrap base model to work with certain postprocessors
if postprocessor_name == 'ash':
    net = ASHNet(net)

net.cuda()
net.eval()

data_root = "./data"
config_root = "./openood/configs"
saving_root = "./results"

# get postprocessor
postprocessor = get_postprocessor(config_root, postprocessor_name, 'imagenet')

# load data
loader_kwargs = {
    'batch_size': 256,
    'shuffle': False,
    'num_workers': 4
}
dataloader_dict = get_id_ood_dataloader_level_split(data_root, **loader_kwargs)

# postprocessor setup
postprocessor.setup(net, dataloader_dict['id'], dataloader_dict['ood'])

# obtain metrics
metrics = eval_ood(net, postprocessor, dataloader_dict)

# saving and recording
if args.save_csv:
    if not os.path.exists(saving_root):
        os.makedirs(saving_root)

    save_name = postprocessor_name

    if not os.path.isfile(
            os.path.join(saving_root, f'{save_name}.csv')):
        metrics.to_csv(os.path.join(saving_root, f'{save_name}.csv'),
                       float_format='{:.2f}'.format)

