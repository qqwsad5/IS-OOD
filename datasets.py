import os
import torchvision as tvs
from PIL import Image

from torch.utils.data import DataLoader, ConcatDataset, Dataset

class BaseDataset(Dataset):
    def __init__(self, imglist_pth, data_dir, preprocessor, OOD, **kwargs):
        super(BaseDataset, self).__init__(**kwargs)

        with open(imglist_pth) as imgfile:
            self.imglist = [line.strip('\n') for line in imgfile.readlines()]
        self.data_dir = data_dir
        self.transform_image = preprocessor
        self.OOD = OOD

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        image_name = self.imglist[index]
        path = os.path.join(self.data_dir, image_name)
        sample = dict()

        image = Image.open(path).convert('RGB')
        sample['data'] = self.transform_image(image)
        sample['label'] = -1 if self.OOD else 0

        return sample

def get_id_ood_dataloader_level_split(data_root, **loader_kwargs):

    # ignore subsets that are too small
    DATASET_NUM_TH = 10

    # root of the datasets
    IN1K_ROOT = os.path.join(data_root, 'imagenet1k')
    IN21K_ROOT = os.path.join(data_root, 'imagenet21k')
    SynIS_ROOT = os.path.join(data_root, 'SynIS')

    # augmentation for imagenet1k
    imagenet_1k_preprocessor = tvs.transforms.Compose([
        tvs.transforms.Resize(256),
        tvs.transforms.CenterCrop(224),
        tvs.transforms.ToTensor(),
        tvs.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])

    dataloader_dict = {}

    # id dataloader
    dataloader_dict['id'] = {}
    for split in ['train', 'val', 'test']:
        dataset = BaseDataset(
            imglist_pth=os.path.join(data_root, './imglist/imagenet1k/{}_imagenet.txt'.format(split)),
            data_dir=IN1K_ROOT,
            preprocessor=imagenet_1k_preprocessor,
            OOD=False)
        dataloader = DataLoader(dataset, **loader_kwargs)
        dataloader_dict['id'][split] = dataloader

    # augmentation for resized imagenet21k
    imagenet_21k_preprocessor = tvs.transforms.Compose([
        tvs.transforms.Resize(224),
        tvs.transforms.CenterCrop(224),
        tvs.transforms.ToTensor(),
        tvs.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])

    # ood dataloader
    ood_datasets = ["train", "val"]
    dataloader_dict['ood'] = {}
    for cov_level in range(8):
        for sem_level in range(8):
            concat_dataset = []
            shift_level = 'cov_{}_sem_{}'.format(cov_level, sem_level)
            for split in ood_datasets:
                dataset = BaseDataset(
                    imglist_pth=os.path.join(data_root, './imglist/imagenet21k/{}/{}.txt'.format(split, shift_level)),
                    data_dir=IN21K_ROOT,
                    preprocessor=imagenet_21k_preprocessor,
                    OOD=True)
                concat_dataset.append(dataset)

            dataset = ConcatDataset(concat_dataset)
            if len(dataset) < DATASET_NUM_TH:
                dataloader = None
            else:
                dataloader = DataLoader(dataset, **loader_kwargs)
            dataloader_dict['ood'][shift_level] = dataloader

    # SynIS dataloader
    dataloader_dict['SynIS'] = {}
    for cov_level in range(8):
        for sem_level in range(8):
            shift_level = 'cov_{}_sem_{}'.format(cov_level, sem_level)
            dataset = BaseDataset(
                imglist_pth=os.path.join(data_root, './imglist/SynIS/{}.txt'.format(shift_level)),
                data_dir=SynIS_ROOT,
                preprocessor=imagenet_1k_preprocessor,
                OOD=True)
            if len(dataset) < DATASET_NUM_TH:
                dataloader = None
            else:
                dataloader = DataLoader(dataset, **loader_kwargs)
            dataloader_dict['SynIS'][shift_level] = dataloader

    return dataloader_dict