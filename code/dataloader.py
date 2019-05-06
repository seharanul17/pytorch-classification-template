import os
import json
from PIL import Image
import numpy as np
from torch.utils import data
import torch
from logger import logger

from config import config


class Dataset(data.Dataset):
    def __init__(self, split, fold_idx):
        self.transform = config['transform'][split]
        self.table_path = os.path.join(config['@root'], config['@data'], config['@table'])
        self.img_path = os.path.join(config['@root'], config['@data'], config['@image'])

        with open(self.table_path, 'r') as f:
            table = json.load(f)

        self.data = sum([table[str(i)] for i in fold_idx], [])
        self.labels = [self.data[i][1] for i in range(len(self.data))]
        logger.info('{} Dict {} Data [{}]'.format(split.upper(), fold_idx, len(self.data)))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        _, label, path = self.data[index]
        img = Image.open(os.path.join(self.img_path, path)).convert('RGB')
        assert len(img.getbands()) == 3
        img = self.transform(img)
        label = torch.tensor(label).long()
        #print(img.size())
        # invasion = torch.tensor(invasion).long()
        
        return img, label


def Dataloader(split, fold):
    if split == 'train':
        fold_idx = [i for i in range(1, config['#fold'] + 1) if i != fold]
        dataset = Dataset(split=split, fold_idx=fold_idx)

        if config['sampler'] == 'ROS':
            labels = dataset.labels
            class_weight = {label: 1 / float(labels.count(label)) for label in range(config['#class'])}
            weight = [class_weight[label] for label in labels]
            sampler = data.WeightedRandomSampler(weight, len(weight) * config['#sample_ROS'], replacement=True)
            shuffle = False
        elif config['sampler'] == 'None':
            sampler = None
            shuffle = True

    else:
        fold_idx = [fold]
        dataset = Dataset(split=split, fold_idx=fold_idx)
        sampler = None
        shuffle = False

    data_loader = data.DataLoader(dataset, shuffle=shuffle, sampler=sampler, worker_init_fn=_init_fn,
                                  batch_size=config['#batch'], num_workers=config['#worker'])

    return data_loader


def _init_fn(worker_id):
    np.random.seed(config['seed'] + worker_id)
