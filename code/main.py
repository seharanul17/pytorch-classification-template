import os
import numpy as np
import random

import torch.backends.cudnn as cudnn
import torch

from train import train
from config import config

os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('Error : Use GPU')
        raise

    
    device_ids = list(range(len(config['gpu'].split(','))))
    device = torch.device('cuda')

    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    
    train(device_ids, device)
    print('{} done'.format(config['version']))
