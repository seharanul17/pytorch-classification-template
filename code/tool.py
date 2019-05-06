import os

import torch

from config import config


def save_model(checkpoint, fold, epoch, metric, model, optimizer):
    state = {
        'fold': fold,
        'epoch': epoch,
        'metric': metric,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'config': config
    }
    if checkpoint:
        save_path = os.path.join(config['@root'], config['@save'], config['@checkpoint'])
        if not os.path.exists(save_path): os.makedirs(save_path)
        torch.save(state, os.path.join(save_path, config['version']+'.pth'))

    else:
        save_path = os.path.join(config['@root'], config['@save'], config['@best_model'], config['version'])
        if not os.path.exists(save_path): os.makedirs(save_path)
        torch.save(state, os.path.join(save_path, str(fold)+'.pth'))

