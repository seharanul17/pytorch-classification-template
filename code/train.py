import os

import torch
import torch.nn as nn

from config import config
from run import run
from dataloader import Dataloader
from tool import save_model
from logger import logger

def train(device_ids, device) :
    
    #if training stopped suddenly, resume 
    if config['resume']:
        state = torch.load(os.path.join(config['@root'], config['@save'], config['@checkpoint'], config['version']+'.pth'))

        if sum([str(config[key])==str(state['config'][key]) for key in config.keys() if key != 'resume']) != (len(config)-1):
            print('Error : resume with different config')
            print([(config[key], state['config'][key]) 
                   for key in config.keys() if str(config[key]) != str(state['config'][key]) and key !='resume'])
            raise

        folds = range(state['fold'], config['#fold']+1)
        epochs = range(state['epoch']+1, config['#epoch']+1)
        logger.debug('Resume {} : starts from Fold[{}/{}] Epoch[{}/{}]'.format(
            config['version'], state['fold'], config['#fold'], state['epoch']+1, config['#epoch']))
    
    #if not resume then fresh start 
    else :
        folds = range(1, config['#fold']+1)
        epochs = range(1, config['#epoch']+1)
    
    #Cross-validation
    fold_best_metric = []
    for fold in folds:
        
        #bring Dataset
        train_loader = Dataloader(split='train', fold=fold)
        val_loader = Dataloader(split='val', fold=fold)
        
        #define Model
        model = nn.DataParallel(config['model'](config['#class']), device_ids=device_ids)
        model.to(device)
        if config['resume'] and fold == folds[0] : model.load_state_dict(state['model_state'])

        optimizer = config['optimizer'](model.parameters(), lr=config['lr'])
        if config['resume'] and fold == folds[0] : optimizer.load_state_dict(state['optimizer_state'])

        criterion = config['criterion']()
        
        #check performance with model of random initialization
        metric  = run(fold=fold, epoch=-1, grad=False, model=model, optimizer=optimizer,
                                      criterion=criterion, loader=val_loader, device=device)
        logger.info('Random check {} | Fold[{}/{}] \n       Metric {}'.format(
            ' val ', fold, config['#fold'], metric))

        patience = 0
        best_metric = {key:-1 for key in config['metric']}
        for epoch in epochs :
            #train
            model, optimizer, metric = run(fold=fold, epoch=epoch, grad=True, model=model, optimizer=optimizer,
                                      criterion=criterion, loader=train_loader, device=device)
            logger.info('{} | Fold[{}/{}] Epoch[{}/{}] \n       Metric {}'.format(
                'train', fold, config['#fold'], epoch, config['#epoch'], metric))
            
            #validate
            metric  = run(fold=fold, epoch=epoch, grad=False, model=model, optimizer=optimizer,
                                      criterion=criterion, loader=val_loader, device=device)
            logger.info('{} | Fold[{}/{}] Epoch[{}/{}] \n       Metric {}'.format(
                ' val ', fold, config['#fold'], epoch, config['#epoch'], metric))

            #update te best performance
            if metric[config['decision_metric']] > best_metric[config['decision_metric']] :
                best_metric = metric
                save_model(checkpoint=False, fold=fold, epoch=epoch, metric=metric, model=model, optimizer=optimizer)
                logger.debug('Fold[{}/{}] Epoch[{}/{}] Model saved'.format(
                fold, config['#fold'], epoch, config['#epoch']))
                patience = 0
            else :
                patience += 1
                #early stopping
                if patience == config['patience']:
                    logger.debug('Stopped early at Epoch {}'.format(epoch))
                    break
            if epoch % config['checkpoint_term'] == 0:
                save_model(checkpoint=True, fold=fold, epoch=epoch, metric=metric, model=model, optimizer=optimizer)

        if config['#fold'] == 2:
            break

        if config['resume']:
            epochs = range(1, config['#epoch']+1)

        fold_best_metric.append(best_metric)

    logger.info(' -------------------- END OF TRAIN -------------------- ')
    
    #print performance
    for key in config['metric']:
        avg_metric=sum([best_metric[key] for best_metric in fold_best_metric])/len(fold_best_metric)
        logger.info('{}   | Avg {} : {}'.format('val', key, avg_metric))

    #test
    if config['test']:
        test_loader = Dataloader(split='test', fold=0)
        metric_list = []
        for fold in range(1, config['#fold']+1):
            path = os.path.join(config['@root'], config['@save'], config['@best_model'], config['version'], str(fold)+'.pth')
            state = torch.load(path)
            model.load_state_dict(state['model_state'])
            optimizer.load_state_dict(state['optimizer_state'])
            metric = run(fold=fold, epoch=epoch, grad=False, model=model, optimizer=optimizer,
                     criterion=criterion, loader=test_loader, device=device)
            logger.info('{} | Fold[{}/{}] \n       Metric {}'.format(
                'test ', fold, config['#fold'], metric))
            metric_list.append(metric)
        for key in config['metric']:
            avg_metric=sum([metric[key] for metric in metric_list])/len(metric_list)
            logger.info('{}  | Avg {} : {}'.format('test', key, avg_metric))
    