import torch
import torch.nn as nn
from torch.autograd import Variable

from config import config
from metric import metrics
from logger import logger

def run(fold, epoch, grad, model, optimizer, criterion, loader, device):
    if grad :
        model.train()
    else:
        model.eval()

    softmax = nn.Softmax(1)
    label_list = []
    pred_list = []
    pos_prob_list = []
    avg_loss = 0
    for i, (images, labels) in enumerate(loader):
        images = Variable(images.to(device))
        labels = Variable(labels.to(device))
        if grad :
            #model prediction
            outputs = model(images)
            #calcuate loss with model prediction and ground truth
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else :
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)

        prob_outputs = softmax(outputs).data.cpu()
        label_list += labels.data.cpu().tolist()
        pos_prob_list += prob_outputs[:,1].tolist()
        pred_list += prob_outputs.max(1)[1].tolist()
        avg_loss += loss.item()
        if grad and (i+1) % config['trainlog_term'] == 0:
            logger.debug('Fold[{}/{}] Epoch[{}/{}] Batch[{}/{}] AvgLoss[{:.3f}]'.format(
                fold, config['#fold'], epoch, config['#epoch'], i+1, len(loader), avg_loss/config['trainlog_term']))
            avg_loss=0
    metric = metrics(label_list, pred_list, pos_prob_list)

    if grad :
        return model, optimizer, metric
    else :
        return metric

