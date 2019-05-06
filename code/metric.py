from sklearn.metrics import fbeta_score as FBS
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import roc_auc_score as AUC

from config import config

def metrics(label_list, pred_list, pos_prob_list):
    
    metric_dict = dict()
    for m in config['metric']:
        if m == 'fbs':
            metric_dict[m] = FBS(label_list, pred_list, 1)
        elif m == 'acc':
            metric_dict[m] = ACC(label_list, pred_list)
        elif m == 'auc':
            metric_dict[m] = AUC(label_list, pos_prob_list)
        else :
            print('Error : No such metric. Implement it.')
            raise
    
    return metric_dict
    