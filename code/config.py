import torch.optim as optim
import torch.nn as nn

import argparse
from model import model
from transform import transform

config = dict()

config['#class'] = 2
config['version'] = 'default' #must be uniquely set everytime

config['resume'] = False #cannot use now; should fix error
config['test'] = False

config['#epoch'] = 300
config['#fold'] = 5 #if 2, no cross validation
config['#batch']= 8
config['lr'] = 1e-4
config['img_size'] = (224, 224)

config['transform'] = transform('transform2', config['img_size'])
config['model_name'] = 'vgg19bn_hfx' # w/o ()

config['metric'] = ['auc', 'acc', 'fbs']
config['decision_metric'] = config['metric'][0]

config['patience'] = 20

config['sampler'] = 'None' #'ROS', 'None', ('SOMTE')
config['#sample_ROS'] = 1 #1x

config['optimizer'] = optim.Adam 
config['criterion'] = nn.CrossEntropyLoss
config['gpu'] = '0,1'

config['seed'] = 0 #random seed
config['#worker'] = 4
config['checkpoint_term'] = 5
config['trainlog_term'] = 20

config['visdom'] = None



config['@root'] = '../'
config['@data'] = 'data/'
config['@image'] = 'images/'
config['@table'] = 'data.json'
config['@save'] = 'save/'
config['@checkpoint'] = 'checkpoint/'
config['@best_model'] = 'model/'
config['@log'] = 'log/'




def parse_args():
    parser = argparse.ArgumentParser('Hyperparams | Pytorch Classification code')
    parser.add_argument('--version', nargs='?', type=str, default=None, help='version of model')
    parser.add_argument('--resume', nargs='?', type=bool, default=None, help='if True resume checkpoint')
    parser.add_argument('--test', nargs='?', type=bool, default=None, help='if True test model')
    parser.add_argument('--gpu', nargs='?', type=str, default=None, help='GPU number')
    parser.add_argument('--epoch', nargs='?', type=int, default=None, help='number of epochs')
    parser.add_argument('--batch', nargs='?', type=int, default=None, help='batch size')
    parser.add_argument('--lr', nargs='?', type=float, default=None, help='learning rate')
    parser.add_argument('--patience', nargs='?', type=int, default=None, help='Early stopping patience')
    parser.add_argument('--sampler', nargs='?', type=str, default=None, help='None, ROS, SMOTE')
    parser.add_argument('--visdom', nargs='?', type=int, default=None, help='visdom server port')
    parser.add_argument('--class', nargs='?', type=int, default=None, help='number of class')
    parser.add_argument('--model', nargs='?', type=str, default=None, help='model')
    parser.add_argument('--transform', nargs='?', type=str, default=None, help='transform')
    parser.add_argument('--decision_metric', nargs='?', type=str, default=None, help='decision_metric')
    return parser.parse_args()

args = vars(parse_args())
for key in args:
    if args[key] is not None:
        if key == 'epoch' or key == 'batch' or key == 'class':
            config_key = '#' + key
        elif key == 'model':
            config_key = 'model_name'
        else :
            config_key = key
        if config_key not in config:
            print('Error : config and arg KEY not match')
            raise
        elif config_key == 'transform':
            config[config_key] = transform(args[key], config['img_size'])
        else:
            config[config_key] = args[key]
        
if config['#class'] == 2:
    config['@table'] = 'data.json'
elif config['#class'] == -1:
    config['@table'] = 'data_test.json'
    config['#class'] = 2
else:
    print('Error : Wrong class number')
    raise        
        
config['model'] = model[config['model_name']]