import torch.nn as nn
import torchvision

model=dict()

def resnet18(n_class):
    resnet = torchvision.models.resnet18(pretrained=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, n_class)
    return resnet

def vgg19bn(n_class):
    vgg19 = torchvision.models.vgg19_bn(pretrained=True)
    vgg19.classifier[6] = nn.Linear(vgg19.classifier[6].in_features, n_class)
    return vgg19

def vgg19bn_hfx(n_class):
    vgg = torchvision.models.vgg19_bn(pretrained=True)
    for param in vgg.features[:26].parameters():
        param.requires_grad = False
    vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, n_class)
    return vgg

def vgg11bn_hfx(n_class):
    vgg = torchvision.models.vgg11_bn(pretrained=True)
    for param in vgg.features[:14].parameters():
        param.requires_grad = False
    vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, n_class)
    return vgg

model['resnet18'] = resnet18
model['vgg19bn'] = vgg19bn
model['vgg19bn_hfx'] = vgg19bn_hfx
model['vgg11bn_hfx'] = vgg11bn_hfx
