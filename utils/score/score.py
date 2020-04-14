################################################  2019 SCORE  ################################################

import torch
import torch.nn as nn
from .profile import *

def score2019(model):
    input = torch.randn(1, 3, 32, 32)
    input = input.to(model.device)
    macs, params = profile(model.net, inputs=(input, ))
    return str2019(params, macs*2)


def str2019(total_params, total_ops):
        title = 'Score for 2019 MicroNet competition :' + '\n' + '\n'
        parameters             = 'Parameters................:  ' + reformat(total_params) + '\n'
        parameters_score       = 'Parameters score..........:  ' + truncate(total_params/34500000) + '\n'
        FLOPs                  = 'FLOPs.....................:  ' + reformat(total_ops) + '\n'
        ops_score              = 'Operations score..........:  ' + truncate((total_ops)/10490000000) + '\n'
        score                  = 'Score.....................:  ' + truncate(total_params/34500000 + (total_ops)/10490000000) + '\n'
        parameters_summary = parameters + parameters_score
        ops_summary = FLOPs + ops_score
        print(80*'_' + '\n' + title + parameters_summary + ops_summary + score + 80*'_')



################################################  2020 SCORE  ################################################



def score2020(model):
    total_params, total_mul, total_add = 0, 0, 0
    net = model.net

    for module in net.modules():

        if isinstance(module, nn.Conv2d):
            print('Conv2d')
            #module_params, module_mul, module_add = count_conv2d(module)
            #total_params += module_params
            #total_mul += module_mul
            #total_add += module_ad

        elif isinstance(module, nn.ReLU):
            print('ReLU')
            #module_params, module_mul, module_add = count_relu(module)
            #total_params += module_params
            #total_mul += module_mul
            #total_add += module_ad

        elif isinstance(module, nn.Linear):
            print('Linear')
           #module_params, module_mul, module_add = count_linear(module)
            #total_params += module_params
            #total_mul += module_mul
            #total_add += module_ad

        elif isinstance(module, nn.BatchNorm2d):
            print('BatchNorm2d')
            #module_params, module_mul, module_add = count_bn(module)
            #total_params += module_params
            #total_mul += module_mul
            #total_add += module_ad

        elif isinstance(module, nn.AvgPool2d):
            print('AvgPool2d')
            #module_params, module_mul, module_add = count_pool(module)
            #total_params += module_params
            #total_mul += module_mul
            #total_add += module_ad

        elif isinstance(module, nn.Dropout):
            print('Dropout')
            #module_params, module_mul, module_add = count_conv2d(module)
            #total_params += module_params
            #total_mul += module_mul
            #total_add += module_ad

        else :
            print("Unsupported module : ", type(module))

    return str2020(total_params, total_mul, total_add)

def str2020(total_params, total_mul, total_add):
        title = 'Score for 2020 MicroNet competition :' + '\n' + '\n'
        parameters             = 'Parameters................:  ' + reformat(total_params) + '\n'
        parameters_score       = 'Parameters score..........:  ' + truncate(total_params/34500000) + '\n'
        multiplication         = 'Multiplication............:  ' + reformat(total_mul) + '\n'
        addition               = 'Addition..................:  ' + reformat(total_add) + '\n'
        FLOPs                  = 'FLOPs.....................:  ' + reformat(total_mul + total_add) + '\n'
        ops_score              = 'Operations score..........:  ' + truncate((total_mul + total_add)/10490000000) + '\n'
        score                  = 'Score.....................:  ' + truncate(total_params/34500000 + (total_mul + total_add)/10490000000) + '\n'
        parameters_summary = parameters + parameters_score
        ops_summary = multiplication + addition + FLOPs + ops_score
        return print(80*'_' + '\n' + title + parameters_summary + ops_summary + score + 80*'_')

def count_relu(module):

    nelements = x.numel()

    m.total_ops += torch.Tensor([int(nelements)])

################################################  UTILS  ################################################

def truncate(num):
    truncate_string = f"{num:.4f}"
    return truncate_string

def reformat(num):
    if num > 1e12:
        num = f"{(num / 1e12):.3f}" + " Trillion"
    elif num > 1e9:
        num = f"{(num / 1e9):.3f}" + " Billion"
    elif num > 1e6:
        num = f"{(num / 1e6):.3f}" + " Million"
    elif num > 1e3:
        num = f"{(num / 1e3):.3f}" + " Thousand"
    else:
        num = str(int(num))
    return num
