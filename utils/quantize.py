import torch.nn as nn

############################################ QUANTIZER ############################################

import brevitas.nn as qnn
from config import model_config

def quantize_before_training(net):
    for name, module in net._modules.items():
        if len(list(module.children())) > 0:
            if isinstance(module, nn.Conv2d):
                net._modules[name] = qnn.QuantConv2d(in_channels,
                           out_channels,
                           kernel_size,
                           stride=stride,
                           padding=padding,
                           dilation=dilation,
                           groups=groups,
                           bias=bias,
                           padding_mode=pading_mode,
                           weight_quant_type=model_config['weight_quant_type'],
                           weight_bit_width=model_config['weight_bit_width'])
            if isinstance(module, nn.ReLU):
                net._modules[name] = QuantReLU
            if isinstance(module, nn.Linear):
                net._modules[name] = QuantLinear
    return net


def QuantConv2d(in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode='zeros'):
    return qnn.QuantConv2d(in_channels,
                           out_channels,
                           kernel_size,
                           stride=stride,
                           padding=padding,
                           dilation=dilation,
                           groups=groups,
                           bias=bias,
                           padding_mode=pading_mode,
                           weight_quant_type=model_config['weight_quant_type'],
                           weight_bit_width=model_config['weight_bit_width'])


def QuantReLU(inplace=False):
    return qnn.QuantReLU(inplace=inplace,
                         weight_quant_type=model_config['weight_quant_type'],
                         weight_bit_width=model_config['weight_bit_width'])

def QuantLinear(in_features, out_features, bias=True):
    return qq.QuantLinear(in_features=in_features,
                          out_features=out_features,
                          bias=bias,
                          weight_quant_type=model_config['weight_quant_type'],
                          weight_bit_width=model_config['weight_bit_width'])
