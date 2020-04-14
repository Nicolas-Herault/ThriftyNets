from brevitas.core.quant import QuantType

########################################## DATASET ############################################

dataset = 'cifar100' # 'cifar10' or 'cifar100'

############################################ MODEL ############################################

# net can be one of:
# resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202, efficientnetb0, wide_resnet_28_10
# Modify utils/init_net.py to add new net
# See models/ to check which net are availables

model_config = {
    'net': 'thrifty',
    'mode': 'baseline', # can be 'baseline' or 'boosted'
    'activation': 'ReLU', # can be 'ReLU' or 'Mish'
    # baseline means SGD + ROP, 'boosted' means RangerLars + DelayedCosineAnnealingLR
    'label_smoothing': False, # if True, use utils.LabelSmoothingCrossEntropy instead of vanilla CrossEntropy.
    'smoothing': 0.1, # if smoothing = x, final label value will be 1-x instead of 1 with vanilla CrossEntropy.
    'reduction': 'mean', # loss reduction. Can be 'mean' or 'sum'.
    # See the reduce_loss method in utils.label_smoothing for more info
    'quantize': False,
    'weight_quant_type': QuantType.INT, # can be BINARY, TERNARY, INT, FP
    'weight_bit_width': 8
}

######################################### DATALOADER ##########################################

# The default data augmentation parameters can be modified in dataloaders.py
# via the aug variable in get_transforms()
# Cutout parameters should be modified only in the following dictionaries.

dataloader_config = {
    'rootdir': './data/',
    'download': True,
    'batch_size': 128,
    'nb_workers': 6,
    'data_aug': False,
    'use_cutout': False,
    'n_holes': 1,
    'length': 16
}


#########################################  PRUNING  ###########################################

pruning_config = {
    'use_pruning': False,
    'pruning_rate': 0.6, # for non asymptotic pruning only
    'asymptotic': True,
    'min': 0, # minimal pruning rate for asymptotic pruning
    'goal': 0.8, # goal pruning rate for asymptotic pruning
    'D': 1/8, # asymptotic pruning rate speed: at D*nb_epochs, pruning rate will be (3/4)*goal.
    # verbose can be:
    # 0 for no verbose
    # 1 for soft verbose (total zeros and nonzeros)
    # 2 for hard verbose (zeros and nonzeros counts per layer)
    'verbose': 1,
}


########################################### TRAIN #############################################

train_config = {
    'nb_epochs' : 1,
    'tensorboard_path': './runs/'+dataset+'/'+ model_config['net']+'_'+model_config['mode']+'_test'+'/',
    'checkpoints_path': './checkpoints/'+dataset+'/',
    'use_early_stopping': False,
    'patience': 30, # early stopping patience
    'delta': 0.01, # value of loss decrease to cancel early stopping
    'use_binary_connect': False,
    'pruning': pruning_config,
    'verbose': True
}
