import torch.nn as nn
import torch.nn.functional as F



class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing, self.reduction = smoothing, reduction

    def reduce_loss(self, loss, reduction):
        return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

    # linear combination = exponentially weighted moving average
    def lin_comb(self, v1, v2, beta): return beta*v1 + (1-beta)*v2

    def forward(self, output, target):
        nb_classes = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return self.lin_comb(loss/nb_classes, nll, self.smoothing)




        
