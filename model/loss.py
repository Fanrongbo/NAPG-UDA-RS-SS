import torch
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable


def cross_entropyfc(input, target, weight=None, reduction='mean', ignore_index=255):

    target = target.long()
    if target.dim() == 2:
        target = torch.squeeze(target, dim=1)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)
def cross_entropy(input, target, weight=None, reduction='mean', ignore_index=255):

    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)
class UnchgInCenterLoss(nn.Module):

    def __init__(self, gamma=0):
        super(UnchgInCenterLoss, self).__init__()
        # self.focal = FocalLoss(gamma=gamma, alpha=None)

    def cross_entropy(self,input, target, weight=None, reduction='mean', ignore_index=255):
        target = target.long()
        if target.dim() == 4:
            target = torch.squeeze(target, dim=1)

        return F.cross_entropy(input=input, target=target, weight=weight,
                               ignore_index=ignore_index, reduction=reduction)
    def forward(self, predictions, target, margin):
        ce = self.cross_entropy(predictions[0], target)

        dist = torch.abs(predictions[2].mean(1))
        dist = dist.unsqueeze(1)
        target = (target.repeat([1, dist.shape[1], 1, 1])).float()
        zeros = torch.zeros_like(dist)
        ones = torch.ones_like(dist)
        margin_out_sim_out = dist - margin
        margin_out_sim = torch.where(margin_out_sim_out > 0, margin_out_sim_out, zeros).float()
        margin_out_sim_flag = torch.where(margin_out_sim_out > 0, ones, zeros).float()
        unchgnum = (margin_out_sim_flag * (1 - target)).sum() + 1
        unchgFeat = (1 - target) * (torch.exp(margin_out_sim) - 1)
        unchgFeatloss = unchgFeat.sum() / unchgnum
        chgFeat = ((target) * (torch.exp(-dist)))
        chgnum = target.sum() + 1
        chgFeatloss = chgFeat.sum() / chgnum
        CenterLoss=unchgFeatloss+0.1*chgFeatloss

        # print('unchgFeat', unchgFeat.shape,unchgFeatloss, chgFeat.shape, target.shape,chgFeatloss)

        # unchgFeat=(1-target)*predictions[2]
        # unchgDist = torch.abs(unchgFeat).sum(1)
        #
        # zeros = torch.zeros_like(unchgDist)
        # ones = torch.ones_like(unchgDist)
        # margin_out_sim_out = unchgDist - margin
        #
        # margin_unchgDist = torch.where(margin_out_sim_out > 0, margin_out_sim_out, zeros).float()
        # margin_out_sim_flag = torch.where(margin_out_sim_out > 0, ones, zeros).float()
        # unchgnum = (margin_out_sim_flag * (1 - target)).sum() + 1
        # unchgFeat = (1 - target[:,0,:,:]) * (torch.exp(margin_unchgDist) - 1)
        #
        # unchgFeatloss=unchgFeat.sum()/unchgnum
        #
        # chgFeat = ((target) * (torch.exp(-torch.abs(predictions[2]).sum(1))))
        # chgnum = target.sum() + 1
        # chgFeatloss = chgFeat.sum() / (chgnum*predictions[2].shape[1])
        # CenterLoss = unchgFeatloss + 0.1 * chgFeatloss

        return ce, CenterLoss
class UnchgInCenterLossNew(nn.Module):

    def __init__(self, gamma=0):
        super(UnchgInCenterLossNew, self).__init__()
        # self.focal = FocalLoss(gamma=gamma, alpha=None)

    def cross_entropy(self,input, target, weight=None, reduction='mean', ignore_index=255):
        target = target.long()
        if target.dim() == 4:
            target = torch.squeeze(target, dim=1)

        return F.cross_entropy(input=input, target=target, weight=weight,
                               ignore_index=ignore_index, reduction=reduction)
    def forward(self, predictions, target, chgCenter,unchgCenter):
        ce = self.cross_entropy(predictions[0], target)
        if chgCenter is not None:
            # chgCenter=chgCenter*chgCenter
            # unchgCenter=unchgCenter*unchgCenter
            chgCenter = chgCenter
            unchgCenter = unchgCenter
            unchgNum=(1-target).sum([2,3])+1
            chgNum=target.sum([2,3])+1
            prePow=torch.pow(predictions[2],2)#[13, 32, 256, 256]

            # unchgfdist = ((torch.abs(prePow - unchgCenter).mean(1)).unsqueeze(1)) * (1 - target)  # [13, 1, 256, 256]
            # unchgfdist=((torch.abs(prePow-unchgCenter).mean(1)).unsqueeze(1))*(1-target)#[13, 1, 256, 256]
            unchgfdist0 = ((torch.abs(prePow.mean(1) - 0)).unsqueeze(1)) * (1 - target)  # [13, 1, 256, 256]
            unchgfdist0=((unchgfdist0.sum([2,3]))/unchgNum).mean()
            chgfdist0 = ((torch.abs(prePow.mean(1) - 0)).unsqueeze(1)) * (target)
            chgfdist0=((chgfdist0.sum([2,3]))/chgNum).mean()

            unchgfdist = ((torch.abs(prePow.mean(1) - unchgCenter)).unsqueeze(1)) * (1 - target)  # [13, 1, 256, 256]
            chgfdist=((torch.abs(prePow.mean(1)-chgCenter)).unsqueeze(1))*(target)
            unchgFeatloss=((unchgfdist.sum([2,3]))/unchgNum).mean()
            chgFeatloss = ((chgfdist.sum([2,3]))/chgNum).mean()#13, 1
            CenterLoss=unchgFeatloss+chgFeatloss
        else:
            unchgCenter = unchgCenter * unchgCenter
            unchgNum = (1 - target).sum([2, 3]) + 1
            # prePow = torch.exp(predictions[2])-1  # [13, 32, 256, 256]
            prePow = torch.pow(predictions[2],2)  # [13, 32, 256, 256]
            unchgfdist = ((torch.abs(prePow - unchgCenter).mean(1)).unsqueeze(1)) * (1 - target)  # [13, 1, 256, 256]
            # unchgfdist = ((torch.abs(prePow - unchgCenter).mean(1)).unsqueeze(1)) * (1 - target)  # [13, 1, 256, 256]
            unchgFeatloss = (unchgfdist.sum([2, 3])) / unchgNum
            CenterLoss = unchgFeatloss.mean().detach()
        return ce, CenterLoss,[unchgfdist0.item(),chgfdist0.item()]

class UnchgNoCenterLoss(nn.Module):

    def __init__(self, gamma=0):
        super(UnchgNoCenterLoss, self).__init__()
        # self.focal = FocalLoss(gamma=gamma, alpha=None)

    def cross_entropy(self, input, target, weight=None, reduction='mean', ignore_index=255):
        target = target.long()
        if target.dim() == 4:
            target = torch.squeeze(target, dim=1)
        # print('input', input.shape, target.shape)
        return F.cross_entropy(input=input, target=target, weight=weight,
                               ignore_index=ignore_index, reduction=reduction)

    def forward(self, predictions, target):
        ce = self.cross_entropy(predictions[0], target)
        chgnum = target.sum() + 1
        unchgnum = (1-target).sum() + 1
        chgFeat = (target * predictions[2])
        unchgFeat = ((1 - target) * predictions[2])
        chgFeatMean=chgFeat.sum([0, 2, 3]) / chgnum
        unchgFeatMean=unchgFeat.sum([0, 2, 3]) / unchgnum
        dist = torch.square(chgFeatMean - unchgFeatMean)
        NoCenterLoss = (torch.exp(-dist).sum())/predictions[2].shape[1]
        # centerdict={'c':chgFeatMean.detach().cpu().numpy(),'u':unchgFeatMean.detach().cpu().numpy()}

        centerlist=[chgFeatMean.detach().cpu().numpy(),unchgFeatMean.detach().cpu().numpy()]
        return ce, NoCenterLoss,centerlist


class Hybrid(nn.Module):

    def __init__(self, gamma=0):
        super(Hybrid, self).__init__()
        self.focal = FocalLoss(gamma=gamma, alpha=None)

    def forward(self, predictions, target):

        focal = self.focal(predictions, target.long())
        dice = dice_loss(predictions, target.long())
        loss=focal+dice
        return focal, dice,loss


#When gamma=0, the focal loss reduces to Cross Entropy.
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):

        input = input.view(input.size(0), input.size(1), -1)
        input = input.transpose(1, 2)
        input = input.contiguous().view(-1, input.size(2))

        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def dice_loss(logits, true, eps=1e-7):
    device=true.device
    num_classes = logits.shape[1]
    if num_classes == 1:
        # true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = (torch.eye(num_classes + 1).to(torch.device(device)))[true.squeeze(1)]

        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        # true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = (torch.eye(num_classes).to(torch.device(device)))[true.squeeze(1)]

        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)