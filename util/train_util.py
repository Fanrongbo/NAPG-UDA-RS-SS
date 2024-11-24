import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

# from model.networks import *
from model.loss import cross_entropy, Hybrid,UnchgInCenterLoss,UnchgNoCenterLoss,cross_entropyfc,UnchgInCenterLossNew
from util import util
import time
import torch.nn.init as init
from option.config import cfg
from util.kmeanTorch import KMEANS
from sklearn.cluster import KMeans
import numpy as np

class BCL(nn.Module):
    """
    batch-balanced contrastive loss
    no-change，1
    change，-1
    """
    def __init__(self, margin=2.0):
        super(BCL, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        label[label == 1] = -1
        label[label == 0] = 1

        mask = (label != 255).float()
        distance = distance * mask

        pos_num = torch.sum((label==1).float())+0.0001
        neg_num = torch.sum((label==-1).float())+0.0001

        loss_1 = torch.sum((1+label) / 2 * torch.pow(distance, 2)) /pos_num
        loss_2 = torch.sum((1-label) / 2 *
            torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        ) / neg_num
        loss = loss_1 + loss_2
        return loss
class GetCenterNormall(nn.Module):
    def __init__(self, gamma=0):
        super(GetCenterNormall, self).__init__()
    def forward(self, predictions, target, DEVICE):
        chgnum = target.sum() + 1
        unchgnum = (1 - target).sum() + 1
        featNorm=predictions[2].reshape(-1,predictions[2].shape[1])
        # featNorm = torch.cat((featNorm, torch.ones(featNorm.size(0), 1).to(DEVICE)), 1)
        featNorm = (featNorm.t() / torch.norm(featNorm, p=2, dim=1)).t()
        featNorm = featNorm
        target=target.reshape(-1,1)

        chgFeat = (target * featNorm)
        unchgFeat = ((1 - target) * featNorm)
        # print(predictions[2].shape, featNorm.shape)
        chgFeatMean = (chgFeat.sum([0]) / chgnum).unsqueeze(1)
        unchgFeatMean = (unchgFeat.sum([0]) / unchgnum).unsqueeze(1)
        # return [chgFeatMean.detach().float().cpu().numpy(), unchgFeatMean.detach().float().cpu().numpy()]

        return [chgFeatMean.detach(), unchgFeatMean.detach()]
class GetCenterNorm(nn.Module):
    def __init__(self):
        super(GetCenterNorm, self).__init__()
        self.affunchg = np.expand_dims(np.array([1]), axis=0)
        self.affchg = np.expand_dims(np.array([1]),axis=0)
    def forward(self, predictions, target, DEVICE):
        chgnum = target.sum() + 1
        unchgnum = (1 - target).sum() + 1
        chgFeat = (target * predictions[2])

        unchgFeat = ((1 - target) * predictions[2])

        chgFeatMean = (chgFeat.sum([0, 2, 3]) / chgnum).unsqueeze(1)
        # chgFeatMean = torch.cat((chgFeatMean, torch.ones(chgFeatMean.size(0), 1).to(DEVICE)), 1)
        chgFeatMean = (chgFeatMean.t() / torch.norm(chgFeatMean, p=2, dim=1)).t()
        chgFeatMean= chgFeatMean.detach()

        unchgFeatMean = (unchgFeat.sum([0, 2, 3]) / unchgnum).unsqueeze(1)
        # unchgFeatMean = torch.cat((unchgFeatMean, torch.ones(unchgFeatMean.size(0), 1).to(DEVICE)), 1)
        unchgFeatMean = (unchgFeatMean.t() / torch.norm(unchgFeatMean, p=2, dim=1)).t()
        unchgFeatMean= unchgFeatMean.detach()
        return [chgFeatMean, unchgFeatMean]
class GetCenterS(nn.Module):
    def __init__(self):
        super(GetCenterS, self).__init__()
        # self.refs = (torch.LongTensor(range(self.num_classes)).unsqueeze(1)).to(self.device)
    def forward(self, predictions, target,DEVICE):

        chgnum = target.sum() + 1
        unchgnum = (1 - target).sum() + 1
        chgFeat = (target * predictions[2])
        unchgFeat = ((1 - target) * predictions[2])
        chgFeatMean = (chgFeat.sum([0, 2, 3]) / chgnum).unsqueeze(1)
        unchgFeatMean = (unchgFeat.sum([0, 2, 3]) / unchgnum).unsqueeze(1)
        return [chgFeatMean.detach(), unchgFeatMean.detach()]


class GetCenterS2(nn.Module):
    def __init__(self,device):
        super(GetCenterS2, self).__init__()
        self.num_classes=2
        self.device=device
        self.refs = (torch.LongTensor(range(self.num_classes)).unsqueeze(1)).to(self.device)

    def forward(self, predictionsIn, targetIn, DEVICE,val=False):

        centersIterout=0
        predictionsIn = predictionsIn.reshape(predictionsIn.shape[0], predictionsIn.shape[1], -1)
        if not val:
            targetIn=targetIn.reshape(targetIn.shape[0],-1)

        for b in range(predictionsIn.shape[0]):
            # predictions=predictionsIn[b]
            predictions = predictionsIn[b, :, :].transpose(1, 0)
            target=targetIn[b]
            target = target.unsqueeze(0)  # 1, 65536
            # print('target',target.shape)
            mask_l = (target == self.refs).unsqueeze(2).type(torch.cuda.FloatTensor)  # [2, 65536, 1]
            reshaped_feature = (predictions.unsqueeze(0)) # [1, 65536, 32]
            # print('reshaped_feature',reshaped_feature.shape)
            # update centers
            centersIter = torch.sum(reshaped_feature * mask_l, dim=1)  # 计算伪标签情况下每一个目标域特征的质心[2, 32]
            centersIter = centersIter / (mask_l.sum([1, 2]).unsqueeze(1) + 1)
            # print('centersIter',centersIter.shape)
            centersIterout=centersIterout+centersIter
            # chgnum = target.sum() + 1
            # unchgnum = (1 - target).sum() + 1
            # chgFeat = (target * predictions[2])
            # unchgFeat = ((1 - target) * predictions[2])
            # chgFeatMean = (chgFeat.sum([0, 2, 3]) / chgnum).unsqueeze(1)
            # unchgFeatMean = (unchgFeat.sum([0, 2, 3]) / unchgnum).unsqueeze(1)
        centersIterout=centersIterout/predictionsIn.shape[0]
        centersIterout=centersIterout.transpose(1, 0)
        # print(centersIterout.shape)
        return [centersIterout[:,0].unsqueeze(1).detach(), centersIterout[:,1].unsqueeze(1).detach()]

def GetCenterS3(features, pseudo_labels,centroidsLast=None,device=None):
    """
    Calculate centroids of positive and negative samples for each channel.

    :param features: Tensor of shape (batch_size, num_channels, height, width)
    :param pseudo_labels: Tensor of shape (batch_size, 1, height, width)
    :return: Tensor of shape (num_channels, 2) representing the centroids
    """
    batch_size, num_channels, height, width = features.shape
    centroids = torch.zeros(2, num_channels).to(device)
    if centroidsLast==None:
        centroidsLast=torch.zeros(2,num_channels).to(device)
    else:
        centroidsLast=centroidsLast.permute(1,0)
    if len(pseudo_labels)!=2:
        pseudo_labels=[1-pseudo_labels,pseudo_labels]
    for batch in range(batch_size):
        # Extract channel features
        batch_features = features[batch, :, :, :].unsqueeze(0)
        # Calculate centroids for positive and negative samples
        for label, mask in enumerate([pseudo_labels[0][batch],pseudo_labels[1][batch]]):
            # print('label',label,mask.sum())
            if mask.sum() > 0:
                centroid=(batch_features*mask).sum(dim=[2,3])/mask.sum()#[1, 32]
                centroids[label,:]=centroids[label,:]+centroid
            else:
                centroids[label, :] = centroids[label, :] + centroidsLast[label, :]

    centroids=(centroids/batch_size).permute(1,0)
    return centroids

def GetCenterS4(features, pseudo_labels,centroidsLast=None,device=None):
    """
    Calculate centroids of positive and negative samples for each channel.

    :param features: Tensor of shape (batch_size, num_channels, height, width)
    :param pseudo_labels: Tensor of shape (batch_size, 1, height, width)
    :return: Tensor of shape (num_channels, 2) representing the centroids
    """
    batch_size, num_channels, height, width = features.shape
    centroids = torch.zeros(2, num_channels,batch_size).to(device)
    if centroidsLast==None:
        centroidsLast=torch.zeros(2,num_channels).to(device)
    else:
        centroidsLast=centroidsLast.permute(1,0)
    if len(pseudo_labels)!=2:
        pseudo_labels=[1-pseudo_labels,pseudo_labels]
    for batch in range(batch_size):
        # Extract channel features
        batch_features = features[batch, :, :, :].unsqueeze(0)
        # Calculate centroids for positive and negative samples
        for label, mask in enumerate([pseudo_labels[0][batch],pseudo_labels[1][batch]]):
            # print('label',label,mask.sum())
            if mask.sum() > 0:
                centroid=(batch_features*mask).sum(dim=[2,3])/mask.sum()#[1, 32]
                centroids[label,:,batch]=centroid
            else:
                centroids[label, :,batch] =  centroidsLast[label, :]

    # centroids=(centroids/batch_size).permute(1,0)
    # print('centroids',centroids.shape)#torch.Size([2, 32, 14])
    return centroids

class GetCenterT(nn.Module):
    def __init__(self):
        super(GetCenterT, self).__init__()
    def forward(self, predictions):
        center=(predictions.mean([0,2,3])).unsqueeze(1)
        return center

class DIST(object):
    def __init__(self, dist_type='cos'):
        self.dist_type = dist_type

    def get_dist(self, pointA, pointB, cross=False):
        # print('pointA',pointA.shape, pointB.shape)
        return getattr(self, self.dist_type)(
		pointA, pointB, cross)

    def cos(self, pointA, pointB, cross):
        # print('pointA',pointA.sum())
        pointA = F.normalize(pointA, dim=1)
        pointB = F.normalize(pointB, dim=1)

        if not cross:
            dist = F.cosine_similarity(pointA,pointB, dim=1)
            # return 0.5 * (1.0 - torch.sum(pointA * pointB, dim=1))
            return dist
        else:
            NA = pointA.size(0)
            NB = pointB.size(0)
            assert(pointA.size(1) == pointB.size(1))
            return 0.5 * (1.0 - torch.matmul(pointA, pointB.transpose(0, 1)))
            # return F.cosine_similarity(pointA,pointB, dim=1)

class CenterTOp(nn.Module):
    def __init__(self, DEVICE,dist_type='cos'):
        super(CenterTOp,self).__init__()
        self.Dist = DIST(dist_type)
        self.device = DEVICE
        self.num_classes=2
        self.refs = (torch.LongTensor(range(self.num_classes)).unsqueeze(1)).to(self.device)

    def to_onehot(self,label, num_classes):
        identity = (torch.eye(num_classes)).to(self.device)
        onehot = torch.index_select(identity, 0, label)
        return onehot
    def assign_labels(self, feats,filter=False):  # 分别计算每一个目标域特征与质心之间的距离，并将最近距离的质心标签给目标域打上

        dists = self.Dist.get_dist(feats, self.centers, cross=True)
        _, labels = torch.min(dists, dim=1)
        # print('label',labels)
        if filter:
            threshold=0.3
            min_dist = torch.min(dists, dim=1)[0]  ##计算target每一类别的质心与source质心的最小距离
            mask=(min_dist<threshold).to(self.device)
            feats = torch.cat([feats[m] for m in range(mask.size(0)) if mask[m].item() == 1],dim=0)
            dists = torch.cat([dists[m] for m in range(mask.size(0)) if mask[m].item() == 1],dim=0)
            labels = torch.masked_select(labels, mask)
        return dists, labels,feats
    def forward(self,FeatureT,centerInit):
        count = 0
        centersIter=None
        Ci=0
        CdistThreshold=0.01

        FeatureT = FeatureT.reshape(FeatureT.shape[0], FeatureT.shape[1], -1)
        centersIterout=0
        labelsout=[]
        labels_onehotout=[]
        dist2centerT=[]
        Cinidist=0
        count=0

        for b in range(FeatureT.shape[0]):
            while True:
                if centersIter is None:
                    self.centers = centerInit
                    FeatureTb=FeatureT[b,:,:].transpose(1,0)#[32, 65536]
                else:
                    self.centers = centersIter
                    if Cdist<CdistThreshold or Ci>3:
                    # if  Ci > 2:
                        # Cinidist=Cinidist+Cdist
                        labelsout.append(labels)
                        labels_onehotout.append(labels_onehot.unsqueeze(0))
                        dist2centerT.append(dist2center.unsqueeze(0))
                        centersIterout = centersIterout + centersIter
                        # centersIter = None
                        Ci=0
                        break
                dist2center, labels,FeatureTb = self.assign_labels(FeatureTb,filter=False)  # [65536, 2] [65536] [65536, 32]
                # labels=1-labels
                labels_onehot = self.to_onehot(labels, self.num_classes)#[65536, 2]


                # count = torch.sum(labels_onehot, dim=0)  #[2] count the num of each category
                labels = labels.unsqueeze(0)#1, 65536

                mask_l = (labels == self.refs).unsqueeze(2).type(torch.cuda.FloatTensor)#[2, 65536, 1]
                reshaped_feature = FeatureTb.unsqueeze(0) # [1, 65536, 32]
                # update centers
                centersIter = torch.sum(reshaped_feature * mask_l, dim=1)  # 计算伪标签情况下每一个目标域特征的质心[2, 32]
                # mask = (count.unsqueeze(1) > 0).type(torch.cuda.FloatTensor)
                # centersIter = mask * centersIter + (1 - mask) * self.centers  # [2, 32] 滤除掉目标域没有的类的特征
                centersIter=centersIter/(mask_l.sum([1,2]).unsqueeze(1)+1)

                Cdist = torch.mean(self.Dist.get_dist(centersIter, self.centers) ,dim=0)
                # print('Cdist',Cdist)
                if Ci==0: Cinidist=Cinidist+Cdist
                Ci=Ci+1
        centersIterout=centersIterout/FeatureT.shape[0]
        labelsout=torch.cat(labelsout,dim=0)
        labels_onehotout=torch.cat(labels_onehotout,dim=0)#[13, 65536, 2]
        dist2centerT=torch.cat(dist2centerT,dim=0)#[13, 65536, 2]
        # dist2centerT=(dist2centerT*labels_onehotout)#[13, 65536, 2]
        # dist2centerTmax1=torch.max(dist2centerT,dim=1)[0].unsqueeze(1)
        # dist2centerT=(dist2centerT.max(1)[0].unsqueeze(1)-dist2centerT)/(dist2centerT.max(1)[0].unsqueeze(1)-dist2centerT.min(1)[0].unsqueeze(1)+0.0000001)#[13, 65536, 2]
        # print(torch.cat([dist2centerT.max(1)[0].unsqueeze(1),dist2centerT.min(1)[0].unsqueeze(1)],dim=1).sum([0,1])) #[2]
        dist2centerT=-dist2centerT/(dist2centerT.max(1)[0].unsqueeze(1)-dist2centerT.min(1)[0].unsqueeze(1)+0.0000001)#[13, 65536, 2]
        # print(dist2centerT.shape)
        dist2centerT=(1-dist2centerT)+0.1
        # dist2centerT = 1-dist2centerT/(dist2centerT.sum(1).unsqueeze(1))
        # print(dist2centerT.shape)

        # dist2centerT=torch.cat([dist2centerT[:,:,1].unsqueeze(2),dist2centerT[:,:,0].unsqueeze(2)],dim=2)
        # labels_onehotout=1-labels_onehotout
        # zeros=torch.zeros_like(dist2centerT)
        # print('dist2centerT', dist2centerT.shape,dist2centerT.sum([0,1]))
        Weight=(dist2centerT*labels_onehotout).sum(2)#[13, 65536, 2]
        # print('dist2centerT',dist2centerT.shape)
        # print(dist2centerT.shape,dist2centerT.max(1)[0],dist2centerT.min(1)[0])
        Cinidist=Cinidist/FeatureT.shape[0]
        return centersIterout.detach(),[labelsout,labels_onehotout,Weight],Cinidist.detach()

class CenterTOp2(nn.Module):
    def __init__(self, DEVICE, dist_type='cos'):
        super(CenterTOp2, self).__init__()
        self.Dist = DIST(dist_type)
        self.device = DEVICE
        self.num_classes = 2
        self.refs = (torch.LongTensor(range(self.num_classes)).unsqueeze(1)).to(self.device)

    def to_onehot(self, label, num_classes):
        identity = (torch.eye(num_classes)).to(self.device)
        onehot = torch.index_select(identity, 0, label)
        return onehot

    def assign_labels(self, feats, filter=False):  # 分别计算每一个目标域特征与质心之间的距离，并将最近距离的质心标签给目标域打上
        dists = self.Dist.get_dist(feats, self.centers, cross=True)
        _, labels = torch.min(dists, dim=1)

        if filter:
            threshold = 0.3
            min_dist = torch.min(dists, dim=1)[0]  ##计算target每一类别的质心与source质心的最小距离
            mask = (min_dist < threshold).to(self.device)
            feats = torch.cat([feats[m] for m in range(mask.size(0)) if mask[m].item() == 1], dim=0)
            dists = torch.cat([dists[m] for m in range(mask.size(0)) if mask[m].item() == 1], dim=0)
            labels = torch.masked_select(labels, mask)
        return dists, labels, feats

    def forward(self, FeatureT, centerInit):
        centersIter = None
        Ci = 0
        CdistThreshold = 0.01

        FeatureT = FeatureT.reshape(FeatureT.shape[0], FeatureT.shape[1], -1)

        labelsout = []
        Cinidist = 0
        centersIterout = 0
        CdistT = 0
        while True:
            Ci = Ci + 1
            if centersIter is None:
                self.centers = centerInit
                FeatureT = FeatureT.reshape(FeatureT.shape[0], FeatureT.shape[1], -1)
            else:
                self.centers = centersIterout/FeatureT.shape[0]
                CurDist=CdistT/FeatureT.shape[0]
                if CurDist < CdistThreshold or Ci > 3:
                    labels_onehot_out=[]
                    for label in labelsout:
                        labels_onehot_out.append(self.to_onehot(label.squeeze(0), self.num_classes))
                    labels_onehot_out=torch.cat(labels_onehot_out,dim=0)
                    labelsout=torch.cat(labelsout,dim=0)
                    centersIterout=self.centers
                    break
            CdistT = 0
            centersIterout = 0
            labelsout=[]
            for b in range(FeatureT.shape[0]):
                FeatureTb = FeatureT[b, :, :].transpose(1, 0)  # [32, 65536]
                dist2center, labels, FeatureTb = self.assign_labels(FeatureTb,
                                                                    filter=False)  # [65536, 2] [65536] [65536, 32]
                # labels_onehot = self.to_onehot(labels, self.num_classes)  # [65536, 2]
                labels = labels.unsqueeze(0)# 1, 65536
                mask_l = (labels == self.refs).unsqueeze(2).type(torch.cuda.FloatTensor)  # [2, 65536, 1]
                reshaped_feature = FeatureTb.unsqueeze(0)  # [1, 65536, 32]
                # print(mask_l)
                centersIter = torch.sum(reshaped_feature * mask_l, dim=1)  # 计算伪标签情况下每一个目标域特征的质心[2, 32]

                centersIter = centersIter / (mask_l.sum([1, 2]).unsqueeze(1)+1)

                centersIterout=centersIterout+centersIter
                Cdist = torch.mean(self.Dist.get_dist(centersIter, self.centers), dim=0)
                # print(centersIter,self.centers)
                CdistT=CdistT+Cdist
                # if Ci == 1: Cinidist = Cinidist + Cdist
                labelsout.append(labels)

        # centersIterout = centersIterout / FeatureT.shape[0]
        # Cinidist = Cinidist / FeatureT.shape[0]
        return centersIterout.detach(), [labelsout, labels_onehot_out], CurDist
class CenterTOpEX(nn.Module):
    def __init__(self, DEVICE,dist_type='cos'):
        super(CenterTOpEX,self).__init__()
        self.Dist = DIST(dist_type)
        self.device = DEVICE
        self.num_classes=2
        self.refs = (torch.LongTensor(range(self.num_classes)).unsqueeze(1)).to(self.device)
    # def to_onehot2(self,labelT):
    #     y_one_hot = torch.zeros(labelT.shape[0], self.num_classes).scatter_(1, labelT, 1)

    def to_onehot(self,label, num_classes):
        identity = (torch.eye(num_classes)).to(self.device)
        onehot = torch.index_select(identity, 0, label)
        return onehot
    def assign_labels(self, feats,filter=False):  # 分别计算每一个目标域特征与质心之间的距离，并将最近距离的质心标签给目标域打上
        dists = self.Dist.get_dist(feats, self.centers, cross=True)
        _, labels = torch.min(dists, dim=1)

        if filter:
            threshold=0.3
            min_dist = torch.min(dists, dim=1)[0]  ##计算target每一类别的质心与source质心的最小距离
            mask=(min_dist<threshold).to(self.device)
            feats = torch.cat([feats[m] for m in range(mask.size(0)) if mask[m].item() == 1],dim=0)
            dists = torch.cat([dists[m] for m in range(mask.size(0)) if mask[m].item() == 1],dim=0)
            labels = torch.masked_select(labels, mask)
        return dists, labels,feats
    def forward(self,FeatureT,centerInit):
        centersIter=None
        Ci=0
        CdistThreshold=0.01

        FeatureT = FeatureT.reshape(FeatureT.shape[0], FeatureT.shape[1], -1)
        centersIterout=0
        labelsout=[]
        labels_onehotout=[]
        dist2centerT=[]
        Cinidist=0
        labelPinit=[]
        for b in range(FeatureT.shape[0]):
            while True:
                if Ci==0 and centersIter is not None:
                    FeatureTb = FeatureT[b, :, :].transpose(1, 0).contiguous()  # [32, 65536]
                    # self.centers=self.centers+0.1*(centersIter-self.centers)
                    self.centers = centerInit
                elif Ci==0 and centersIter is None:
                    FeatureTb = FeatureT[b, :, :].transpose(1, 0).contiguous()  # [32, 65536]
                    self.centers = centerInit
                else:
                    self.centers = centersIter
                    # if Cdist<CdistThreshold or Ci>3:
                    if Ci > 5:
                        # Cinidist=Cinidist+Cdist
                        labelsout.append(labels)
                        labels_onehotout.append(labels_onehot.unsqueeze(0))
                        dist2centerT.append(dist2center.unsqueeze(0))
                        centersIterout = centersIterout + centersIter
                        # centersIter = None
                        Ci = 0
                        break

                dist2center, labels,FeatureTb = self.assign_labels(FeatureTb,filter=False)  # [65536, 2] [65536] [65536, 32]
                # labels=1-labels
                labels_onehot = self.to_onehot(labels, self.num_classes)#[65536, 2]
                # count = torch.sum(labels_onehot, dim=0)  #[2] count the num of each category
                labels = labels.unsqueeze(0)#1, 65536
                if Ci==0:
                    labelPinit.append(labels)

                mask_l = (labels == self.refs).unsqueeze(2).type(torch.cuda.FloatTensor)#[2, 65536, 1]
                reshaped_feature = FeatureTb.unsqueeze(0) # [1, 65536, 32]
                # update centers
                centersIter = torch.sum(reshaped_feature * mask_l, dim=1)  # 计算伪标签情况下每一个目标域特征的质心[2, 32]
                # mask = (count.unsqueeze(1) > 0).type(torch.cuda.FloatTensor)
                # centersIter = mask * centersIter + (1 - mask) * self.centers  # [2, 32] 滤除掉目标域没有的类的特征
                centersIter=centersIter/(mask_l.sum([1,2]).unsqueeze(1)+1)

                Cdist = torch.mean(self.Dist.get_dist(centersIter, self.centers) ,dim=0)
                # print('Cdist',Cdist)
                if Ci==0: Cinidist=Cinidist+Cdist
                Ci=Ci+1
        centersIterout=centersIterout/FeatureT.shape[0]
        print('centersIterout',centersIterout.shape)
        labelsout=torch.cat(labelsout,dim=0)
        labelPinit=torch.cat(labelPinit,dim=0)
        labels_onehotout=torch.cat(labels_onehotout,dim=0)#[13, 65536, 2]
        dist2centerT=torch.cat(dist2centerT,dim=0)#[13, 65536, 2]
        # dist2centerT=(dist2centerT*labels_onehotout)#[13, 65536, 2]

        # dist2centerTmax1=torch.max(dist2centerT,dim=1)[0].unsqueeze(1)
        # print(dist2centerT.max(1)[0].shape)  13,2
        # dist2centerT=(dist2centerT.max(1)[0].unsqueeze(1)-dist2centerT)/(dist2centerT.max(1)[0].unsqueeze(1)-dist2centerT.min(1)[0].unsqueeze(1)+0.0000001)#[13, 65536, 2]
        # print(torch.cat([dist2centerT.max(1)[0].unsqueeze(1),dist2centerT.min(1)[0].unsqueeze(1)],dim=1).sum([0,1])) #[2]
        dist2centerT=(dist2centerT-dist2centerT.min(1)[0].unsqueeze(1))/(dist2centerT.max(1)[0].unsqueeze(1)-dist2centerT.min(1)[0].unsqueeze(1)+0.0000001)
        # dist2centerT=dist2centerT/(dist2centerT.max(1)[0].unsqueeze(1)-dist2centerT.min(1)[0].unsqueeze(1)+0.0000001)#[13, 65536, 2]
        dist2centerT=(1-dist2centerT)

        # dist2centerT = 1-dist2centerT/(dist2centerT.sum(1).unsqueeze(1))
        # print(dist2centerT.shape)

        # dist2centerT=torch.cat([dist2centerT[:,:,1].unsqueeze(2),dist2centerT[:,:,0].unsqueeze(2)],dim=2)
        # labels_onehotout=1-labels_onehotout
        # zeros=torch.zeros_like(dist2centerT)
        # print('dist2centerT', dist2centerT.shape,dist2centerT.sum([0,1]))
        Weight=dist2centerT

        # Weight=(dist2centerT*labels_onehotout).sum(2)#[13, 65536, 2]
        # print('dist2centerT',dist2centerT.shape)
        # print(dist2centerT.shape,dist2centerT.max(1)[0],dist2centerT.min(1)[0])
        Cinidist=Cinidist/FeatureT.shape[0]

        return centersIterout.detach(),[labelsout,labels_onehotout,Weight,labelPinit],Cinidist.detach()

class CenterTOpEXnew(nn.Module):
    def __init__(self, DEVICE,dist_type='cos'):
        super(CenterTOpEXnew,self).__init__()
        self.Dist = DIST(dist_type)
        self.device = DEVICE
        self.num_classes=2
        self.refs = (torch.LongTensor(range(self.num_classes)).unsqueeze(1)).to(self.device)
    # def to_onehot2(self,labelT):
    #     y_one_hot = torch.zeros(labelT.shape[0], self.num_classes).scatter_(1, labelT, 1)

    def to_onehot(self,label, num_classes):
        identity = (torch.eye(num_classes)).to(self.device)
        onehot = torch.index_select(identity, 0, label)
        return onehot
    def assign_labels(self, feats,filter=False):  # 分别计算每一个目标域特征与质心之间的距离，并将最近距离的质心标签给目标域打上
        # print('feats', feats.sum())
        dists = self.Dist.get_dist(feats, self.centers, cross=True)
        _, labels = torch.min(dists, dim=1)

        if filter:
            threshold=0.3
            min_dist = torch.min(dists, dim=1)[0]  ##计算target每一类别的质心与source质心的最小距离
            mask=(min_dist<threshold).to(self.device)
            feats = torch.cat([feats[m] for m in range(mask.size(0)) if mask[m].item() == 1],dim=0)
            dists = torch.cat([dists[m] for m in range(mask.size(0)) if mask[m].item() == 1],dim=0)
            labels = torch.masked_select(labels, mask)
        return dists, labels,feats
    def selecdata(self, feature, label):
        #label 6, 1, 128, 128
        #feature 6, 32, 128, 128

        label_flatten = label
        feature_flatten = feature
        label_index = torch.nonzero(label_flatten)
        label_index=label_index.squeeze(1)
        # print('label_index',label_index.shape,feature.shape)

        # label_index = torch.flatten(label_index)
        # label_index_rand = torch.randperm(label_index.nelement())
        # label_index = label_index[label_index]
        feature_flatten_select = feature_flatten[label_index]#bs,c

        return feature_flatten_select, label_index
    def forward(self,FeatureT,centerInit,num1,num2,varflag=False):
        centersIter=None
        Ci=0
        CdistThreshold=0.01

        FeatureT = FeatureT.reshape(FeatureT.shape[0], FeatureT.shape[1], -1)
        centersIterout=0
        labelsout=[]
        labels_onehotout=[]
        dist2centerT=[]
        Cinidist=0
        labelPinit=[]
        for b in range(FeatureT.shape[0]):
            while True:
                if Ci==0 and centersIter is not None:
                    FeatureTb = FeatureT[b, :, :].transpose(1, 0)  # [32, 65536]
                    # self.centers=self.centers+0.1*(centersIter-self.centers)
                    self.centers = centerInit
                elif Ci==0 and centersIter is None:
                    FeatureTb = FeatureT[b, :, :].transpose(1, 0)  # [32, 65536]
                    self.centers = centerInit
                else:
                    self.centers = self.centers+0.1*(centersIter-self.centers)
                    # self.centers=centersIter
                    # print(b,'self.centers',self.centers.sum(1)[0].detach().cpu().numpy())
                    # print(b, 'self.centers', self.centers.sum(1)[1].detach().cpu().numpy())
                    # if Cdist<CdistThreshold or Ci>3:
                    if Ci > 5:
                        # Cinidist=Cinidist+Cdist
                        labelsout.append(labels)
                        labels_onehotout.append(labels_onehot.unsqueeze(0))
                        dist2centerT.append(dist2center.unsqueeze(0))
                        centersIterout = centersIterout + centersIter
                        # centersIter = None
                        Ci = 0
                        break

                dist2center, labels,_ = self.assign_labels(FeatureTb,filter=False)  # [65536, 2] [65536] [65536, 32]
                # labels=1-labels
                # print(b,'dist2center',dist2center.sum(),FeatureTb.sum(),self.centers.sum())
                labels_onehot = self.to_onehot(labels, self.num_classes)#[65536, 2]
                dist2centerTh=dist2center*labels_onehot#[65536, 2]

                chgNum=labels.sum()+1
                unchgNum=(1-labels).sum()+1
                chgDistMean=(dist2centerTh[:,1]).sum()/chgNum
                unchgDistMean=(dist2centerTh[:,0]).sum()/unchgNum
                # print()
                zeros = torch.zeros_like(labels)
                if varflag:

                    # dist2centerVar=torch.var(dist2centerTh,dim=0)
                    dist2centerunchg,_=self.selecdata(dist2centerTh[:,0],labels_onehot[:,0])
                    dist2centerchg,_=self.selecdata(dist2centerTh[:,1],labels_onehot[:,1])
                    dist2centerunchgvar=torch.var(dist2centerunchg)
                    dist2centerchgvar=torch.var(dist2centerchg)

                    # print('dist2centerVar',dist2centerunchg.shape,chgNum,dist2centerchg.shape,unchgNum)
                    unchgFeatFilterOneHot = torch.where(dist2centerTh[:, 0] > unchgDistMean + num2 * dist2centerunchgvar,
                                                        zeros, 1 - labels).float()
                    chgFeatFilterOneHot = torch.where(dist2centerTh[:, 1] > chgDistMean - num1 * dist2centerchgvar,
                                                      zeros, labels).float()
                else:
                    unchgFeatFilterOneHot = torch.where(dist2centerTh[:,0] > unchgDistMean*num2, zeros, 1-labels).float()
                    chgFeatFilterOneHot = torch.where(dist2centerTh[:, 1] > chgDistMean / num1, zeros, labels).float()
                # unchgFeatFilterOneHot=1-labels
                # print(b,'chgNum',chgNum,chgFeatFilterOneHot.sum(),chgDistMean,(dist2centerTh[:,1]).sum())
                # print(b,'unchgNum',unchgNum,unchgFeatFilterOneHot.sum(),unchgDistMean,(dist2centerTh[:,0]).sum())

                centersIterchg=(FeatureTb*chgFeatFilterOneHot.unsqueeze(1)).sum(0)/(chgFeatFilterOneHot.sum()+1)
                centersIterunchg=(FeatureTb*unchgFeatFilterOneHot.unsqueeze(1)).sum(0)/(unchgFeatFilterOneHot.sum()+1)
                centersIter=torch.Tensor(torch.cat([centersIterunchg.unsqueeze(0),centersIterchg.unsqueeze(0)],dim=0)).to(self.device)
                # print(b,'centersIter',centersIter.sum())
                # centersIter=centersIter.mean([1])

                labels = labels.unsqueeze(0)#1, 65536
                if Ci==1:
                    labelPinit.append(labels)
                #
                # mask_l = (labels == self.refs).unsqueeze(2).type(torch.cuda.FloatTensor)#[2, 65536, 1]
                # reshaped_feature = FeatureTb.unsqueeze(0) # [1, 65536, 32]
                # print('sum',(reshaped_feature * mask_l).shape)
                # # update centers
                # centersIter = torch.sum(reshaped_feature * mask_l, dim=1)  # 计算伪标签情况下每一个目标域特征的质心[2, 65536, 32]->[2, 32]
                # mask = (count.unsqueeze(1) > 0).type(torch.cuda.FloatTensor)
                # centersIter = mask * centersIter + (1 - mask) * self.centers  # [2, 32] 滤除掉目标域没有的类的特征
                # centersIter=centersIter/(mask_l.sum([1,2]).unsqueeze(1)+1)

                # Cdist = torch.mean(self.Dist.get_dist(centersIter, self.centers) ,dim=0)
                # Cdist = torch.mean(self.Dist.get_dist(centersIter, self.centers), dim=0)
                # print('Cdist',Cdist)
                if Ci==0: Cinidist=F.cosine_similarity(centersIter, self.centers)
                Ci=Ci+1
        centersIterout=centersIterout/FeatureT.shape[0]
        labelsout=torch.cat(labelsout,dim=0)
        labelPinit=torch.cat(labelPinit,dim=0)
        labels_onehotout=torch.cat(labels_onehotout,dim=0)#[13, 65536, 2]
        dist2centerTori=torch.cat(dist2centerT,dim=0)#[13, 65536, 2]
        # dist2centerT=(dist2centerT*labels_onehotout)#[13, 65536, 2]

        # dist2centerTmax1=torch.max(dist2centerT,dim=1)[0].unsqueeze(1)
        # print(dist2centerT.max(1)[0].shape)  13,2
        # dist2centerT=(dist2centerT.max(1)[0].unsqueeze(1)-dist2centerT)/(dist2centerT.max(1)[0].unsqueeze(1)-dist2centerT.min(1)[0].unsqueeze(1)+0.0000001)#[13, 65536, 2]
        # print(torch.cat([dist2centerT.max(1)[0].unsqueeze(1),dist2centerT.min(1)[0].unsqueeze(1)],dim=1).sum([0,1])) #[2]

        dist2centerT=(dist2centerTori-dist2centerTori.min(1)[0].unsqueeze(1))/(dist2centerTori.max(1)[0].unsqueeze(1)-dist2centerTori.min(1)[0].unsqueeze(1)+0.0000001)
        dist2centerT=(1-dist2centerT)

        # dist2centerT = 1-dist2centerT/(dist2centerT.sum(1).unsqueeze(1))
        # print(dist2centerT.shape)

        # dist2centerT=torch.cat([dist2centerT[:,:,1].unsqueeze(2),dist2centerT[:,:,0].unsqueeze(2)],dim=2)
        # labels_onehotout=1-labels_onehotout
        # zeros=torch.zeros_like(dist2centerT)
        # print('dist2centerT', dist2centerT.shape,dist2centerT.sum([0,1]))
        Weight=dist2centerT

        # Weight=(dist2centerT*labels_onehotout).sum(2)#[13, 65536, 2]
        # print('dist2centerT',dist2centerT.shape)
        # print(dist2centerT.shape,dist2centerT.max(1)[0],dist2centerT.min(1)[0])
        Cinidist=Cinidist.sum()/FeatureT.shape[0]
        # print('Cinidist',Cinidist.shape)
        return centersIterout.detach(),[labelsout,labels_onehotout,Weight,dist2centerTori,labelPinit],Cinidist.detach()
class CenterTOpEXnewMultiC(nn.Module):
    def __init__(self, DEVICE,dist_type='cos'):
        super(CenterTOpEXnewMultiC,self).__init__()
        self.Dist = DIST(dist_type)
        self.device = DEVICE
        self.num_classes=2
        self.refs = (torch.LongTensor(range(self.num_classes)).unsqueeze(1)).to(self.device)
    # def to_onehot2(self,labelT):
    #     y_one_hot = torch.zeros(labelT.shape[0], self.num_classes).scatter_(1, labelT, 1)
    def to_onehot(self,label, num_classes):
        identity = (torch.eye(num_classes)).to(self.device)
        onehot = torch.index_select(identity, 0, label)
        return onehot
    def assign_labels(self, feats, filter=False):  # 分别计算每一个目标域特征与质心之间的距离，并将最近距离的质心标签给目标域打上
        dists = self.Dist.get_dist(feats, self.centers, cross=True)
        _, labelsori = torch.min(dists, dim=1)
        if filter:
            zeros=torch.zeros_like(labelsori)
            ones=torch.ones_like(labelsori)
            # print('self.unchgCenterNum-1',self.unchgCenterNum-1)
            labels=torch.where(labelsori>self.unchgCenterNum-1,ones,zeros)

        return dists, labels,labelsori
    def selecdata(self, feature, label):
        #label 6, 1, 128, 128
        #feature 6, 32, 128, 128
        label_flatten = label
        feature_flatten = feature
        label_index = torch.nonzero(label_flatten)
        label_index=label_index.squeeze(1)
        feature_flatten_select = feature_flatten[label_index]#bs,c

        return feature_flatten_select, label_index
    def forward(self,FeatureT,centerInit,num1,num2,varflag=False,unchgN=1,chgN=1,iterC=False):
        self.unchgCenterNum=unchgN
        self.chgCenterNum=chgN
        centersIter=None
        Ci=0 #num of iter
        CdistThreshold=0.01
        B,C,W,H=FeatureT.shape
        # FeatureT = FeatureT.reshape(FeatureT.shape[0], FeatureT.shape[1], -1)
        FeatureT = FeatureT.reshape(FeatureT.shape[0], FeatureT.shape[1], -1)

        centersIterout=0
        labelsout=[]
        labels_onehotout=[]
        dist2centerT=[]
        Cinidist=0
        labelPinit=[]
        for b in range(FeatureT.shape[0]):
            while True:
                if iterC:
                    if Ci==0 and centersIter is None:
                        self.centers = centerInit
                        FeatureTb = FeatureT[b, :, :].transpose(1, 0)#init feature
                    elif Ci==0 and centersIter is not None:
                        FeatureTb = FeatureT[b, :, :].transpose(1, 0)  # [32, 65536]
                        self.centers = self.centers + CinidistW*CinidistW * (centersIter - self.centers)#weighted
                    else:
                        self.centers = self.centers + CinidistW*CinidistW * (centersIter - self.centers)
                        if Ci > 2:
                            labelsout.append(labels)
                            labels_onehotout.append(labels_onehot.unsqueeze(0))
                            dist2centerT.append(dist2center.unsqueeze(0))
                            centersIterout = centersIterout + centersIter
                            Ci = 0
                            break
                else:
                    if Ci==0 and centersIter is not None:
                        FeatureTb = FeatureT[b, :, :].transpose(1, 0)  # [32, 65536]
                        # self.centers=self.centers+0.1*(centersIter-self.centers)
                        self.centers = centerInit
                    elif Ci==0 and centersIter is None:
                        FeatureTb = FeatureT[b, :, :].transpose(1, 0)  # [32, 65536]
                        self.centers = centerInit
                    else:
                        self.centers = self.centers+0.1 * (centersIter - self.centers)
                        # self.centers=centersIter
                        # if Cdist<CdistThreshold or Ci>3:
                        if Ci > 0:
                            # Cinidist=Cinidist+Cdist
                            labelsout.append(labels)
                            labels_onehotout.append(labels_onehot.unsqueeze(0))
                            dist2centerT.append(dist2center.unsqueeze(0))
                            centersIterout = centersIterout + centersIter

                            # centersIter = None
                            Ci = 0
                            break
                dist2center, labels, labelsori = self.assign_labels(FeatureTb,filter=True)  # [65536, 2] [65536] [65536, 32]
                dist2center=torch.cat([dist2center[:,0:self.unchgCenterNum].mean(1).unsqueeze(1),
                                       dist2center[:,self.unchgCenterNum:].mean(1).unsqueeze(1)],dim=1)

                labels_onehot = self.to_onehot(labels, self.num_classes)#[65536, 2]
                dist2centerTh=dist2center*labels_onehot#[65536, 2]

                chgNum=labels.sum()+1
                unchgNum=(1-labels).sum()+1
                chgDistMean=(dist2centerTh[:,1]).sum()/chgNum
                unchgDistMean=(dist2centerTh[:,0]).sum()/unchgNum
                # zeros = torch.zeros_like(labels)
                if varflag:
                    dist2centerunchg, _ = self.selecdata(dist2centerTh[:, 0], labels_onehot[:, 0])
                    dist2centerchg, _ = self.selecdata(dist2centerTh[:, 1], labels_onehot[:, 1])
                    dist2centerunchgvar = torch.var(dist2centerunchg)
                    dist2centerchgvar = torch.var(dist2centerchg)
                    # dist2centerVar=torch.var(dist2centerTh,dim=0)
                    labelOriOnehot = torch.zeros(labelsori.shape[0], self.unchgCenterNum + self.chgCenterNum).to(
                        self.device).scatter_(1, labelsori.unsqueeze(1), 1)  # ([65536, 6])
                    zeros = torch.zeros_like(labelOriOnehot)
                    labelOriOnehot = torch.where(dist2centerTh[:, 1].unsqueeze(1) > chgDistMean +num1*dist2centerchgvar, zeros,
                                                 labelOriOnehot).float()
                    labelOriOnehot = torch.where(dist2centerTh[:, 0].unsqueeze(1) > unchgDistMean +num2*dist2centerunchgvar, zeros,
                                                 labelOriOnehot).float()  # [65536, 6]

                else:
                    labelOriOnehot = torch.zeros(labelsori.shape[0], self.unchgCenterNum + self.chgCenterNum).to(
                        self.device).scatter_(1, labelsori.unsqueeze(1), 1)  # ([65536, 6])
                    zeros = torch.zeros_like(labelOriOnehot)
                    labelOriOnehot = torch.where(dist2centerTh[:, 1].unsqueeze(1) > chgDistMean / num1, zeros,
                                                 labelOriOnehot).float()#chg
                    labelOriOnehot = torch.where(dist2centerTh[:, 0].unsqueeze(1) > unchgDistMean * num2, zeros,
                                                 labelOriOnehot).float()  # [65536, 6]#unchg
                    # chgFeatFilterOneHot = torch.where(dist2centerTh[:, 1] > chgDistMean-num1*dist2centerchgvar, zeros, labels).float()
                    # unchgFeatFilterOneHot = torch.where(dist2centerTh[:, 0] > unchgDistMean +num2*dist2centerunchgvar, zeros,
                    #                                     1 - labels).float()

                # if True:
                #     labelOriOnehot = torch.zeros(labelsori.shape[0], self.unchgCenterNum + self.chgCenterNum).to(
                #         self.device).scatter_(1, labelsori.unsqueeze(1), 1)  # ([65536, 6])
                #     zeros = torch.zeros_like(labelOriOnehot)
                #     labelOriOnehot = torch.where(dist2centerTh[:, 1].unsqueeze(1) > chgDistMean / num1, zeros, labelOriOnehot).float()
                #     labelOriOnehot = torch.where(dist2centerTh[:, 0].unsqueeze(1) > unchgDistMean * num2, zeros,
                #                                   labelOriOnehot).float()#[65536, 6]
                # else:
                #     chgFeatFilterOneHot = torch.where(dist2centerTh[:,1] > chgDistMean, zeros, labels).float()
                #     unchgFeatFilterOneHot = torch.where(dist2centerTh[:,0] > unchgDistMean, zeros, 1-labels).float()
                # if unchgSelectFeat.shape[0]>labels_onehot.shape[0]/50 and chgSelectFeat.shape[0]>labels_onehot.shape[0]/50:
                if True:
                    FeatureTbFilter = FeatureTb.unsqueeze(2)*labelOriOnehot.unsqueeze(1)
                    Num = labelOriOnehot.sum(0)+1
                    FeatureTbFilter = FeatureTbFilter.sum(0)/Num.unsqueeze(0)#[32, 6]
                    # unchgSelectFeat
                    centersIter = FeatureTbFilter.permute(1,0)
                else:
                    centersIter = self.centers
                labels = labels.unsqueeze(0)#1, 65536
                if Ci==0:
                    labelPinit.append(labels)
                if Ci==0:
                    Cinidist = F.cosine_similarity(centersIter, self.centers)
                    CinidistW = Cinidist.unsqueeze(1).detach()
                else:
                    CinidistW = F.cosine_similarity(centersIter, self.centers).unsqueeze(1)
                # if Ci==0: Cinidist=torch.Tensor(0)
                Ci=Ci+1
        # print('centersIterout',centersIterout.shape)
        centersIterout=centersIterout/FeatureT.shape[0]#torch.Size([2, 16])
        labelsout=torch.cat(labelsout,dim=0)
        labelPinit=torch.cat(labelPinit,dim=0)
        labels_onehotout=torch.cat(labels_onehotout,dim=0)#[13, 65536, 2]
        dist2centerTori=torch.cat(dist2centerT,dim=0)#[13, 65536, 2]

        dist2centerT=(dist2centerTori-dist2centerTori.min(1)[0].unsqueeze(1))/(dist2centerTori.max(1)[0].unsqueeze(1)-dist2centerTori.min(1)[0].unsqueeze(1)+0.0000001)
        dist2centerT=(1-dist2centerT)

        Weight=dist2centerT
        # Weight=0
        Cinidist=Cinidist.sum()/FeatureT.shape[0]#total center distance
        # print('Cinidist',Cinidist.shape)
        # if True:#reshape
        #     labelsout=labelsout.reshape(B,1,W,H)
            # print('labelsout',labelsout.shape)

        return centersIterout.detach(),[labelsout,labels_onehotout,Weight,dist2centerTori,labelPinit],Cinidist.detach()
class CenterVal(nn.Module):
    def __init__(self, DEVICE, dist_type='cos'):
        super(CenterVal, self).__init__()
        self.Dist = DIST(dist_type)
        self.device = DEVICE
        self.num_classes = 2
        self.refs = (torch.LongTensor(range(self.num_classes)).unsqueeze(1)).to(self.device)

    def to_onehot(self, label, num_classes):
        identity = (torch.eye(num_classes)).to(self.device)
        onehot = torch.index_select(identity, 0, label)
        return onehot

    def assign_labels(self, feats, filter=False):  # 分别计算每一个目标域特征与质心之间的距离，并将最近距离的质心标签给目标域打上
        dists = self.Dist.get_dist(feats, self.centers, cross=True)
        _, labelsori = torch.min(dists, dim=1)
        if filter:
            zeros = torch.zeros_like(labelsori)
            ones = torch.ones_like(labelsori)
            labels = torch.where(labelsori > self.unchgCenterNum - 1, ones, zeros)

        return dists, labels, labelsori

    def forward(self, FeatureT, centerInit, unchgN=1, chgN=1):
        self.unchgCenterNum = unchgN
        self.chgCenterNum = chgN
        FeatureT = FeatureT.reshape(FeatureT.shape[0], FeatureT.shape[1], -1)
        labelsout = []
        labels_onehotout = []
        dist2centerT=[]
        self.centers=centerInit
        # labels = labels.unsqueeze(0)
        for b in range(FeatureT.shape[0]):
            FeatureTb = FeatureT[b, :, :].transpose(1, 0)  # [32, 65536]
            dist2center, labels, labelsori = self.assign_labels(FeatureTb,
                                                                filter=True)  # [65536, 2] [65536] [65536, 32]
            labelsout.append(labels.unsqueeze(0))

            dist2center = torch.cat([dist2center[:, 0:self.unchgCenterNum].mean(1).unsqueeze(1),
                                     dist2center[:, self.unchgCenterNum:].mean(1).unsqueeze(1)], dim=1)
            labels_onehot = self.to_onehot(labels, self.num_classes)  # [65536, 2]
            labels_onehotout.append(labels_onehot.unsqueeze(0))
            dist2centerT.append(dist2center.unsqueeze(0))
        labelsout = torch.cat(labelsout, dim=0)
        labels_onehotout = torch.cat(labels_onehotout, dim=0)  # [13, 65536, 2]
        dist2centerTori = torch.cat(dist2centerT, dim=0)  # [13, 65536, 2]
        return labelsout,labels_onehotout,dist2centerTori

# def Entropy(input_,n):
#     # bs = input_.size(0)
#     epsilon = 1e-5
#     # entropyunchg = -input_[:,0] * torch.log(input_[:,0] + epsilon)
#     # entropychg = -input_[:,1] * torch.log(input_[:,1] + epsilon)
#     chgP = input_[:, 1]
#     # print(input_.shape)
#     # entropy = entropyunchg+n*entropychg
#     entropy = -n*chgP * torch.log(chgP + epsilon)-(1-chgP) * torch.log(1-chgP + epsilon)
#
#
#     return entropy

def Entropy(input_,n):
    # bs = input_.size(0)
    epsilon = 1e-5
    # unchgP=input_[:,0]
    unchgP = input_[:, 0]
    chgP = input_[:, 1]

    # print(unchgP+chgP)
    # if n>0 and n<0.5:
    #     chgP=input_[:,1]+n
    #     ones=torch.ones_like(chgP).to(device)
    #     chgP = torch.where(chgP > 1, ones, chgP).float()
    #     # chgP = torch.where(chgP < 0.5, input_[:,1], chgP).float()
    # else:
    #     chgP = input_[:, 1]
    # entropyunchg =
    # entropychg = -n*chgP * torch.log(chgP + epsilon)-(1-chgP) * torch.log(1-chgP + epsilon)
    entropyunchg = -unchgP * torch.log(unchgP + epsilon)
    entropychg =  -chgP * torch.log(chgP + epsilon)

    return [entropyunchg,entropychg]

class CDModelutil():
    def __init__(self,opt):
        self.loss_filter = self.init_loss_filter(opt.use_ce_loss, opt.use_UnchgInCenterLoss,opt.use_UnchgNoCenterLoss)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.getCenterS=GetCenterS()
        # self.getCenterS2=GetCenterS2(self.device)
        self.getCenterS2=GetCenterS3
        self.getCenterSmulti=GetCenterS4
        self.CenterTOp=CenterTOp(self.device)
        self.CenterTOpEX=CenterTOpEXnew(self.device)
        self.CenterTOpEXmc=CenterTOpEXnewMultiC(self.device)
        self.getCenterNorm=GetCenterNormall()
        self.CenterVal=CenterVal(self.device)
        self.entropy=Entropy
        self.CEfc=cross_entropyfc
        if opt.use_ce_loss:
            self.loss = cross_entropy
        elif opt.use_hybrid_loss:
            self.loss = Hybrid(gamma=opt.gamma).to(self.device)
        elif opt.use_UnchgInCenterLoss:
            self.loss=UnchgInCenterLoss()
        elif opt.use_UnchgNoCenterLoss:
            self.loss=UnchgNoCenterLoss()
        elif opt.use_UnchgInCenterLossNew:

            self.loss = UnchgInCenterLossNew()
        else:
            raise NotImplementedError
        self.loss_names = self.loss_filter('CE', 'UnchgInCenterLoss', 'UnchgNoCenterLoss')

    def load_ckptGAN(self, save_path):
        # save_filename = 'epoch_%s.pth' % which_epoch
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            modelL_state_dict = None
            modelGAN_state_dict = None
            modelGAN2_state_dict = None
            optimizer_state_dict = None
            # raise ('%s must exist!' % (save_filename))
        else:
            checkpoint = torch.load(save_path,
                                    map_location=self.device)

            modelL_state_dict = checkpoint['network']
            modelGAN_state_dict = checkpoint['GAN']
            modelGAN2_state_dict = checkpoint['GAN2']
            optimizer_state_dict = checkpoint['optimizer']
        return modelL_state_dict, modelGAN_state_dict, modelGAN2_state_dict, optimizer_state_dict,
    def load_ckptGANCenter(self, save_path):
        # save_filename = 'epoch_%s.pth' % which_epoch

        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            modelL_state_dict = None
            modelGAN_state_dict = None
            modelGAN2_state_dict = None
            optimizer_state_dict = None
            Center = None
            unchgN = None
            chgN=None
            # raise ('%s must exist!' % (save_filename))
        else:
            checkpoint = torch.load(save_path,
                                    map_location=self.device)

            modelL_state_dict = checkpoint['network']
            modelGAN_state_dict = checkpoint['GAN']
            modelGAN2_state_dict = checkpoint['GAN2']
            # modelGAN3_state_dict = checkpoint['GANP']
            optimizer_state_dict = checkpoint['optimizer']
            Center=checkpoint['Center']
            unchgN=checkpoint['unchgN']
            chgN=checkpoint['chgN']
            # print('unchgN',unchgN,chgN,Center)
        return modelL_state_dict, modelGAN_state_dict, modelGAN2_state_dict, optimizer_state_dict,[Center,unchgN,chgN]
    def load_ckpt(self, save_path):
        # save_filename = 'epoch_%s.pth' % which_epoch
        # save_path = os.path.join(self.save_dir, save_filename)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            # raise ('%s must exist!' % (save_filename))
        else:
            checkpoint = torch.load(save_path,
                                    map_location=self.device)
            # print(checkpoint)
            # network.load_state_dict(checkpoint['network'], False)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            modelL_state_dict= checkpoint['network']
            optimizer_state_dict = checkpoint['optimizer']
        return modelL_state_dict,None,optimizer_state_dict
    def load_dackpt(self, save_path):
        # save_filename = 'epoch_%s.pth' % which_epoch
        # save_path = os.path.join(self.save_dir, save_filename)
        # if not os.path.isfile(save_path):
        #     print('%s not exists yet!' % save_path)
        #     # raise ('%s must exist!' % (save_filename))
        # else:
        assert (os.path.isfile(save_path)), \
                '%s not exists yet!' % save_path

        param_dict = torch.load(save_path,map_location=self.device)
        # print(checkpoint)
        optimizer_state_dict = param_dict['optimizer']
        # cfg.DA.BN_DOMAIN_MAP = {cfg.TRAINLOG.DATA_NAMES[cfg.DA.S]: 0, cfg.TRAINLOG.DATA_NAMES[cfg.DA.T]: 1}
        # model_state_dict = param_dict['model_state_dict']
        modelAL_state_dict = param_dict['modelAL_state_dict']
        modelAH_state_dict = param_dict['modelAH_state_dict']

        modelB_state_dict = param_dict['modelB_state_dict']
        modelC_state_dict = param_dict['modelC_state_dict']
        bn_domain_map = param_dict['bn_domain_map']
                # cfg.DA.BN_DOMAIN_MAP = {cfg.TRAINLOG.DATA_NAMES[cfg.DA.S]: 0, cfg.TRAINLOG.DATA_NAMES[cfg.DA.T]: 1}
        return [modelAL_state_dict,modelAH_state_dict,modelB_state_dict,modelC_state_dict],bn_domain_map,optimizer_state_dict




    def load_lowPretrain(self, save_path):
        assert (os.path.isfile(save_path)), \
                '%s not exists yet!' % save_path

        param_dict = torch.load(save_path,map_location=self.device)
        # print(checkpoint)
        optimizer_state_dict = param_dict['optimizer']
        # cfg.DA.BN_DOMAIN_MAP = {cfg.TRAINLOG.DATA_NAMES[cfg.DA.S]: 0, cfg.TRAINLOG.DATA_NAMES[cfg.DA.T]: 1}
        modelAL_state_dict = param_dict['modelAL_state_dict']
        modelAH_state_dict = param_dict['modelAH_state_dict']

        modelB_state_dict = param_dict['modelB_state_dict']
        modelC_state_dict = param_dict['modelC_state_dict']
        bn_domain_map = param_dict['bn_domain_map']
                # cfg.DA.BN_DOMAIN_MAP = {cfg.TRAINLOG.DATA_NAMES[cfg.DA.S]: 0, cfg.TRAINLOG.DATA_NAMES[cfg.DA.T]: 1}
        return [modelAL_state_dict,modelAH_state_dict,modelB_state_dict,modelC_state_dict],bn_domain_map,optimizer_state_dict
    def init_loss_filter(self, use_ce_loss, UnchgInCenterLoss,UnchgNoCenterLoss):
        flags = (use_ce_loss, UnchgInCenterLoss, UnchgNoCenterLoss)#损失函数

        def loss_filter(ce, focal, dice):
            return [l for (l, f) in zip((ce, focal, dice), flags) if f]

        return loss_filter
    def init_method(self,net, init_type='normal'):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'resnet'):
                pass
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, 0.02)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=0.02)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=0.02)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)

    def save_ckpt(self, network, optimizer, save_str):
        # save_filename = 'epoch_%s.pth' % which_epoch
        # save_path = os.path.join(self.save_dir, save_filename)
        save_path = save_str
        torch.save({
            'network': network.cpu().state_dict(),
            'optimizer': optimizer.state_dict()},
            save_path)
        if torch.cuda.is_available():
            network.cuda()
    def save_ckptGAN(self, network, optimizer, save_str):
        # save_filename = 'epoch_%s.pth' % which_epoch
        # save_path = os.path.join(self.save_dir, save_filename)
        save_path = save_str
        torch.save({
            'network': network[0].cpu().state_dict(),
            'GAN':network[1].cpu().state_dict(),
            'GAN2': network[2].cpu().state_dict(),
            'optimizer': optimizer.state_dict()},
            save_path)
        if torch.cuda.is_available():
            network[0].cuda()
            network[1].cuda()
            network[2].cuda()
    def save_ckptGANCenter(self, network,Center, optimizer, save_str):
        # save_filename = 'epoch_%s.pth' % which_epoch
        # save_path = os.path.join(self.save_dir, save_filename)
        save_path = save_str
        # if network[1] is None:
        #     torch.save({
        #         'network': network[0].cpu().state_dict(),
        #         'GAN': None,
        #         'GAN2': None,
        #         'optimizer': optimizer.state_dict(),
        #         'Center': Center[0],
        #         'unchgN': Center[1],
        #         'chgN': Center[2]},
        #         save_path)
        #     if torch.cuda.is_available():
        #         network[0].cuda()
        # else:
        #     torch.save({
        #         'network': network[0].cpu().state_dict() if network[0] is not None else None,
        #         'GAN': network[1].cpu().state_dict() if network[1] is not None else None,
        #         # 'GAN2': network[1].cpu().state_dict() if network[2] is not None else None,
        #         'optimizer': optimizer.state_dict() if optimizer is not None else None,
        #         'Center': Center[0] if Center[0] is not None else None,
        #         'unchgN': Center[1] if Center[1] is not None else None,
        #         'chgN': Center[2] if Center[2] is not None else None,},
        #         save_path)
        #
        #     if torch.cuda.is_available():
        #         if network[0] is not None: network[0].cuda()
        #         if network[1] is not None: network[1].cuda()
        #         if network[2] is not None: network[2].cuda()
        torch.save({
            'network': network[0].cpu().state_dict() if network[0] is not None else None,
            'GAN': network[1].cpu().state_dict() if network[1] is not None else None,
            'GAN2': network[2].cpu().state_dict() if network[2] is not None else None,
            # 'GANP': network[3].cpu().state_dict() if network[3] is not None else None,

            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'Center': Center[0] if Center[0] is not None else None,
            'unchgN': Center[1] if Center[1] is not None else None,
            'chgN': Center[2] if Center[2] is not None else None, },
            save_path)

        if torch.cuda.is_available():
            if network[0] is not None: network[0].cuda()
            if network[1] is not None: network[1].cuda()
            if network[2] is not None: network[2].cuda()
            # if network[3] is not None: network[3].cuda()

    def save_ckptDA(self, iters,network, bn_domain_map,optimizer, save_str,):
        save_path = save_str
        torch.save({'iters': iters,
                    'model_state_dict': network.cpu().state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'bn_domain_map': bn_domain_map
                    }, save_path)
        if torch.cuda.is_available():
            network.cuda()
    def save_ckptDApre(self, iters,network, bn_domain_map,optimizer, save_str,):
        # save_filename = 'epoch_%s.pth' % which_epoch
        # save_path = os.path.join(self.save_dir, save_filename)
        save_path = save_str
        # torch.save({
        #     'network': network.cpu().state_dict(),
        #     'optimizer': optimizer.state_dict()},
        #     save_path)
        # print('optimizer.state_dict()',optimizer.state_dict())
        torch.save({'iters': iters,
                    'modelAL_state_dict': network[0].cpu().state_dict(),
                    'modelAH_state_dict': network[1].cpu().state_dict(),
                    'modelB_state_dict': network[2].cpu().state_dict(),
                    'modelC_state_dict': network[3].cpu().state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'bn_domain_map': bn_domain_map
                    }, save_path)
        if torch.cuda.is_available():
            for net in network:
                net.cuda()
class CDModel(nn.Module):
    def name(self):
         return 'CDModel'


    def init_loss_filter(self, use_ce_loss, use_hybrid_loss):
        flags = (use_ce_loss, use_hybrid_loss, use_hybrid_loss)#损失函数

        def loss_filter(ce, focal, dice):
            return [l for (l, f) in zip((ce, focal, dice), flags) if f]

        return loss_filter


    def initialize(self, opt):
        torch.backends.cudnn.benchmark = True
        self.device = torch.device("cuda:%s" % opt.gpu_ids[0] if torch.cuda.is_available() and len(opt.gpu_ids) > 0
                                   else "cpu")
        self.Tensor = torch.cuda.FloatTensor if self.device else torch.Tensor
        self.num_class = opt.num_class
        self.opt = opt
        self.old_lr = opt.lr
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.name, 'trained_models')
        util.mkdirs([self.save_dir])
        print('opt.gpu_idsopt.gpu_ids',opt.gpu_ids)
        # define model
        self.model = define_model(model_type=opt.model_type, resnet=opt.resnet, init_type=opt.init_type,
                                  initialize=opt.initialize, gpu_ids=opt.gpu_ids)

        # define optimizers
        if opt.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=opt.lr,
                                       momentum=0.9,
                                       weight_decay=5e-4)
        elif opt.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr,
                                        betas=(0.5, 0.999))
        else:
            raise NotImplemented(opt.optimizer)


            # load models
        if opt.load_pretrain:
            self.load_ckpt(self.model, self.optimizer, opt.which_epoch)

        print('---------- Networks initialized -------------')

        # define loss functions
        self.loss_filter = self.init_loss_filter(opt.use_ce_loss, opt.use_hybrid_loss)
        if opt.use_ce_loss:
            self.loss = cross_entropy
        elif opt.use_hybrid_loss:
            self.loss = Hybrid(gamma=opt.gamma).to(self.device)
        else:
            raise NotImplementedError
        self.loss_names = self.loss_filter('CE', 'Focal', 'Dice')

    def forward(self, t1_input, t2_input, label,val=False):
        if val:
            self.model.eval()
            with torch.no_grad():
                pred = self.model.forward(t1_input, t2_input)
                print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
                time.sleep(4)
                print('cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
                # print('nonoo')
                if self.opt.use_ce_loss:
                    ce_loss = self.loss(pred[0], label.long())
                    focal_loss = 0
                    dice_loss = 0
                elif self.opt.use_hybrid_loss:
                    ce_loss = 0
                    focal_loss, dice_loss = self.loss(pred[0], label.long())
                else:
                    raise NotImplementedError
        else:
            pred = self.model(t1_input, t2_input)
            # print('aa')
            # time.sleep(5)
            # print('bb')
            if self.opt.use_ce_loss:
                ce_loss = self.loss(pred[0], label.long())
                focal_loss = 0
                dice_loss = 0
            elif self.opt.use_hybrid_loss:
                ce_loss = 0
                focal_loss, dice_loss = self.loss(pred[0], label.long())
            else:
                raise NotImplementedError

        return [self.loss_filter(ce_loss, focal_loss, dice_loss), pred]


    def inference(self, t1_input, t2_input):
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                pred = self.model.forward(t1_input, t2_input)
                print('ddddddddddddddddddddddssssssssss')
        else:
            pred = self.model.forward(t1_input, t2_input)
            print('dddddssssssdddddddddddddddddssdddddddddddddddddssssssssss')

        return pred


    def load_ckpt(self, network, optimizer, which_epoch):
        save_filename = 'epoch_%s.pth' % which_epoch
        save_path = os.path.join(self.save_dir, save_filename)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            raise ('%s must exist!' % (save_filename))
        else:
            checkpoint = torch.load(save_path,
                                    map_location=self.device)
            network.load_state_dict(checkpoint['network'], False)
            # optimizer.load_state_dict(checkpoint['optimizer'], False)


    def save_ckpt(self, network, optimizer, save_str):
        # save_filename = 'epoch_%s.pth' % which_epoch
        # save_path = os.path.join(self.save_dir, save_filename)
        save_path=save_str
        torch.save({
                    'network': network.cpu().state_dict(),
                    'optimizer': optimizer.state_dict()},
            save_path)
        if torch.cuda.is_available():
            network.cuda()


    def save(self, save_str):
        self.save_ckpt(self.model, self.optimizer, save_str)


    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.num_decay_epochs
        lr = self.old_lr - lrd

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print('update learning rate to %f' % lr)
        self.old_lr = lr


class InferenceModel(CDModel):
    def forward(self, t1_input, t2_input):
        return self.inference(t1_input, t2_input)


def create_model(opt):
    model = CDModel()
    model.initialize(opt)

    print("model [%s] was created" % (model.name()))

    if len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
