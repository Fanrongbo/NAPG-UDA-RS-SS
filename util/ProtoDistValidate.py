import optparse

import torch
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
from torch.nn import functional as F
# import math
from model.loss import cross_entropy
from util.metric_toolSeg import MetricsTracker
from util.visualizer import Visualizer
import time
from util.drawTool import setFigurevalSeg
from util.ValidateVisualization_tool import *

class VailidateMax():
    def __init__(self, opt, device=None,Osize=32,n_cluster=1,CatFlag=False):
        self.opt = opt
        self.DEVICE = device
        # self.dataloader = dataloader
        self.Osize = Osize
        self.running_metric = MetricsTracker(num_classes=6)
        self.visualizer = Visualizer(opt)
        self.n_cluster=n_cluster
        self.metrics = setFigurevalSeg(num_class=6)
        self.CatFlag=CatFlag
        self.figure_metrics = self.metrics.initialize_figure()
    def main(self, model, figure_metrics=None, epoch=1, cfg=None, iterFlag=False, dataloader=None, currentOpt=None,ProtoInput=None,classifierFlag=False):
        val_dataload = dataloader.load_data()
        iter_val = iter(val_dataload)
        val_data_len = len(val_dataload)
        epoch_start_time = time.time()
        # if epoch != start_epoch:
        #     epoch_iter = epoch_iter % len(train_loader)
        running_metric=self.running_metric
        running_metric.clear()
        model.eval()
        tbar = tqdm(range(val_data_len - 1))
        # train_data_len = len_source_loader
        record = {'LossT': 0, 'SCET': 0, 'DGANT': 0, 'TCET': 0}
        if currentOpt.phase == 'target':
            dataIndex = currentOpt.t
            excelName = 'wsT-' + cfg.TRAINLOG.DATA_NAMES[dataIndex][0]+'-Max'
        elif currentOpt.phase == 'val':
            dataIndex = currentOpt.s
            excelName = 'wsVal'+'-Max'
        with torch.no_grad():
            for i in tbar:
                data_val = next(iter_val)
                # epoch_iter += currentOpt.batch_size
                ############## Forward Pass ######################
                data_img_val = Variable(data_val['img_full']).to(self.DEVICE)
                label = Variable(data_val['label_full']).to(self.DEVICE)
                seg_val_pred, _, Feat = model.forward(data_img_val,ProtoInput=ProtoInput)
                # Generation: source
                label = F.interpolate(label.unsqueeze(1).float(), size=(self.Osize, self.Osize), mode='nearest')
                label = label.squeeze(1)  # 尺寸变为 [14, 512, 512]
                # labelUp = F.interpolate(label.unsqueeze(1).float(), size=(self.Osize, self.Osize), mode='nearest')
                # labelUp = labelUp.squeeze(1)  # 尺寸变为 [14, 512, 512]
                if classifierFlag:
                    FeatSim=Feat['cOut']
                    val_pred = torch.argmax(FeatSim.detach(), dim=1)
                    # print('val_pred',val_pred.max(),label.shape,label.max())
                else:
                    FeatSim=Feat['cat']
                    SSoft = F.softmax(FeatSim, dim=1)
                    SSoft = SSoft.view(SSoft.size(0), SSoft.size(1) // (self.n_cluster), (self.n_cluster),
                                       SSoft.size(2),
                                       SSoft.size(3))
                    SSoft = SSoft.max(dim=2)[0]

                    val_pred = torch.argmax(SSoft.detach(), dim=1)
                    # FeatSim=Feat['cat']
                    # val_pred = torch.argmax(FeatSim.detach(), dim=1) // (FeatSim.size(1) // 6)
                SSoft=F.interpolate(SSoft,size=(seg_val_pred['outUp'].size(2),seg_val_pred['outUp'].size(3)))
                # print(SSoft.shape,seg_val_pred['outUp'].shape)

                maxP=SSoft*self.n_cluster+seg_val_pred['outUp']
                val_predP = torch.argmax(maxP.detach(), dim=1)


                val_pred = torch.argmax(maxP.detach(), dim=1)

                # SCELoss = cross_entropy(FeatSim, label.long())
                SCELoss = cross_entropy(seg_val_pred['outUp'], label.long())

                TCELoss = torch.tensor([0.0]).to(self.DEVICE)
                record['SCET'] += SCELoss.item()

                current_scoreval = running_metric.update(val_pred, label)
                #############################Discrimination##################################
                ####Background
                record['TCET'] += TCELoss.item()
                lossT = TCELoss.item() + SCELoss.item()

                valScore = {'LossT': lossT, 'SCET': SCELoss.item(), 'TCET': TCELoss.item()}
                record['LossT'] += lossT
                valScore.update(current_scoreval)
                valMessage = self.visualizer.print_current_sorces_Seg(currentOpt.phase+'-ProMix', epoch, i, val_data_len, valScore)
                tbar.set_description(valMessage)
                if i > 10 and iterFlag:
                    break
        lossAvg = record['LossT'] / i
        CES_lossAvg = record['SCET'] / i
        DGAN_LossAvg = record['DGANT'] / i
        CET_lossAvg = record['TCET'] / i
        val_scores = running_metric.calculate_metrics()
        IterValScore = {'LossT': lossAvg, 'SCE': CES_lossAvg, 'DGAN': DGAN_LossAvg, 'TCE': CET_lossAvg}
        IterValScore.update(val_scores)
        message = self.visualizer.print_sorces_Seg(currentOpt.phase+'-ProMix', epoch, IterValScore)
        # messageT, core_dictT = running_metric.get_scoresT()
        cfg.TRAINLOG.LOGTXT.write(message + '\n')
        self.val_scores=val_scores
        exel_out = currentOpt.phase+'-ProSim-'+ cfg.TRAINLOG.DATA_NAMES[dataIndex][0] , epoch, lossAvg, CES_lossAvg, DGAN_LossAvg, CET_lossAvg, \
                   IterValScore['acc'].item(), IterValScore['mIoU'].item(), IterValScore['mf1'].item(), \
                   IterValScore['preacc'][0].item(), IterValScore['preacc'][1].item(), IterValScore['preacc'][2].item(), \
                   IterValScore['preacc'][3].item(), IterValScore['preacc'][4].item(), IterValScore['preacc'][5].item(), \
                   IterValScore['preIou'][0].item(), IterValScore['preIou'][1].item(), IterValScore['preIou'][2].item(), \
                   IterValScore['preIou'][3].item(), IterValScore['preIou'][4].item(), IterValScore['preIou'][5].item(), \
            # 0,0,0,0
        # core_dictT['accT'], core_dictT['unchgT'], core_dictT['chgT'], core_dictT['mF1T']
        cfg.TRAINLOG.EXCEL_LOG[excelName].append(exel_out)
        # cfg.TRAINLOG.EXCEL_LOG.save('./log/%s/%s/log.xlsx' % (name, time_now))
        figure_metrics = self.metrics.set_figure(metric_dict=figure_metrics,
                                                    acc=IterValScore['acc'].item(),
                                                    miou=IterValScore['mIoU'],
                                                    mf1=IterValScore['mf1'], preacc=IterValScore['preacc'],
                                                    premiou=IterValScore['preIou'],
                                                    CES_lossAvg=CES_lossAvg
                                                    )
        return figure_metrics

class VailidateMax2():
    def __init__(self, opt, device=None,Osize=32,n_cluster=1,CatFlag=False):
        self.opt = opt
        self.DEVICE = device
        # self.dataloader = dataloader
        self.Osize = Osize
        self.running_metric = MetricsTracker(num_classes=6)
        self.visualizer = Visualizer(opt)
        self.n_cluster=n_cluster
        self.metrics = setFigurevalSeg(num_class=6)
        self.CatFlag=CatFlag
        self.figure_metrics = self.metrics.initialize_figure()
    def main(self, model, figure_metrics=None, epoch=1, cfg=None, iterFlag=False, dataloader=None, currentOpt=None,ProtoInput=None,classifierFlag=False):
        val_dataload = dataloader.load_data()
        iter_val = iter(val_dataload)
        val_data_len = len(val_dataload)
        epoch_start_time = time.time()
        # if epoch != start_epoch:
        #     epoch_iter = epoch_iter % len(train_loader)
        running_metric=self.running_metric
        running_metric.clear()
        model.eval()
        tbar = tqdm(range(val_data_len - 1))
        # train_data_len = len_source_loader
        record = {'LossT': 0, 'SCET': 0, 'DGANT': 0, 'TCET': 0}
        if currentOpt.phase == 'target':
            dataIndex = currentOpt.t
            excelName = 'wsT-' + cfg.TRAINLOG.DATA_NAMES[dataIndex][0]+'-Max'
        elif currentOpt.phase == 'val':
            dataIndex = currentOpt.s
            excelName = 'wsVal'+'-Max'
        with torch.no_grad():
            for i in tbar:
                data_val = next(iter_val)
                # epoch_iter += currentOpt.batch_size
                ############## Forward Pass ######################
                data_img_val = Variable(data_val['img_full']).to(self.DEVICE)
                label = Variable(data_val['label_full']).to(self.DEVICE)
                seg_val_pred, _, Feat = model.forward(data_img_val,ProtoInput=ProtoInput)
                # Generation: source
                label = F.interpolate(label.unsqueeze(1).float(), size=(self.Osize, self.Osize), mode='nearest')
                label = label.squeeze(1)  # 尺寸变为 [14, 512, 512]
                # labelUp = F.interpolate(label.unsqueeze(1).float(), size=(self.Osize, self.Osize), mode='nearest')
                # labelUp = labelUp.squeeze(1)  # 尺寸变为 [14, 512, 512]
                if classifierFlag:
                    FeatSim=Feat['cOut']
                    val_pred = torch.argmax(FeatSim.detach(), dim=1)
                    # print('val_pred',val_pred.max(),label.shape,label.max())
                else:
                    FeatSim=Feat['cat']
                    SSoft = F.softmax(FeatSim, dim=1)
                    SSoft = SSoft.view(SSoft.size(0), SSoft.size(1) // (self.n_cluster), (self.n_cluster),
                                       SSoft.size(2),
                                       SSoft.size(3))
                    SSoft = SSoft.max(dim=2)[0]

                    val_pred = torch.argmax(SSoft.detach(), dim=1)
                    # FeatSim=Feat['cat']
                    # val_pred = torch.argmax(FeatSim.detach(), dim=1) // (FeatSim.size(1) // 6)

                SSoft=F.interpolate(SSoft,size=(seg_val_pred['outUp'].size(2),seg_val_pred['outUp'].size(3)))
                # print(SSoft.shape,seg_val_pred['outUp'].shape)

                maxP=SSoft*2+seg_val_pred['outUp']
                val_pred = torch.argmax(maxP.detach(), dim=1)
                val_predTotal=torch.zeros_like(val_pred)
                val_predP = torch.argmax(SSoft.detach(), dim=1)
                val_predClass = torch.argmax(seg_val_pred['outUp'].detach(), dim=1)

                mask = (val_predP == 0) | (val_predP == 2)
                result_label = val_predClass.clone()
                result_label[mask] = val_predP[mask]

                # val_predTotal[val_predP==0]=val_predP[val_predP==0]
                # val_predTotal[val_predP==2]=val_predP[val_predP==0]
                # val_predTotal[val_predClass==1]=val_predClass[val_predClass==1]
                # val_predTotal[val_predClass==3]=val_predClass[val_predClass==3]
                # val_predTotal[val_predClass==4]=val_predClass[val_predClass==4]
                # val_predTotal[val_predClass==5]=val_predClass[val_predClass==5]

                # SCELoss = cross_entropy(FeatSim, label.long())
                SCELoss = cross_entropy(seg_val_pred['outUp'], label.long())

                TCELoss = torch.tensor([0.0]).to(self.DEVICE)
                record['SCET'] += SCELoss.item()

                current_scoreval = running_metric.update(result_label, label)
                #############################Discrimination##################################
                ####Background
                record['TCET'] += TCELoss.item()
                lossT = TCELoss.item() + SCELoss.item()

                valScore = {'LossT': lossT, 'SCET': SCELoss.item(), 'TCET': TCELoss.item()}
                record['LossT'] += lossT
                valScore.update(current_scoreval)
                valMessage = self.visualizer.print_current_sorces_Seg(currentOpt.phase+'-ProMix', epoch, i, val_data_len, valScore)
                tbar.set_description(valMessage)
                if i > 10 and iterFlag:
                    break
        lossAvg = record['LossT'] / i
        CES_lossAvg = record['SCET'] / i
        DGAN_LossAvg = record['DGANT'] / i
        CET_lossAvg = record['TCET'] / i
        val_scores = running_metric.calculate_metrics()
        IterValScore = {'LossT': lossAvg, 'SCE': CES_lossAvg, 'DGAN': DGAN_LossAvg, 'TCE': CET_lossAvg}
        IterValScore.update(val_scores)
        message = self.visualizer.print_sorces_Seg(currentOpt.phase+'-ProMix', epoch, IterValScore)
        # messageT, core_dictT = running_metric.get_scoresT()
        cfg.TRAINLOG.LOGTXT.write(message + '\n')
        self.val_scores=val_scores
        exel_out = currentOpt.phase+'-ProSim-'+ cfg.TRAINLOG.DATA_NAMES[dataIndex][0] , epoch, lossAvg, CES_lossAvg, DGAN_LossAvg, CET_lossAvg, \
                   IterValScore['acc'].item(), IterValScore['mIoU'].item(), IterValScore['mf1'].item(), \
                   IterValScore['preacc'][0].item(), IterValScore['preacc'][1].item(), IterValScore['preacc'][2].item(), \
                   IterValScore['preacc'][3].item(), IterValScore['preacc'][4].item(), IterValScore['preacc'][5].item(), \
                   IterValScore['preIou'][0].item(), IterValScore['preIou'][1].item(), IterValScore['preIou'][2].item(), \
                   IterValScore['preIou'][3].item(), IterValScore['preIou'][4].item(), IterValScore['preIou'][5].item(), \
            # 0,0,0,0
        # core_dictT['accT'], core_dictT['unchgT'], core_dictT['chgT'], core_dictT['mF1T']
        cfg.TRAINLOG.EXCEL_LOG[excelName].append(exel_out)
        # cfg.TRAINLOG.EXCEL_LOG.save('./log/%s/%s/log.xlsx' % (name, time_now))
        figure_metrics = self.metrics.set_figure(metric_dict=figure_metrics,
                                                    acc=IterValScore['acc'].item(),
                                                    miou=IterValScore['mIoU'],
                                                    mf1=IterValScore['mf1'], preacc=IterValScore['preacc'],
                                                    premiou=IterValScore['preIou'],
                                                    CES_lossAvg=CES_lossAvg
                                                    )
        return figure_metrics
class Vailidate():
    def __init__(self, opt, device=None,Osize=32):
        self.opt = opt
        self.DEVICE = device
        # self.dataloader = dataloader
        self.Osize = Osize
        self.running_metric = MetricsTracker(num_classes=6)
        self.visualizer = Visualizer(opt)

        self.metrics = setFigurevalSeg(num_class=6)
        self.figure_metrics = self.metrics.initialize_figure()
    def main(self, model, figure_metrics=None, epoch=1, cfg=None, iterFlag=False, dataloader=None, currentOpt=None,ProtoInput=None):
        val_dataload = dataloader.load_data()
        iter_val = iter(val_dataload)
        val_data_len = len(val_dataload)
        running_metric=self.running_metric
        running_metric.clear()
        model.eval()
        tbar = tqdm(range(val_data_len - 1))
        # train_data_len = len_source_loader
        record = {'LossT': 0, 'SCET': 0, 'DGANT': 0, 'TCET': 0}
        if currentOpt.phase == 'target':
            dataIndex = currentOpt.t
            excelName = 'wsT-' + cfg.TRAINLOG.DATA_NAMES[dataIndex][0]
        elif currentOpt.phase == 'val':
            dataIndex = currentOpt.s
            excelName = 'wsVal'
        with torch.no_grad():
            for i in tbar:
                data_val = next(iter_val)
                # epoch_iter += currentOpt.batch_size
                ############## Forward Pass ######################
                data_img_val = Variable(data_val['img_full']).to(self.DEVICE)
                label = Variable(data_val['label_full']).to(self.DEVICE)
                seg_val_pred, _, _ = model.forward(data_img_val, DomainLabel=0,ProtoInput=ProtoInput)

                # Generation: source
                label = F.interpolate(label.unsqueeze(1).float(), size=(self.Osize, self.Osize), mode='nearest')
                label = label.squeeze(1)  # 尺寸变为 [14, 512, 512]
                SCELoss = cross_entropy(seg_val_pred['outUp'], label.long())
                TCELoss = torch.tensor([0.0]).to(self.DEVICE)
                record['SCET'] += SCELoss.item()
                # predict
                val_pred = torch.argmax(seg_val_pred['outUp'].detach(), dim=1)
                # print('val_pred',val_pred.shape)
                current_scoreval = running_metric.update(val_pred, label)
                #############################Discrimination##################################
                ####Background
                record['TCET'] += TCELoss.item()
                lossT = TCELoss.item() + SCELoss.item()

                valScore = {'LossT': lossT, 'SCET': SCELoss.item(), 'TCET': TCELoss.item()}
                record['LossT'] += lossT
                valScore.update(current_scoreval)
                valMessage = self.visualizer.print_current_sorces_Seg(currentOpt.phase, epoch, i, val_data_len, valScore)
                tbar.set_description(valMessage)
                if i > 10 and iterFlag:
                    break
        lossAvg = record['LossT'] / i
        CES_lossAvg = record['SCET'] / i
        DGAN_LossAvg = record['DGANT'] / i
        CET_lossAvg = record['TCET'] / i
        val_scores = running_metric.calculate_metrics()
        IterValScore = {'LossT': lossAvg, 'SCE': CES_lossAvg, 'DGAN': DGAN_LossAvg, 'TCE': CET_lossAvg}
        IterValScore.update(val_scores)
        message = self.visualizer.print_sorces_Seg(currentOpt.phase, epoch, IterValScore)
        # messageT, core_dictT = running_metric.get_scoresT()
        cfg.TRAINLOG.LOGTXT.write(message + '\n')
        self.val_scores=val_scores
        exel_out = currentOpt.phase+'-'+ cfg.TRAINLOG.DATA_NAMES[dataIndex][0] , epoch, lossAvg, CES_lossAvg, DGAN_LossAvg, CET_lossAvg, \
                   IterValScore['acc'].item(), IterValScore['mIoU'].item(), IterValScore['mf1'].item(), \
                   IterValScore['preacc'][0].item(), IterValScore['preacc'][1].item(), IterValScore['preacc'][2].item(), \
                   IterValScore['preacc'][3].item(), IterValScore['preacc'][4].item(), IterValScore['preacc'][5].item(), \
                   IterValScore['preIou'][0].item(), IterValScore['preIou'][1].item(), IterValScore['preIou'][2].item(), \
                   IterValScore['preIou'][3].item(), IterValScore['preIou'][4].item(), IterValScore['preIou'][5].item(), \
            # 0,0,0,0
        # core_dictT['accT'], core_dictT['unchgT'], core_dictT['chgT'], core_dictT['mF1T']
        cfg.TRAINLOG.EXCEL_LOG[excelName].append(exel_out)
        # cfg.TRAINLOG.EXCEL_LOG.save('./log/%s/%s/log.xlsx' % (name, time_now))
        figure_metrics = self.metrics.set_figure(metric_dict=figure_metrics,
                                                    acc=IterValScore['acc'].item(),
                                                    miou=IterValScore['mIoU'],
                                                    mf1=IterValScore['mf1'], preacc=IterValScore['preacc'],
                                                    premiou=IterValScore['preIou'],
                                                    CES_lossAvg=CES_lossAvg,
                                                    )
        return figure_metrics


def classify_out(out, boundaries):
    # Calculate the cumulative sums of the boundaries to get the ranges
    thresholds_tensor = torch.tensor(boundaries, dtype=out.dtype, device=out.device)
    ranges = torch.cumsum(thresholds_tensor, dim=0)

    # Initialize a tensor to store the result with the same shape as `out`
    result = torch.zeros_like(out, dtype=torch.long)

    # Compare `out` with the boundaries to classify the elements
    for i in range(len(boundaries)):
        if i == 0:
            result[out < ranges[i]] = i
        else:
            result[(out >= ranges[i - 1]) & (out < ranges[i])] = i

    return result

def multiFeatClaSum(Featin, values,M='sum'):
    Feat=torch.zeros((Featin.size(0),6,Featin.size(2),Featin.size(3)),device=Featin.device)
    cumulative_sum = 0  # 初始化累加和
    if len(values)!=6:
        return None
    for index, value in enumerate(values):
        # 更新累加和
        if value == 0:
            return None
        if M == 'max':
            Feat[:, index] = Featin[:, cumulative_sum:cumulative_sum+value].max(dim=1)[0]
        elif M == 'sum':
            Feat[:, index] = Featin[:, cumulative_sum:cumulative_sum+value].sum(dim=1)
        elif M == 'mean':
            Feat[:, index] = Featin[:, cumulative_sum:cumulative_sum+value].mean(dim=1)
        # cumulative_sums = cumulative_sums +Featin[:,cumulative_sum:cumulative_sum+value].size(1)
        cumulative_sum += value
        # 检查out是否在当前累加的区间内
    # print('cumulative_sums',cumulative_sums)
    return Feat  # 返回当前区间的索引
class VailidateFusion():
    def __init__(self, opt, device=None,Osize=32,n_cluster=1,CatFlag=False):
        self.opt = opt
        self.DEVICE = device
        # self.dataloader = dataloader
        self.Osize = Osize
        self.running_metric = MetricsTracker(num_classes=6)
        self.visualizer = Visualizer(opt)
        self.n_cluster=n_cluster
        self.metrics = setFigurevalSeg(num_class=6)
        self.CatFlag=CatFlag
        self.figure_metrics = self.metrics.initialize_figure()
    def main(self, model, figure_metrics=None, epoch=1, cfg=None, iterFlag=False, dataloader=None, currentOpt=None,ProtoInput=None,classifierFlag=False):
        val_dataload = dataloader.load_data()
        iter_val = iter(val_dataload)
        val_data_len = len(val_dataload)
        epoch_start_time = time.time()
        # if epoch != start_epoch:
        #     epoch_iter = epoch_iter % len(train_loader)
        running_metric=self.running_metric
        running_metric.clear()
        model.eval()
        tbar = tqdm(range(val_data_len - 1))
        # train_data_len = len_source_loader
        record = {'LossT': 0, 'SCET': 0, 'DGANT': 0, 'TCET': 0}
        if currentOpt.phase == 'target':
            dataIndex = currentOpt.t
            excelName = 'wsT-' + cfg.TRAINLOG.DATA_NAMES[dataIndex][0]+'-Fusion'
        elif currentOpt.phase == 'val':
            dataIndex = currentOpt.s
            excelName = 'wsVal'+'-Fusion'
        with torch.no_grad():
            for i in tbar:
                data_val = next(iter_val)
                # epoch_iter += currentOpt.batch_size
                ############## Forward Pass ######################
                data_img_val = Variable(data_val['img_full']).to(self.DEVICE)
                label = Variable(data_val['label_full']).to(self.DEVICE)
                seg_val_pred, _, Feat = model.forward(data_img_val,ProtoInput=ProtoInput)
                # Generation: source
                label = F.interpolate(label.unsqueeze(1).float(), size=(self.Osize, self.Osize), mode='nearest')
                label = label.squeeze(1)  # 尺寸变为 [14, 512, 512]
                # if classifierFlag:
                #     FeatSim=Feat['cOut']
                #     val_pred = torch.argmax(FeatSim.detach(), dim=1)
                #     # print('val_pred',val_pred.max(),label.shape,label.max())
                # else:
                #     FeatSim=Feat['cat']
                #     val_pred = torch.argmax(FeatSim.detach(), dim=1) // (FeatSim.size(1) // 6)
                SSoft = F.softmax(Feat['cat'], dim=1)
                # SCELossP = cross_entropy(FeatS['cat'], labelsP.long())
                SSoft = SSoft.view(SSoftLog.size(0), SSoftLog.size(1) // (self.n_cluster), (self.n_cluster),
                                         SSoftLog.size(2),
                                         SSoftLog.size(3))
                SSoftLog = SSoft.max(dim=2)[0]
                CSoft = F.softmax(Feat['outUp'], dim=1)

                val_pred = torch.argmax((SSoft+CSoft).detach(), dim=1)

                SCELoss = cross_entropy(Feat['cat'], label.long())
                TCELoss = torch.tensor([0.0]).to(self.DEVICE)
                record['SCET'] += SCELoss.item()
                # predict
                # print(Feat['cat'].shape)
                # print(Feat['cOut'].shape)
                # val_pred = torch.argmax(Feat['cOut'].detach(), dim=1)
                # if ProtoInput == None or self.CatFlag:
                #     if self.CatFlag:
                #         val_pred = torch.argmax(Feat['cat'].detach(), dim=1)//(self.n_cluster+1)
                #     else:
                #         val_pred = torch.argmax(Feat['cat'].detach(), dim=1)
                # else:
                #     val_pred = torch.argmax(Feat['cat'].detach(), dim=1)//self.n_cluster
                # print('val_pred',val_pred.shape)
                current_scoreval = running_metric.update(val_pred, label)
                #############################Discrimination##################################
                ####Background
                record['TCET'] += TCELoss.item()
                lossT = TCELoss.item() + SCELoss.item()

                valScore = {'LossT': lossT, 'SCET': SCELoss.item(), 'TCET': TCELoss.item()}
                record['LossT'] += lossT
                valScore.update(current_scoreval)
                valMessage = self.visualizer.print_current_sorces_Seg(currentOpt.phase+'-ProSim', epoch, i, val_data_len, valScore)
                tbar.set_description(valMessage)
                if i > 10 and iterFlag:
                    break
        lossAvg = record['LossT'] / i
        CES_lossAvg = record['SCET'] / i
        DGAN_LossAvg = record['DGANT'] / i
        CET_lossAvg = record['TCET'] / i
        val_scores = running_metric.calculate_metrics()
        IterValScore = {'LossT': lossAvg, 'SCE': CES_lossAvg, 'DGAN': DGAN_LossAvg, 'TCE': CET_lossAvg}
        IterValScore.update(val_scores)
        message = self.visualizer.print_sorces_Seg(currentOpt.phase+'-ProSim', epoch, IterValScore)
        # messageT, core_dictT = running_metric.get_scoresT()
        cfg.TRAINLOG.LOGTXT.write(message + '\n')
        self.val_scores=val_scores
        exel_out = currentOpt.phase+'-ProSim-'+ cfg.TRAINLOG.DATA_NAMES[dataIndex][0] , epoch, lossAvg, CES_lossAvg, DGAN_LossAvg, CET_lossAvg, \
                   IterValScore['acc'].item(), IterValScore['mIoU'].item(), IterValScore['mf1'].item(), \
                   IterValScore['preacc'][0].item(), IterValScore['preacc'][1].item(), IterValScore['preacc'][2].item(), \
                   IterValScore['preacc'][3].item(), IterValScore['preacc'][4].item(), IterValScore['preacc'][5].item(), \
                   IterValScore['preIou'][0].item(), IterValScore['preIou'][1].item(), IterValScore['preIou'][2].item(), \
                   IterValScore['preIou'][3].item(), IterValScore['preIou'][4].item(), IterValScore['preIou'][5].item(), \
            # 0,0,0,0
        # core_dictT['accT'], core_dictT['unchgT'], core_dictT['chgT'], core_dictT['mF1T']
        cfg.TRAINLOG.EXCEL_LOG[excelName].append(exel_out)
        # cfg.TRAINLOG.EXCEL_LOG.save('./log/%s/%s/log.xlsx' % (name, time_now))
        figure_metrics = self.metrics.set_figure(metric_dict=figure_metrics,
                                                    acc=IterValScore['acc'].item(),
                                                    miou=IterValScore['mIoU'],
                                                    mf1=IterValScore['mf1'], preacc=IterValScore['preacc'],
                                                    premiou=IterValScore['preIou'],
                                                    CES_lossAvg=CES_lossAvg,
                                                    )
        return figure_metrics
class VailidateSim2():
    def __init__(self, opt, device=None,Osize=32,n_cluster=1,CatFlag=False):
        self.opt = opt
        self.DEVICE = device
        # self.dataloader = dataloader
        self.Osize = Osize
        self.running_metric = MetricsTracker(num_classes=6)
        self.visualizer = Visualizer(opt)
        self.n_cluster=n_cluster
        self.metrics = setFigurevalSeg(num_class=6)
        self.CatFlag=CatFlag
        self.figure_metrics = self.metrics.initialize_figure()
    def main(self, model, figure_metrics=None, epoch=1, cfg=None, iterFlag=False, dataloader=None, currentOpt=None,ProtoInput=None,classifierFlag=False):
        val_dataload = dataloader.load_data()
        iter_val = iter(val_dataload)
        val_data_len = len(val_dataload)
        epoch_start_time = time.time()
        # if epoch != start_epoch:
        #     epoch_iter = epoch_iter % len(train_loader)
        running_metric=self.running_metric
        running_metric.clear()
        model.eval()
        tbar = tqdm(range(val_data_len - 1))
        # train_data_len = len_source_loader
        record = {'LossT': 0, 'SCET': 0, 'DGANT': 0, 'TCET': 0}
        if currentOpt.phase == 'target':
            dataIndex = currentOpt.t
            excelName = 'wsT-' + cfg.TRAINLOG.DATA_NAMES[dataIndex][0]+'-Similarity'
        elif currentOpt.phase == 'val':
            dataIndex = currentOpt.s
            excelName = 'wsVal'+'-Similarity'
        with torch.no_grad():
            for i in tbar:
                data_val = next(iter_val)
                # epoch_iter += currentOpt.batch_size
                ############## Forward Pass ######################
                data_img_val = Variable(data_val['img_full']).to(self.DEVICE)
                label = Variable(data_val['label_full']).to(self.DEVICE)
                seg_val_pred, _, Feat = model.forward(data_img_val,ProtoInput=ProtoInput)
                # Generation: source
                label = F.interpolate(label.unsqueeze(1).float(), size=(self.Osize, self.Osize), mode='nearest')
                label = label.squeeze(1)  # 尺寸变为 [14, 512, 512]
                if classifierFlag:
                    FeatSim=Feat['cOut']
                    val_pred = torch.argmax(FeatSim.detach(), dim=1)
                    # print('val_pred',val_pred.max(),label.shape,label.max())
                else:
                    FeatSim=Feat['cat']
                    SSoft = F.softmax(FeatSim, dim=1)
                    SSoft = SSoft.view(SSoft.size(0), SSoft.size(1) // (self.n_cluster), (self.n_cluster),
                                       SSoft.size(2),
                                       SSoft.size(3))
                    SSoft = SSoft.sum(dim=2)
                    val_pred = torch.argmax(SSoft.detach(), dim=1)
                    # FeatSim=Feat['cat']
                    # val_pred = torch.argmax(FeatSim.detach(), dim=1) // (FeatSim.size(1) // 6)
                SCELoss = cross_entropy(FeatSim, label.long())
                TCELoss = torch.tensor([0.0]).to(self.DEVICE)
                record['SCET'] += SCELoss.item()
                current_scoreval = running_metric.update(val_pred, label)
                #############################Discrimination##################################
                ####Background
                record['TCET'] += TCELoss.item()
                lossT = TCELoss.item() + SCELoss.item()

                valScore = {'LossT': lossT, 'SCET': SCELoss.item(), 'TCET': TCELoss.item()}
                record['LossT'] += lossT
                valScore.update(current_scoreval)
                valMessage = self.visualizer.print_current_sorces_Seg(currentOpt.phase+'-ProSim', epoch, i, val_data_len, valScore)
                tbar.set_description(valMessage)
                if i > 10 and iterFlag:
                    break
        lossAvg = record['LossT'] / i
        CES_lossAvg = record['SCET'] / i
        DGAN_LossAvg = record['DGANT'] / i
        CET_lossAvg = record['TCET'] / i
        val_scores = running_metric.calculate_metrics()
        IterValScore = {'LossT': lossAvg, 'SCE': CES_lossAvg, 'DGAN': DGAN_LossAvg, 'TCE': CET_lossAvg}
        IterValScore.update(val_scores)
        message = self.visualizer.print_sorces_Seg(currentOpt.phase+'-ProSim', epoch, IterValScore)
        # messageT, core_dictT = running_metric.get_scoresT()
        cfg.TRAINLOG.LOGTXT.write(message + '\n')
        self.val_scores=val_scores
        exel_out = currentOpt.phase+'-ProSim-'+ cfg.TRAINLOG.DATA_NAMES[dataIndex][0] , epoch, lossAvg, CES_lossAvg, DGAN_LossAvg, CET_lossAvg, \
                   IterValScore['acc'].item(), IterValScore['mIoU'].item(), IterValScore['mf1'].item(), \
                   IterValScore['preacc'][0].item(), IterValScore['preacc'][1].item(), IterValScore['preacc'][2].item(), \
                   IterValScore['preacc'][3].item(), IterValScore['preacc'][4].item(), IterValScore['preacc'][5].item(), \
                   IterValScore['preIou'][0].item(), IterValScore['preIou'][1].item(), IterValScore['preIou'][2].item(), \
                   IterValScore['preIou'][3].item(), IterValScore['preIou'][4].item(), IterValScore['preIou'][5].item(), \
            # 0,0,0,0
        # core_dictT['accT'], core_dictT['unchgT'], core_dictT['chgT'], core_dictT['mF1T']
        cfg.TRAINLOG.EXCEL_LOG[excelName].append(exel_out)
        # cfg.TRAINLOG.EXCEL_LOG.save('./log/%s/%s/log.xlsx' % (name, time_now))
        figure_metrics = self.metrics.set_figure(metric_dict=figure_metrics,
                                                    acc=IterValScore['acc'].item(),
                                                    miou=IterValScore['mIoU'],
                                                    mf1=IterValScore['mf1'], preacc=IterValScore['preacc'],
                                                    premiou=IterValScore['preIou'],
                                                    CES_lossAvg=CES_lossAvg
                                                    )
        return figure_metrics
class VailidateSim():
    def __init__(self, opt, device=None,Osize=32,n_cluster=1,CatFlag=False):
        self.opt = opt
        self.DEVICE = device
        # self.dataloader = dataloader
        self.Osize = Osize
        self.running_metric = MetricsTracker(num_classes=6)
        self.visualizer = Visualizer(opt)
        self.n_cluster=n_cluster
        self.metrics = setFigurevalSeg(num_class=6)
        self.CatFlag=CatFlag
        self.figure_metrics = self.metrics.initialize_figure()
    def main(self, model, figure_metrics=None, epoch=1, cfg=None, iterFlag=False, dataloader=None, currentOpt=None,ProtoInput=None,classifierFlag=False):
        val_dataload = dataloader.load_data()
        iter_val = iter(val_dataload)
        val_data_len = len(val_dataload)
        epoch_start_time = time.time()
        # if epoch != start_epoch:
        #     epoch_iter = epoch_iter % len(train_loader)
        running_metric=self.running_metric
        running_metric.clear()
        model.eval()
        tbar = tqdm(range(val_data_len - 1))
        # train_data_len = len_source_loader
        record = {'LossT': 0, 'SCET': 0, 'DGANT': 0, 'TCET': 0}
        if currentOpt.phase == 'target':
            dataIndex = currentOpt.t
            excelName = 'wsT-' + cfg.TRAINLOG.DATA_NAMES[dataIndex][0]+'-Similarity'
        elif currentOpt.phase == 'val':
            dataIndex = currentOpt.s
            excelName = 'wsVal'+'-Similarity'
        with torch.no_grad():
            for i in tbar:
                data_val = next(iter_val)
                # epoch_iter += currentOpt.batch_size
                ############## Forward Pass ######################
                data_img_val = Variable(data_val['img_full']).to(self.DEVICE)
                label = Variable(data_val['label_full']).to(self.DEVICE)
                seg_val_pred, _, Feat = model.forward(data_img_val,ProtoInput=ProtoInput)
                # Generation: source
                label = F.interpolate(label.unsqueeze(1).float(), size=(self.Osize, self.Osize), mode='nearest')
                label = label.squeeze(1)  # 尺寸变为 [14, 512, 512]
                if classifierFlag:
                    FeatSim=Feat['cOut']
                    val_pred = torch.argmax(FeatSim.detach(), dim=1)
                    # print('val_pred',val_pred.max(),label.shape,label.max())
                else:
                    FeatSim=Feat['cat']
                    SSoft = F.softmax(FeatSim, dim=1)
                    SSoft = SSoft.view(SSoft.size(0), SSoft.size(1) // (self.n_cluster), (self.n_cluster),
                                       SSoft.size(2),
                                       SSoft.size(3))
                    SSoft = SSoft.max(dim=2)[0]
                    val_pred = torch.argmax(SSoft.detach(), dim=1)
                    # FeatSim=Feat['cat']
                    # val_pred = torch.argmax(FeatSim.detach(), dim=1) // (FeatSim.size(1) // 6)
                SCELoss = cross_entropy(FeatSim, label.long())
                TCELoss = torch.tensor([0.0]).to(self.DEVICE)
                record['SCET'] += SCELoss.item()
                # predict
                # print(Feat['cat'].shape)
                # print(Feat['cOut'].shape)
                # val_pred = torch.argmax(Feat['cOut'].detach(), dim=1)
                # if ProtoInput == None or self.CatFlag:
                #     if self.CatFlag:
                #         val_pred = torch.argmax(Feat['cat'].detach(), dim=1)//(self.n_cluster+1)
                #     else:
                #         val_pred = torch.argmax(Feat['cat'].detach(), dim=1)
                # else:
                #     val_pred = torch.argmax(Feat['cat'].detach(), dim=1)//self.n_cluster
                # print('val_pred',val_pred.shape)
                current_scoreval = running_metric.update(val_pred, label)
                #############################Discrimination##################################
                ####Background
                record['TCET'] += TCELoss.item()
                lossT = TCELoss.item() + SCELoss.item()

                valScore = {'LossT': lossT, 'SCET': SCELoss.item(), 'TCET': TCELoss.item()}
                record['LossT'] += lossT
                valScore.update(current_scoreval)
                valMessage = self.visualizer.print_current_sorces_Seg(currentOpt.phase+'-ProSim', epoch, i, val_data_len, valScore)
                tbar.set_description(valMessage)
                if i > 10 and iterFlag:
                    break
        lossAvg = record['LossT'] / i
        CES_lossAvg = record['SCET'] / i
        DGAN_LossAvg = record['DGANT'] / i
        CET_lossAvg = record['TCET'] / i
        val_scores = running_metric.calculate_metrics()
        IterValScore = {'LossT': lossAvg, 'SCE': CES_lossAvg, 'DGAN': DGAN_LossAvg, 'TCE': CET_lossAvg}
        IterValScore.update(val_scores)
        message = self.visualizer.print_sorces_Seg(currentOpt.phase+'-ProSim', epoch, IterValScore)
        # messageT, core_dictT = running_metric.get_scoresT()
        cfg.TRAINLOG.LOGTXT.write(message + '\n')
        self.val_scores=val_scores
        exel_out = currentOpt.phase+'-ProSim-'+ cfg.TRAINLOG.DATA_NAMES[dataIndex][0] , epoch, lossAvg, CES_lossAvg, DGAN_LossAvg, CET_lossAvg, \
                   IterValScore['acc'].item(), IterValScore['mIoU'].item(), IterValScore['mf1'].item(), \
                   IterValScore['preacc'][0].item(), IterValScore['preacc'][1].item(), IterValScore['preacc'][2].item(), \
                   IterValScore['preacc'][3].item(), IterValScore['preacc'][4].item(), IterValScore['preacc'][5].item(), \
                   IterValScore['preIou'][0].item(), IterValScore['preIou'][1].item(), IterValScore['preIou'][2].item(), \
                   IterValScore['preIou'][3].item(), IterValScore['preIou'][4].item(), IterValScore['preIou'][5].item(), \
            # 0,0,0,0
        # core_dictT['accT'], core_dictT['unchgT'], core_dictT['chgT'], core_dictT['mF1T']
        cfg.TRAINLOG.EXCEL_LOG[excelName].append(exel_out)
        # cfg.TRAINLOG.EXCEL_LOG.save('./log/%s/%s/log.xlsx' % (name, time_now))
        figure_metrics = self.metrics.set_figure(metric_dict=figure_metrics,
                                                    acc=IterValScore['acc'].item(),
                                                    miou=IterValScore['mIoU'],
                                                    mf1=IterValScore['mf1'], preacc=IterValScore['preacc'],
                                                    premiou=IterValScore['preIou'],
                                                    CES_lossAvg=CES_lossAvg
                                                    )
        return figure_metrics
class VailidateSimSecondthr():
    def __init__(self, opt, device=None,Osize=32,n_cluster=1,CatFlag=False):
        self.opt = opt
        self.DEVICE = device
        # self.dataloader = dataloader
        self.Osize = Osize
        self.running_metric = MetricsTracker(num_classes=6)
        self.visualizer = Visualizer(opt)
        self.n_cluster=n_cluster
        self.metrics = setFigurevalSeg(num_class=6)
        self.CatFlag=CatFlag
        self.figure_metrics = self.metrics.initialize_figure()
    def main(self, model, figure_metrics=None, epoch=1, cfg=None, iterFlag=False, dataloader=None,
             currentOpt=None,ProtoInput=None,classifierFlag=False,threshold=0.8):
        val_dataload = dataloader.load_data()
        iter_val = iter(val_dataload)
        val_data_len = len(val_dataload)
        epoch_start_time = time.time()
        # if epoch != start_epoch:
        #     epoch_iter = epoch_iter % len(train_loader)
        running_metric=self.running_metric
        running_metric.clear()
        model.eval()
        tbar = tqdm(range(val_data_len - 1))
        # train_data_len = len_source_loader
        record = {'LossT': 0, 'SCET': 0, 'DGANT': 0, 'TCET': 0}
        if currentOpt.phase == 'target':
            dataIndex = currentOpt.t
            excelName = 'wsT-' + cfg.TRAINLOG.DATA_NAMES[dataIndex][0]+'-SimilarityThr'
        elif currentOpt.phase == 'val':
            dataIndex = currentOpt.s
            excelName = 'wsVal'+'-SimilarityThr'
        with torch.no_grad():
            num_true=0
            for i in tbar:
                data_val = next(iter_val)
                # epoch_iter += currentOpt.batch_size
                ############## Forward Pass ######################
                data_img_val = Variable(data_val['img_full']).to(self.DEVICE)
                label = Variable(data_val['label_full']).to(self.DEVICE)
                seg_val_pred, _, Feat = model.forward(data_img_val,ProtoInput=ProtoInput)
                # Generation: source
                label = F.interpolate(label.unsqueeze(1).float(), size=(self.Osize, self.Osize), mode='nearest')
                label = label.squeeze(1)  # 尺寸变为 [14, 512, 512]
                if classifierFlag:
                    FeatSim=Feat['cOut']
                    val_pred = torch.argmax(FeatSim.detach(), dim=1)
                    # print('val_pred',val_pred.max(),label.shape,label.max())
                else:
                    ######classifier
                    FeatSim=seg_val_pred['out']
                    SSoft = F.softmax(FeatSim, dim=1)
                    FeatSimSoftMax=SSoft.max(dim=1)[0]
                    high_confidence_mask = (FeatSimSoftMax >= threshold)
                    pseudo_labels = torch.full((FeatSim.size(0), FeatSim.size(2), FeatSim.size(3)), 255,
                                               dtype=torch.long, device=FeatSim.device)
                    labelO = torch.full((label.size(0), label.size(1), FeatSim.size(2)), 255, dtype=torch.float,
                                        device=label.device)
                    val_pred = torch.argmax(SSoft.detach(), dim=1)

                    pseudo_labels[high_confidence_mask] = val_pred[high_confidence_mask]
                    labelO[high_confidence_mask] = label[high_confidence_mask]
                    num_true = num_true + torch.sum(high_confidence_mask).item()
                    ########prototype
                    # FeatSim=Feat['cat']
                    # SSoft = F.softmax(FeatSim, dim=1)
                    # SSoft = SSoft.view(SSoft.size(0), SSoft.size(1) // (self.n_cluster), (self.n_cluster),
                    #                    SSoft.size(2),
                    #                    SSoft.size(3))
                    # SSoft = SSoft.sum(dim=2)
                    # # FeatSimSoft=F.softmax(FeatSim,dim=1)
                    # FeatSimSoftMax=SSoft.max(dim=1)[0]
                    # # print(FeatSimSoftMax.max(),FeatSimSoftMax.min())
                    # high_confidence_mask = (FeatSimSoftMax >= threshold)
                    # pseudo_labels = torch.full((FeatSim.size(0), FeatSim.size(2), FeatSim.size(3)), 255,
                    #                            dtype=torch.long, device=FeatSim.device)
                    # labelO = torch.full((label.size(0), label.size(1), FeatSim.size(2)), 255, dtype=torch.float,
                    #                     device=label.device)
                    # # val_pred = torch.argmax(FeatSimSoft.detach(), dim=1) // (FeatSimSoft.size(1) // 6)
                    # val_pred = torch.argmax(SSoft.detach(), dim=1)
                    # # print(val_pred.shape,high_confidence_mask.shape,pseudo_labels.shape)
                    # pseudo_labels[high_confidence_mask] = val_pred[high_confidence_mask]
                    # labelO[high_confidence_mask] = label[high_confidence_mask]
                    # num_true = num_true + torch.sum(high_confidence_mask).item()

                    ###########max
                    # FeatSim = Feat['cat']
                    # SSoft = F.softmax(FeatSim, dim=1)
                    # SSoft = SSoft.view(SSoft.size(0), SSoft.size(1) // (self.n_cluster), (self.n_cluster),
                    #                    SSoft.size(2),
                    #                    SSoft.size(3))
                    # SSoft = SSoft.sum(dim=2)
                    # # FeatSimSoft=F.softmax(FeatSim,dim=1)
                    # FeatSimSoftMax = SSoft.max(dim=1)[0]
                    # # print(FeatSimSoftMax.max(),FeatSimSoftMax.min())
                    # high_confidence_mask = (FeatSimSoftMax >= threshold)
                    # pseudo_labels = torch.full((FeatSim.size(0), FeatSim.size(2), FeatSim.size(3)), 255,
                    #                            dtype=torch.long, device=FeatSim.device)
                    # labelO = torch.full((label.size(0), label.size(1), FeatSim.size(2)), 255, dtype=torch.float,
                    #                     device=label.device)
                    # # val_pred = torch.argmax(FeatSimSoft.detach(), dim=1) // (FeatSimSoft.size(1) // 6)
                    # val_pred = torch.argmax(SSoft.detach(), dim=1)
                    # # print(val_pred.shape,high_confidence_mask.shape,pseudo_labels.shape)
                    # pseudo_labels[high_confidence_mask] = val_pred[high_confidence_mask]
                    # labelO[high_confidence_mask] = label[high_confidence_mask]
                    # num_true = num_true + torch.sum(high_confidence_mask).item()
                    #
                    ####Original
                    # sorted_distances, _ = (FeatSim).sort(dim=1)  # [B, P, W, H]
                    # nearest_distance = sorted_distances[:, -1, :, :]  # 最近邻距离
                    # second_nearest_distance = sorted_distances[:, -2, :, :]  # 次近邻距离
                    # # distance_ratio = nearest_distance / (second_nearest_distance + 1e-8)  # 避免除零
                    # distance_ratio = second_nearest_distance / (nearest_distance + 1e-8)  # 避免除零
                    #
                    # high_confidence_mask = (distance_ratio <= threshold)
                    # pseudo_labels = torch.full((FeatSim.size(0), FeatSim.size(2), FeatSim.size(3)), 255, dtype=torch.long, device=FeatSim.device)
                    # labelO = torch.full((label.size(0), label.size(1), FeatSim.size(2)), 255, dtype=torch.float, device=label.device)
                    #
                    # # print('label',label.shape,labelO.shape,high_confidence_mask.shape)
                    # val_pred = torch.argmax(FeatSim.detach(), dim=1) // (FeatSim.size(1) // 6)
                    # # labelO=label.clone()
                    # pseudo_labels[high_confidence_mask] = val_pred[high_confidence_mask]
                    # labelO[high_confidence_mask] = label[high_confidence_mask]
                    # num_true = num_true+torch.sum(high_confidence_mask).item()
                    # print('num_true',num_true,distance_ratio.max(),distance_ratio.min())
                    # print(val_pred.shape, labelO.shape)
                # print('num_true',num_true,threshold)
                SCELoss = cross_entropy(FeatSim, label.long())
                TCELoss = torch.tensor([0.0]).to(self.DEVICE)
                record['SCET'] += SCELoss.item()
                # predict
                # print(Feat['cat'].shape)
                # print(Feat['cOut'].shape)
                # val_pred = torch.argmax(Feat['cOut'].detach(), dim=1)
                # if ProtoInput == None or self.CatFlag:
                #     if self.CatFlag:
                #         val_pred = torch.argmax(Feat['cat'].detach(), dim=1)//(self.n_cluster+1)
                #     else:
                #         val_pred = torch.argmax(Feat['cat'].detach(), dim=1)
                # else:
                #     val_pred = torch.argmax(Feat['cat'].detach(), dim=1)//self.n_cluster
                # print('val_pred',val_pred.shape)
                current_scoreval = running_metric.update(pseudo_labels, labelO)
                #############################Discrimination##################################
                ####Background
                record['TCET'] += TCELoss.item()
                lossT = TCELoss.item() + SCELoss.item()

                valScore = {'LossT': lossT, 'SCET': SCELoss.item(), 'TCET': TCELoss.item()}
                record['LossT'] += lossT
                valScore.update(current_scoreval)
                valMessage = self.visualizer.print_current_sorces_Seg(currentOpt.phase+'-ProSimThr', epoch, i, val_data_len, valScore)
                tbar.set_description(valMessage)
                if i > 10 and iterFlag:
                    break
        lossAvg = record['LossT'] / i
        CES_lossAvg = record['SCET'] / i
        DGAN_LossAvg = record['DGANT'] / i
        CET_lossAvg = record['TCET'] / i
        val_scores = running_metric.calculate_metrics()
        IterValScore = {'LossT': lossAvg, 'SCE': CES_lossAvg, 'DGAN': DGAN_LossAvg, 'TCE': CET_lossAvg}
        IterValScore.update(val_scores)
        message = self.visualizer.print_sorces_Seg(currentOpt.phase+'-ProSimThr', epoch, IterValScore)
        # messageT, core_dictT = running_metric.get_scoresT()
        cfg.TRAINLOG.LOGTXT.write(message + '\n')
        self.val_scores=val_scores
        exel_out = currentOpt.phase+'-ProSimThr-'+ cfg.TRAINLOG.DATA_NAMES[dataIndex][0] , epoch, lossAvg, CES_lossAvg, DGAN_LossAvg, CET_lossAvg, \
                   IterValScore['acc'].item(), IterValScore['mIoU'].item(), IterValScore['mf1'].item(), \
                   IterValScore['preacc'][0].item(), IterValScore['preacc'][1].item(), IterValScore['preacc'][2].item(), \
                   IterValScore['preacc'][3].item(), IterValScore['preacc'][4].item(), IterValScore['preacc'][5].item(), \
                   IterValScore['preIou'][0].item(), IterValScore['preIou'][1].item(), IterValScore['preIou'][2].item(), \
                   IterValScore['preIou'][3].item(), IterValScore['preIou'][4].item(), IterValScore['preIou'][5].item(), \
                    threshold,num_true
            # 0,0,0,0
        # core_dictT['accT'], core_dictT['unchgT'], core_dictT['chgT'], core_dictT['mF1T']
        cfg.TRAINLOG.EXCEL_LOG[excelName].append(exel_out)
        # cfg.TRAINLOG.EXCEL_LOG.save('./log/%s/%s/log.xlsx' % (name, time_now))
        figure_metrics = self.metrics.set_figure(metric_dict=figure_metrics,
                                                    acc=IterValScore['acc'].item(),
                                                    miou=IterValScore['mIoU'],
                                                    mf1=IterValScore['mf1'], preacc=IterValScore['preacc'],
                                                    premiou=IterValScore['preIou'],
                                                    CES_lossAvg=CES_lossAvg,
                                                    )
        return figure_metrics
class ProtoDistTest():
    def __init__(self, opt, device=None,outFeatSize=32,outClassiferSize=128,n_cluster=1,CatFlag=False):
        # self.model = model
        self.opt = opt
        self.DEVICE = device
        # self.dataloader = dataloader
        self.outFeatSize=outFeatSize
        self.outClassiferSize=outClassiferSize
        self.running_metric = MetricsTracker(num_classes=6)
        self.visualizer = Visualizer(opt)
        self.n_cluster=n_cluster
        self.CatFlag=CatFlag

    def main(self,model,inputProto=None,epoch=1,cfg=None,iterFlag=False,dataloader=None,currentOpt=None):

        val_dataload = dataloader.load_data()
        iter_val = iter(val_dataload)
        val_data_len = len(val_dataload)
        tbar = tqdm(range(val_data_len))

        # if epoch != start_epoch:
        #     epoch_iter = epoch_iter % len(train_loader)
        running_metric = self.running_metric
        running_metric.clear()
        model.eval()
        if inputProto is None:
            TaskName = 'CurrentProto'
        else:
            TaskName = 'GlobalProto'
        excelName = None
        dataIndex = None
        if currentOpt.phase == 'target':
            dataIndex = currentOpt.t
            excelName = 'wsT-' + cfg.TRAINLOG.DATA_NAMES[dataIndex][0] + '-%sP' % (TaskName[0])
        elif currentOpt.phase == 'val':
            dataIndex = currentOpt.s
            excelName = 'wsVal' + '-%sP' % (TaskName[0])

        # train_data_len = len_source_loader
        record = {'LossT': 0, 'SCET': 0, 'DGANT': 0, 'TCET': 0}
        class_countsTT = torch.zeros(6, dtype=torch.int64)

        with torch.no_grad():
            for i in tbar:
                data_val = next(iter_val)
                # epoch_iter += currentOpt.batch_size
                ############## Forward Pass ######################
                data_img_val = Variable(data_val['img_full']).to(self.DEVICE)
                label = Variable(data_val['label_full']).to(self.DEVICE)
                seg_val_pred, PrototypeT, FeatTT = model.forward(data_img_val,ProtoInput=inputProto,getPFlag=2)
                # Generation: source
                label = F.interpolate(label.unsqueeze(1).float(), size=(self.outClassiferSize, self.outClassiferSize), mode='nearest')
                label = label.squeeze(1)  # 尺寸变为 [14, 512, 512]
                #####Similarity
                # if inputProto is None:
                #     prototypes_expanded = PrototypeT['GetProto'] # [B, 6, 128]
                # else:
                #     prototypes_expanded = inputProto # [B, 6, 128]
                # prototype_features_transposed = prototypes_expanded.squeeze(-1).squeeze(-1)  # [B, 128, 6]
                # deep_features_reshaped = FeatTT['asspF'].reshape(FeatTT['asspF'].size(0), FeatTT['asspF'].size(1), -1)  # [B, 128, H*W]
                # # print('prototype_features_transposed',prototype_features_transposed.shape,deep_features_reshaped.shape)
                # prototype_features_transposed = F.normalize(prototype_features_transposed, p=2,
                #                                   dim=2)  # Normalize along the channel dimension
                # deep_features_reshaped = F.normalize(deep_features_reshaped, p=2,
                #                                             dim=2)  # Normalize along the channel dimension
                # similarity = torch.matmul(prototype_features_transposed, deep_features_reshaped)  # [B, 6, H*W]#([10, 6, 1024])
                #
                # similarity_reshaped = similarity.reshape(FeatTT['asspF'].size(0), prototypes_expanded.size(1), FeatTT['asspF'].size(2), FeatTT['asspF'].size(3))  # [B, 6, H, W]
                # val_pred = torch.argmax(similarity_reshaped, dim=1)  # [B, H, W]
                # distancesSoftmax=F.log_softmax(similarity_reshaped,dim=1)

                # # predict
                if inputProto is None or self.CatFlag:
                    # print('P',PrototypeT['GetProto'].shape)
                    prototypes_expanded = PrototypeT['query'].expand(-1, -1, -1, self.outFeatSize, self.outFeatSize)  # [B, 6, 128, H, W]
                else:
                    prototypes_expanded = PrototypeT['query'].expand(-1, -1, -1, self.outFeatSize, self.outFeatSize)  # [B, 6, 128, H, W]
                deep_features_expanded = FeatTT['asspF'].unsqueeze(1)  # [B, 1, 128, H, W]
                prototypes_expanded = F.normalize(prototypes_expanded, p=2, dim=2)  # Normalize along the channel dimension
                deep_features_expanded = F.normalize(deep_features_expanded, p=2, dim=2)
                # # # distances = torch.norm(prototypes_expanded - deep_features_expanded, p=2, dim=2)  # [B, 6, H, W]
                distances = torch.norm(deep_features_expanded - prototypes_expanded, p=2, dim=2)  # [B, 6, H, W]
                val_pred = torch.argmin(distances.detach(), dim=1)
                class_countsTT = class_countsTT+count_elements_per_class(val_pred)
                # distancesSoftmax=F.log_softmax(1/distances,dim=1)
                # if inputProto is not None:
                #     print('G',distances.shape,val_pred.min(),val_pred.max(),self.n_cluster)

                # 计算余弦相似度
                # prototypes_expanded = prototypes_expanded.permute(0, 3, 4, 1, 2)  # Change shape to [B, H, W, 6, 128]
                # deep_features_expanded = deep_features_expanded.permute(0, 3, 4, 1, 2)  # Change shape to [B, H, W, 1, 128]
                # deep_features_expanded = deep_features_expanded.expand(-1, -1, -1, 6, -1)  # Now tensor2 shape is [B, H, W, 6, 128]
                # tensor1_norm = F.normalize(prototypes_expanded, p=2, dim=4)  # Normalize along the channel dimension
                # tensor2_norm = F.normalize(deep_features_expanded, p=2, dim=4)
                # similarity = (tensor1_norm * tensor2_norm).sum(dim=4)  # Dot product along the channel dimension
                # val_pred, _ = similarity.max(dim=3)  # Max over the channel "6" dimension
                # # print('similarity',similarity.shape)
                # similarity=similarity.permute(0,3,1,2)

                TSoft = F.log_softmax(1/distances, dim=1)
                # print("T",TSoft.shape)
                label = F.interpolate(label.unsqueeze(1).float(), size=(self.outFeatSize, self.outFeatSize), mode='nearest')
                label = label.squeeze(1)  # 尺寸变为 [14, 512, 512]
                # print('val_pred',val_pred.shape)
                # print('val_pred',val_pred.shape,val_pred.min(),val_pred.max())
                if inputProto is None or self.CatFlag:
                    if self.CatFlag:
                        current_scoreval = running_metric.update(val_pred//(self.n_cluster+1), label)
                        TSoft = TSoft.view(TSoft.size(0), TSoft.size(1) // (self.n_cluster + 1), (self.n_cluster + 1),
                                           TSoft.size(2),
                                           TSoft.size(3))
                        TSoft = TSoft.sum(dim=2)
                    else:
                        current_scoreval = running_metric.update(val_pred, label)
                else:
                    current_scoreval = running_metric.update(val_pred // self.n_cluster, label)
                    TSoft = TSoft.view(TSoft.size(0), TSoft.size(1) // (self.n_cluster), self.n_cluster,
                                       TSoft.size(2),
                                       TSoft.size(3))
                    TSoft = TSoft.sum(dim=2)
                # SCELoss = cross_entropy(TSoft, label.long())
                SCELoss = F.nll_loss(TSoft, label.long())
                TCELoss = torch.tensor([0.0]).to(self.DEVICE)
                record['SCET'] += SCELoss.item()
                #############################Discrimination##################################
                ####Background
                record['TCET'] += TCELoss.item()
                lossT = TCELoss.item() + SCELoss.item()
                valScore = {'LossT': lossT, 'SCET': SCELoss.item(), 'TCET': TCELoss.item()}
                record['LossT'] += lossT
                valScore.update(current_scoreval)
                valMessage = self.visualizer.print_current_sorces_Seg(currentOpt.phase + '-%sP' % (TaskName[0]), epoch, i, val_data_len,
                                                                      valScore)
                tbar.set_description(valMessage)
                if i > 10 and iterFlag:
                    break
        rate = class_countsTT / class_countsTT.sum()

        lossAvg = record['LossT'] / i
        CES_lossAvg = record['SCET'] / i
        # DGAN_LossAvg = record['DGANT'] / i
        # CET_lossAvg = record['TCET'] / i
        target_scores = self.running_metric.calculate_metrics()
        # IteScore = {'Loss': lossAvg, 'TCE': CET_lossAvg}
        IteScore = {'LossT': lossAvg, 'CE': CES_lossAvg, }

        IteScore.update(target_scores)

        messageT = self.visualizer.print_sorces_Seg(currentOpt.phase + '-%sP' % (TaskName[0]), epoch, IteScore)
        # cfg.TRAINLOG.LOGTXT.write('\n================ [%s] Target Test (%s) ================\n' % (
        #     cfg.TRAINLOG.DATA_NAMES[kk], time.strftime("%c")))
        cfg.TRAINLOG.LOGTXT.write(messageT + '\n')
        # print(cfg.TRAINLOG.DATA_NAMES[dataIndex])
        exel_out = currentOpt.phase + '-' + cfg.TRAINLOG.DATA_NAMES[dataIndex][
                                            :2] + '-%sP' % (TaskName[0]), epoch, lossAvg, CES_lossAvg, 0, 0, \
                   IteScore['acc'].item(), IteScore['mIoU'].item(), IteScore['mf1'].item(), \
                   IteScore['preacc'][0].item(), IteScore['preacc'][1].item(), \
                   IteScore['preacc'][
                       2].item(), \
                   IteScore['preacc'][3].item(), IteScore['preacc'][4].item(), \
                   IteScore['preacc'][
                       5].item(), \
                   IteScore['preIou'][0].item(), IteScore['preIou'][1].item(), \
                   IteScore['preIou'][
                       2].item(), \
                   IteScore['preIou'][3].item(), IteScore['preIou'][4].item(), \
                   IteScore['preIou'][
                       5].item(),
        # print('excelName', excelName)
        cfg.TRAINLOG.EXCEL_LOG[excelName].append(exel_out)
class VailidateSimDP():
    def __init__(self, opt, device=None,Osize=32,n_cluster=1,CatFlag=False):
        self.opt = opt
        self.DEVICE = device
        # self.dataloader = dataloader
        self.Osize = Osize
        self.running_metric = MetricsTracker(num_classes=6)
        self.visualizer = Visualizer(opt)
        self.n_cluster=n_cluster
        self.metrics = setFigurevalSeg(num_class=6)
        self.CatFlag=CatFlag
        self.figure_metrics = self.metrics.initialize_figure()
    def main(self, model, figure_metrics=None, epoch=1, cfg=None, iterFlag=False, dataloader=None, currentOpt=None,ProtoInput=None,classifierFlag=False):
        val_dataload = dataloader.load_data()
        iter_val = iter(val_dataload)
        val_data_len = len(val_dataload)
        running_metric=self.running_metric
        running_metric.clear()
        model.eval()
        tbar = tqdm(range(val_data_len - 1))
        record = {'LossT': 0, 'SCET': 0, 'DGANT': 0, 'TCET': 0}
        if currentOpt.phase == 'target':
            dataIndex = currentOpt.t
            excelName = 'wsT-' + cfg.TRAINLOG.DATA_NAMES[dataIndex][0]+'-Similarity'
        elif currentOpt.phase == 'val':
            dataIndex = currentOpt.s
            excelName = 'wsVal'+'-Similarity'
        with torch.no_grad():
            for i in tbar:
                data_val = next(iter_val)
                ############## Forward Pass ######################
                data_img_val = Variable(data_val['img_full']).to(self.DEVICE)
                label = Variable(data_val['label_full']).to(self.DEVICE)
                seg_val_pred, _, Feat = model.forward(data_img_val,ProtoInput=ProtoInput)
                # Generation: source
                label = F.interpolate(label.unsqueeze(1).float(), size=(self.Osize, self.Osize), mode='nearest')
                label = label.squeeze(1)  # 尺寸变为 [14, 512, 512]
                if classifierFlag:
                    FeatSim=Feat['cOut']
                    val_pred = torch.argmax(FeatSim.detach(), dim=1)
                else:
                    FeatSim = F.softmax(Feat['cat'], dim=1)
                    FeatSim = multiFeatClaSum(FeatSim, self.n_cluster, M='mean')
                    val_pred = torch.argmax(FeatSim.detach(), dim=1)

                SCELoss = cross_entropy(FeatSim, label.long())
                TCELoss = torch.tensor([0.0]).to(self.DEVICE)
                record['SCET'] += SCELoss.item()

                current_scoreval = running_metric.update(val_pred, label)
                #############################Discrimination##################################
                ####Background
                record['TCET'] += TCELoss.item()
                lossT = TCELoss.item() + SCELoss.item()

                valScore = {'LossT': lossT, 'SCET': SCELoss.item(), 'TCET': TCELoss.item()}
                record['LossT'] += lossT
                valScore.update(current_scoreval)
                valMessage = self.visualizer.print_current_sorces_Seg(currentOpt.phase+'-ProSim', epoch, i, val_data_len, valScore)
                tbar.set_description(valMessage)
                if i > 10 and iterFlag:
                    break
        lossAvg = record['LossT'] / i
        CES_lossAvg = record['SCET'] / i
        DGAN_LossAvg = record['DGANT'] / i
        CET_lossAvg = record['TCET'] / i
        val_scores = running_metric.calculate_metrics()
        IterValScore = {'LossT': lossAvg, 'SCE': CES_lossAvg, 'DGAN': DGAN_LossAvg, 'TCE': CET_lossAvg}
        IterValScore.update(val_scores)
        message = self.visualizer.print_sorces_Seg(currentOpt.phase+'-ProSim', epoch, IterValScore)
        # messageT, core_dictT = running_metric.get_scoresT()
        cfg.TRAINLOG.LOGTXT.write(message + '\n')
        self.val_scores=val_scores
        exel_out = currentOpt.phase+'-ProSim-'+ cfg.TRAINLOG.DATA_NAMES[dataIndex][0] , epoch, lossAvg, CES_lossAvg, DGAN_LossAvg, CET_lossAvg, \
                   IterValScore['acc'].item(), IterValScore['mIoU'].item(), IterValScore['mf1'].item(), \
                   IterValScore['preacc'][0].item(), IterValScore['preacc'][1].item(), IterValScore['preacc'][2].item(), \
                   IterValScore['preacc'][3].item(), IterValScore['preacc'][4].item(), IterValScore['preacc'][5].item(), \
                   IterValScore['preIou'][0].item(), IterValScore['preIou'][1].item(), IterValScore['preIou'][2].item(), \
                   IterValScore['preIou'][3].item(), IterValScore['preIou'][4].item(), IterValScore['preIou'][5].item(), \
            # 0,0,0,0
        # core_dictT['accT'], core_dictT['unchgT'], core_dictT['chgT'], core_dictT['mF1T']
        cfg.TRAINLOG.EXCEL_LOG[excelName].append(exel_out)
        # cfg.TRAINLOG.EXCEL_LOG.save('./log/%s/%s/log.xlsx' % (name, time_now))
        figure_metrics = self.metrics.set_figure(metric_dict=figure_metrics,
                                                    acc=IterValScore['acc'].item(),
                                                    miou=IterValScore['mIoU'],
                                                    mf1=IterValScore['mf1'], preacc=IterValScore['preacc'],
                                                    premiou=IterValScore['preIou'],
                                                    CES_lossAvg=CES_lossAvg,
                                                    )
        return figure_metrics
class ProtoDistTestDP():
    def __init__(self, opt, device=None,outFeatSize=32,outClassiferSize=128,n_cluster=1,CatFlag=False):
        # self.model = model
        self.opt = opt
        self.DEVICE = device
        # self.dataloader = dataloader
        self.outFeatSize=outFeatSize
        self.outClassiferSize=outClassiferSize
        self.running_metric = MetricsTracker(num_classes=6)
        self.visualizer = Visualizer(opt)
        self.n_cluster=n_cluster
        self.CatFlag=CatFlag

    def main(self,model,inputProto=None,epoch=1,cfg=None,iterFlag=False,dataloader=None,currentOpt=None):

        val_dataload = dataloader.load_data()
        iter_val = iter(val_dataload)
        val_data_len = len(val_dataload)
        tbar = tqdm(range(val_data_len))

        # if epoch != start_epoch:
        #     epoch_iter = epoch_iter % len(train_loader)
        running_metric = self.running_metric
        running_metric.clear()
        model.eval()
        if inputProto is None:
            TaskName = 'CurrentProto'
        else:
            TaskName = 'GlobalProto'
        excelName = None
        dataIndex = None
        if currentOpt.phase == 'target':
            dataIndex = currentOpt.t
            excelName = 'wsT-' + cfg.TRAINLOG.DATA_NAMES[dataIndex][0] + '-%sP' % (TaskName[0])
        elif currentOpt.phase == 'val':
            dataIndex = currentOpt.s
            excelName = 'wsVal' + '-%sP' % (TaskName[0])

        # train_data_len = len_source_loader
        record = {'LossT': 0, 'SCET': 0, 'DGANT': 0, 'TCET': 0}
        with torch.no_grad():
            for i in tbar:
                data_val = next(iter_val)
                # epoch_iter += currentOpt.batch_size
                ############## Forward Pass ######################
                data_img_val = Variable(data_val['img_full']).to(self.DEVICE)
                label = Variable(data_val['label_full']).to(self.DEVICE)
                seg_val_pred, PrototypeT, FeatTT = model.forward(data_img_val,ProtoInput=inputProto)
                # Generation: source
                label = F.interpolate(label.unsqueeze(1).float(), size=(self.outClassiferSize, self.outClassiferSize), mode='nearest')
                label = label.squeeze(1)  # 尺寸变为 [14, 512, 512]
                #####Similarity
                # if inputProto is None:
                #     prototypes_expanded = PrototypeT['GetProto'] # [B, 6, 128]
                # else:
                #     prototypes_expanded = inputProto # [B, 6, 128]
                # prototype_features_transposed = prototypes_expanded.squeeze(-1).squeeze(-1)  # [B, 128, 6]
                # deep_features_reshaped = FeatTT['asspF'].reshape(FeatTT['asspF'].size(0), FeatTT['asspF'].size(1), -1)  # [B, 128, H*W]
                # # print('prototype_features_transposed',prototype_features_transposed.shape,deep_features_reshaped.shape)
                # prototype_features_transposed = F.normalize(prototype_features_transposed, p=2,
                #                                   dim=2)  # Normalize along the channel dimension
                # deep_features_reshaped = F.normalize(deep_features_reshaped, p=2,
                #                                             dim=2)  # Normalize along the channel dimension
                # similarity = torch.matmul(prototype_features_transposed, deep_features_reshaped)  # [B, 6, H*W]#([10, 6, 1024])
                #
                # similarity_reshaped = similarity.reshape(FeatTT['asspF'].size(0), prototypes_expanded.size(1), FeatTT['asspF'].size(2), FeatTT['asspF'].size(3))  # [B, 6, H, W]
                # val_pred = torch.argmax(similarity_reshaped, dim=1)  # [B, H, W]
                # distancesSoftmax=F.log_softmax(similarity_reshaped,dim=1)

                # # predict
                if inputProto is None or self.CatFlag:
                    # print('P',PrototypeT['GetProto'].shape)
                    prototypes_expanded = PrototypeT['query'].expand(-1, -1, -1, self.outFeatSize, self.outFeatSize)  # [B, 6, 128, H, W]
                else:
                    prototypes_expanded = PrototypeT['query'].expand(-1, -1, -1, self.outFeatSize, self.outFeatSize)  # [B, 6, 128, H, W]
                deep_features_expanded = FeatTT['asspF'].unsqueeze(1)  # [B, 1, 128, H, W]
                # print('deep_features_expanded',deep_features_expanded.shape, prototypes_expanded.shape)
                prototypes_expanded = F.normalize(prototypes_expanded, p=2, dim=2)  # Normalize along the channel dimension
                deep_features_expanded = F.normalize(deep_features_expanded, p=2, dim=2)
                # # # distances = torch.norm(prototypes_expanded - deep_features_expanded, p=2, dim=2)  # [B, 6, H, W]
                distances = torch.norm(deep_features_expanded - prototypes_expanded, p=2, dim=2)  # [B, 6, H, W]
                val_pred = torch.argmin(distances.detach(), dim=1)
                # distancesSoftmax=F.log_softmax(1/distances,dim=1)
                # if inputProto is not None:
                #     print('G',distances.shape,val_pred.min(),val_pred.max(),self.n_cluster)

                # 计算余弦相似度
                # prototypes_expanded = prototypes_expanded.permute(0, 3, 4, 1, 2)  # Change shape to [B, H, W, 6, 128]
                # deep_features_expanded = deep_features_expanded.permute(0, 3, 4, 1, 2)  # Change shape to [B, H, W, 1, 128]
                # deep_features_expanded = deep_features_expanded.expand(-1, -1, -1, 6, -1)  # Now tensor2 shape is [B, H, W, 6, 128]
                # tensor1_norm = F.normalize(prototypes_expanded, p=2, dim=4)  # Normalize along the channel dimension
                # tensor2_norm = F.normalize(deep_features_expanded, p=2, dim=4)
                # similarity = (tensor1_norm * tensor2_norm).sum(dim=4)  # Dot product along the channel dimension
                # val_pred, _ = similarity.max(dim=3)  # Max over the channel "6" dimension
                # # print('similarity',similarity.shape)
                # similarity=similarity.permute(0,3,1,2)

                TSoft = F.log_softmax(1/distances, dim=1)
                # print("T",TSoft.shape)
                label = F.interpolate(label.unsqueeze(1).float(), size=(self.outFeatSize, self.outFeatSize), mode='nearest')
                label = label.squeeze(1)  # 尺寸变为 [14, 512, 512]
                # print('val_pred',val_pred.shape)
                # print('val_pred',val_pred.shape,val_pred.min(),val_pred.max())
                if inputProto is None or self.CatFlag:
                    if self.CatFlag:
                        current_scoreval = running_metric.update(val_pred//(self.n_cluster+1), label)
                        TSoft = TSoft.view(TSoft.size(0), TSoft.size(1) // (self.n_cluster + 1), (self.n_cluster + 1),
                                           TSoft.size(2),
                                           TSoft.size(3))
                        TSoft = TSoft.sum(dim=2)
                    else:
                        current_scoreval = running_metric.update(val_pred, label)
                else:
                    val_pred=classify_out(val_pred,self.n_cluster)
                    # print('val_pred',val_pred.shape)
                    current_scoreval = running_metric.update(val_pred, label)
                    # TSoft = TSoft.view(TSoft.size(0), TSoft.size(1) // (self.n_cluster), self.n_cluster,
                    #                    TSoft.size(2),
                    #                    TSoft.size(3))
                    # TSoft = TSoft.sum(dim=2)
                    TSoft = multiFeatClaSum(TSoft, self.n_cluster, M='max')

                # SCELoss = cross_entropy(TSoft, label.long())
                SCELoss = F.nll_loss(TSoft, label.long())
                TCELoss = torch.tensor([0.0]).to(self.DEVICE)
                record['SCET'] += SCELoss.item()
                #############################Discrimination##################################
                ####Background
                record['TCET'] += TCELoss.item()
                lossT = TCELoss.item() + SCELoss.item()
                valScore = {'LossT': lossT, 'SCET': SCELoss.item(), 'TCET': TCELoss.item()}
                record['LossT'] += lossT
                valScore.update(current_scoreval)
                valMessage = self.visualizer.print_current_sorces_Seg(currentOpt.phase + '-%sP' % (TaskName[0]), epoch, i, val_data_len,
                                                                      valScore)
                tbar.set_description(valMessage)
                if i > 10 and iterFlag:
                    break
        lossAvg = record['LossT'] / i
        CES_lossAvg = record['SCET'] / i
        # DGAN_LossAvg = record['DGANT'] / i
        # CET_lossAvg = record['TCET'] / i
        target_scores = self.running_metric.calculate_metrics()
        # IteScore = {'Loss': lossAvg, 'TCE': CET_lossAvg}
        IteScore = {'LossT': lossAvg, 'CE': CES_lossAvg, }

        IteScore.update(target_scores)

        messageT = self.visualizer.print_sorces_Seg(currentOpt.phase + '-%sP' % (TaskName[0]), epoch, IteScore)
        # cfg.TRAINLOG.LOGTXT.write('\n================ [%s] Target Test (%s) ================\n' % (
        #     cfg.TRAINLOG.DATA_NAMES[kk], time.strftime("%c")))
        cfg.TRAINLOG.LOGTXT.write(messageT + '\n')
        # print(cfg.TRAINLOG.DATA_NAMES[dataIndex])
        exel_out = currentOpt.phase + '-' + cfg.TRAINLOG.DATA_NAMES[dataIndex][
                                            :2] + '-%sP' % (TaskName[0]), epoch, lossAvg, CES_lossAvg, 0, 0, \
                   IteScore['acc'].item(), IteScore['mIoU'].item(), IteScore['mf1'].item(), \
                   IteScore['preacc'][0].item(), IteScore['preacc'][1].item(), \
                   IteScore['preacc'][
                       2].item(), \
                   IteScore['preacc'][3].item(), IteScore['preacc'][4].item(), \
                   IteScore['preacc'][
                       5].item(), \
                   IteScore['preIou'][0].item(), IteScore['preIou'][1].item(), \
                   IteScore['preIou'][
                       2].item(), \
                   IteScore['preIou'][3].item(), IteScore['preIou'][4].item(), \
                   IteScore['preIou'][
                       5].item(),
        # print('excelName', excelName)
        cfg.TRAINLOG.EXCEL_LOG[excelName].append(exel_out)


