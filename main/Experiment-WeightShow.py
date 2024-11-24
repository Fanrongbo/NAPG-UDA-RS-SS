import torch
# from mpl_toolkits.axes_grid1 import make_axes_locatable

import random
from data.data_loader import CreateDataLoader
# from modelDA.retinex import retinex_synthesis
from util.train_util import *
from option.train_options import TrainOptions
from util.visualizer import Visualizer
from util.metric_tool2 import ConfuseMatrixMeter
import math
from tqdm import tqdm
from util.drawTool import setFigureval, plotFigureTarget, plotFigureSegDATarget,plotFigureSegDA, MakeRecordFloder, confuseMatrix, plotFigure, save_pickle,\
    setFigureDASeg,setFigurevalSeg
from torch.autograd import Variable
from option.config import cfg
from modelDA import utils as model_utils
import matplotlib.pyplot as plt
import time
import numpy as np
import os
from model.RSP.networks import BASE_Transformer
from model.RSP.parser import get_parser_with_args
from util.metric_toolSeg import MetricsTracker
# from model.deeplab import DeepLabV3
# from model.deeplabv3plus import DeepLabV3Plus
from model.deeplabv3plusPrototype import DeepLabV3PlusMultiPrototypeSingleKey
from modelDA.discriminator import FCDiscriminator,FCDiscriminatorLow,FCDiscriminatorHigh,FCDiscriminatorLowMask,FCDiscriminatorHighMask
from util.func import prob_2_entropy

from DataLoadAppendix import *
from IPython import display

def lcm(a, b): return abs(a * b) / math.gcd(a, b) if a and b else 0


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, num):
    LEARNING_RATE = 2.5e-4
    lr = lr_poly(LEARNING_RATE, i_iter, num, 0.9)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter, num):
    lr = lr_poly(1e-4, i_iter, num, 0.9)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand,Mask):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    # loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd = (F.kl_div(log_pred_student, pred_teacher, reduction="none")*Mask).sum(1).mean()

    loss_kd *= temperature**2
    return loss_kd
if __name__ == '__main__':
    # time.sleep(2500)
    gpu_id = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    opt = TrainOptions().parse(save=False, gpu=gpu_id)
    opt.num_epochs =1
    opt.batch_size = 10
    opt.use_ce_loss = True
    opt.use_hybrid_loss = False
    opt.num_decay_epochs = 0
    opt.dset = 'mmlab'
    opt.model_type = 'deeplabv3'
    cfg.DA.NUM_DOMAINS_BN = 1
    ttest = False
    saveLast = True
    # name = opt.dset + '-DA/' + opt.model_type + '-GAN-Onekey'
    name = opt.dset + '-Constrast-KL/' + opt.model_type + '-OriginalShow'
    opt.LChannel = False
    opt.dataroot = opt.dataroot + '/' + opt.dset
    opt.s = 0
    opt.t = 1
    cfg.DA.S = opt.s
    cfg.DA.T = opt.t
    if opt.dset == 'mmlab':
        cfg.TRAINLOG.DATA_NAMES = ['potsdamRGB6', 'vailhingen6']
        # cfg.TRAINLOG.DATA_NAMES = ['vailhingen6', 'potsdamRGB6']

    Nepoch = 10
    opt.load_pretrain = True
    if opt.load_pretrain:
        #
        saveroot = '/data/project_frb/SegDA/AttenDA2/zmain/log/mmlab-Constrast-Global-Multi-P/deeplabv3-OneKey_OneGAN_WeightAsspFeat/20240505-00_48_po-va-addProto-01softmaxMeanUp2-1p/'
        save_path = saveroot + '/savemodel/_21_acc-0.8896_miou-0.7947_mf1-0.8836.pth'
        # saveroot = '/data/project_frb/SegDA/AttenDA2/zmain/log/mmlab-Base-GlobalP/deeplabv3-OneKey_OneGANUp2/20240422-10_43_po-va-noprototype/'
        # save_path = saveroot + '/savemodel/_21_acc-0.9044_miou-0.8089_mf1-0.8930.pth'
        # saveroot = '/data/project_frb/SegDA/AttenDA2/zmain/log/mmlab-Base-GlobalP/deeplabv3-OneKey_OneGANUp2/20240429-20_38_po-va-Max-addprototype01-Drop-2P/'
        # save_path = saveroot + '/savemodel/_21_acc-0.9190_miou-0.8316_mf1-0.9069.pth'
        # saveroot = './log/mmlab-DA/deeplab-Resnet101-GAN/20240301-14_19_po-va/'
        # save_path = saveroot + '/savemodel/_6_acc-0.5839_miou-0.4237_mf1-0.5817.pth'
    else:
        saveroot = None
        save_path = None

    cfg.TRAINLOG.EXCEL_LOGSheet = ['wsTrain', 'wsVal']
    for wsN in range(len(cfg.TRAINLOG.DATA_NAMES)):
        if wsN == opt.s:
            continue
        else:
            cfg.TRAINLOG.EXCEL_LOGSheet.append('wsT-' + cfg.TRAINLOG.DATA_NAMES[wsN][0])
            # wsT = [cfg.TRAINLOG.EXCEL_LOG['wsT1'], cfg.TRAINLOG.EXCEL_LOG['wsT2']]
    print('\n########## Recording File Initialization#################')
    SEED = 1240#opt.SEED
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    cfg.TRAINLOG.STARTTIME = time.strftime("%Y%m%d-%H_%M", time.localtime())
    time_now = time.strftime(
        "%Y%m%d-%H_%M_" + cfg.TRAINLOG.DATA_NAMES[opt.s][:2] + '-' + cfg.TRAINLOG.DATA_NAMES[opt.t][:2] +
        '-addProto-0111softmaxMeanUp2-valProto3-KLnew2',
        time.localtime())
    filename_main = os.path.basename(__file__)
    start_epoch, epoch_iter = MakeRecordFloder(name, time_now, opt, filename_main, opt.load_pretrain, saveroot)
    train_metrics = setFigureDASeg(num_class=6)
    val_metrics = setFigurevalSeg(num_class=6)
    T_metrics = setFigurevalSeg(num_class=6)
    val_metrics_ema = setFigurevalSeg(num_class=6)
    T_metrics_ema = setFigurevalSeg(num_class=6)
    figure_train_metrics = train_metrics.initialize_figure()
    figure_val_metrics = val_metrics.initialize_figure()
    figure_T_metrics = T_metrics.initialize_figure()
    figure_val_metrics_ema = val_metrics_ema.initialize_figure()
    figure_T_metrics_ema = T_metrics_ema.initialize_figure()

    print('\n########## Load the Source Dataset #################')
    opt.phase = 'train'
    train_loader = CreateDataLoader(opt)
    print("[%s] dataset [%s] was created successfully! Num= %d" %
          (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.s], len(train_loader)))
    cfg.TRAINLOG.LOGTXT.write("[%s] dataset [%s] was created successfully! Num= %d" %
                              (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.s], len(train_loader)) + '\n')
    # opt.phase = 'val'
    # val_loader = CreateDataLoader(opt)
    # print("[%s] dataset [%s] was created successfully! Num= %d" %
    #       (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.s], len(val_loader)))
    # cfg.TRAINLOG.LOGTXT.write("[%s] dataset [%s] was created successfully! Num= %d" %
    #                           (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.s], len(val_loader)) + '\n')

    print('\n########## Load the Target Dataset #################')
    t_loaderDict = {}
    for i in range(len(cfg.TRAINLOG.DATA_NAMES)):
        if i != opt.s:
            opt.t = i
            opt.phase = 'target'
            t_loader = CreateDataLoader(opt)
            t_loaderDict.update({cfg.TRAINLOG.DATA_NAMES[i]: t_loader})
            print("[%s] dataset [%s] was created successfully! Num= %d" %
                  (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.t], len(t_loader)))
            cfg.TRAINLOG.LOGTXT.write("[%s] dataset [%s] was created successfully! Num= %d" %
                                      (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.t], len(t_loader)) + '\n')
    print(t_loaderDict)

    t_loaderTestDict = {}
    for i in range(len(cfg.TRAINLOG.DATA_NAMES)):
        if i != opt.s:
            opt.t = i
            opt.phase = 'targetTest'
            t_loader = CreateDataLoader(opt)
            t_loaderTestDict.update({cfg.TRAINLOG.DATA_NAMES[i]: t_loader})
            print("[%s] dataset [%s] was created successfully! Num= %d" %
                  (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.t], len(t_loader)))
            cfg.TRAINLOG.LOGTXT.write("[%s] dataset [%s] was created successfully! Num= %d" %
                                      (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.t], len(t_loader)) + '\n')
    print(t_loaderTestDict)
    tool = CDModelutil(opt)

    print('\n########## Build the Molde #################')
    # initialize model
    model_state_dict = None
    opt.phase = 'train'

    parser, metadata = get_parser_with_args()
    opt2 = parser.parse_args()
    # opt2.backbone='resnet'
    opt2.mode = 'office'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = BASE_Transformer(opt2, input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
    #                          with_pos='learned', enc_depth=1, dec_depth=8).to(DEVICE)
    n_clusters=3
    netStudent = DeepLabV3PlusMultiPrototypeSingleKey(num_classes=6, args=opt2,n_cluster=n_clusters).to(DEVICE)
    # netTeacher = DeepLabV3PlusPrototypeSingleKey(num_classes=6, args=opt2).to(DEVICE)

    # device_ids = [0, 1]  # id为0和1的两块显卡
    # model = torch.nn.DataParallel(net, device_ids=device_ids)
    print('DEVICE:', DEVICE)
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(netStudent.parameters(), lr=opt.lr,
                                    momentum=0.9,
                                    weight_decay=5e-4)
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(netStudent.parameters(), lr=opt.lr,
                                     betas=(0.5, 0.999))
    else:
        raise NotImplemented(opt.optimizer)

    source_label = 0
    target_label = 1
    LEARNING_RATE_D = 1e-4
    model_D1 = FCDiscriminatorHighMask(num_classes=6)
    model_D1.to(DEVICE)
    # model_D2 = FCDiscriminatorLowMask(num_classes=256)
    # model_D2.to(DEVICE)
    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=LEARNING_RATE_D, betas=(0.9, 0.99))
    bce_loss = torch.nn.functional.binary_cross_entropy

    if opt.load_pretrain:
        modelL_state_dict, modelGAN_state_dict, modelGAN2_state_dict, optimizer_state_dict,Centerdict = tool.load_ckptGANCenter(save_path)#,Centerdict
        if modelL_state_dict is not None:
            model_utils.init_weights(netStudent, modelL_state_dict, None, False)
            # model_utils.init_weights(netTeacher, modelL_state_dict, None, False)

        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
        if modelGAN_state_dict is not None:
            model_utils.init_weights(model_D1, modelGAN_state_dict, None, False)
        # if modelGAN2_state_dict is not None:
        #     model_utils.init_weights(model_D2, modelGAN2_state_dict, None, False)
        if Centerdict is not None:
            unchgN=Centerdict[1]
            chgN=Centerdict[2]
            # Center=Centerdict[0]
            # print('Center',Center.shape,unchgN,chgN)
            # CenterSingel = torch.cat([Center[:, :, :unchgN].mean(dim=2), Center[:, :, chgN:].mean(dim=2)],
            #                          dim=-1).unsqueeze(-1)  # ([14, 32, 2, 1])
    else:
        cfg.DA.BN_DOMAIN_MAP = {cfg.TRAINLOG.DATA_NAMES[cfg.DA.S]: 0, cfg.TRAINLOG.DATA_NAMES[cfg.DA.T]: 1}

    print('optimizer:', opt.optimizer)
    cfg.TRAINLOG.LOGTXT.write('optimizer: ' + opt.optimizer + '\n')
    criterion = nn.CrossEntropyLoss()


    print('\n########## Load the Log Moudle #################')
    visualizer = Visualizer(opt)
    tmp = 1
    running_metric = MetricsTracker(num_classes=6)

    DA = True
    TagerMetricNum = 0
    cm = plt.cm.get_cmap('jet')
    Osize = 32

    for epoch in range(start_epoch, opt.num_epochs + opt.num_decay_epochs + 1):
        if True:
            opt.phase = 'val'
            # Load Data
            val_dataload = train_loader.load_data()
            iter_val = iter(val_dataload)
            val_data_len = len(val_dataload)
            epoch_start_time = time.time()

            running_metric.clear()
            netStudent.eval()
            # netTeacher.eval()

            tbar = tqdm(range(val_data_len - 1))
            # train_data_len = len_source_loader
            record = {'LossT': 0, 'SCET': 0, 'DGANT': 0, 'TCET': 0}
            centerCur = torch.zeros(128, 6).to(DEVICE)
            # ProtoList=[]
            proto_lists = {f'ProtoList_{ii}': [] for ii in range(6)}

            with torch.no_grad():
                for i in tbar:
                    data_val = next(iter_val)
                    epoch_iter += opt.batch_size
                    ############## Forward Pass ######################
                    data_img_val = Variable(data_val['img_full']).to(DEVICE)
                    label = Variable(data_val['label_full']).to(DEVICE)
                    seg_val_pred, _, features = netStudent.forward(data_img_val)
                    assp_features = features['asspF']
                    label = F.interpolate(label.unsqueeze(1).float(), size=(Osize, Osize),
                                          mode='nearest').long()  # torch.Size([10, 1, 16, 16])
                    # print('label',label.shape)
                    # 处理每个类别
                    for b in range(label.size(0)):
                        for cls_idx in range(6):
                            cls_mask = (label[b].unsqueeze(0) == cls_idx)  # 创建掩码
                            cls_mask = cls_mask.expand_as(assp_features[b].unsqueeze(0))  # 扩展掩码至 [10, 256, 16, 16]
                            if cls_mask.any():  # 检查是否有该类别的数据
                                cls_features = assp_features[b].unsqueeze(0)[cls_mask].view(-1, 128)  # 应用掩码并重塑张量
                                # print('cls_features',(cls_features.mean(dim=0, keepdim=True).shape))
                                proto_lists[f'ProtoList_{cls_idx}'].append(
                                    cls_features.mean(dim=0, keepdim=True))  # 计算平均特征
                    if i > 10 and ttest:
                        break
                # n_clusters = 1
                # 存储聚类中心和标签的字典
                # print('proto_lists',proto_lists)
                cluster_centers = {}
                cluster_labels = {}
                PrototypeS = []
                # 对每类特征进行KMeans聚类
                for key in proto_lists:
                    # print('key',key)
                    # 获取当前类别的特征，转换为合适的 numpy 数组格式
                    features = torch.cat(proto_lists[key], dim=0).squeeze(0).squeeze(
                        1)  # 去掉最后一个维度，并转换为numpy数组 #features torch.Size([52, 256])
                    # 创建KMeans实例并拟合数据
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features.detach().cpu().numpy())
                    # 存储聚类结果
                    cluster_centers[key] = kmeans.cluster_centers_  # (1, 256)(2, 128)
                    cluster_labels[key] = kmeans.labels_
                    expand_Center = np.expand_dims(kmeans.cluster_centers_, axis=0)
                    expanded_cluster_centers_ = np.tile(expand_Center,
                                                        (opt.batch_size, 1, 1))  # 在第一个维度复制10次，第二个维度保持不变

                    PrototypeS.append(torch.tensor(expanded_cluster_centers_, requires_grad=False).to(DEVICE))
            PG = torch.cat([p.unsqueeze(1) for p in PrototypeS], dim=1)  # [5, 6, 2, 128]



        # end of epoch
        # print(opt.num_epochs,opt.num_decay_epochs)
        iter_end_time = time.time()
        # print('End of epoch %d / %d \t Time Taken: %d sec \t best acc: %.5f (at epoch: %d) ' %
        #       (epoch, opt.num_epochs + opt.num_decay_epochs, time.time() - epoch_start_time, best_val_acc, best_epoch))
        # np.savetxt(cfg.TRAINLOG.ITER_PATH, (epoch + 1, 0), delimiter=',', fmt='%d')
        cfg.TRAINLOG.EXCEL_LOG.save('./log/%s/%s/log.xlsx' % (name, time_now))
        if epoch % 1 == 0:

            for kk in range(len(cfg.TRAINLOG.DATA_NAMES)):
                if kk == opt.s:
                    continue
                print('target:', cfg.TRAINLOG.DATA_NAMES[kk])
                TagerMetricNum += 1
                t_loader = t_loaderTestDict[cfg.TRAINLOG.DATA_NAMES[kk]]
                t_data = t_loader.load_data()
                t_data_len = len(t_data)
                tbar = tqdm(range(t_data_len))
                iter_t = iter(t_data)
                running_metric.clear()
                opt.phase = 'target'
                netStudent.eval()
                # netTeacher.eval()
                with torch.no_grad():
                    record = {'LossT': 0, 'SCET': 0, 'DGANT': 0, 'TCET': 0}
                    for i in tbar:
                        data_test = next(iter_t)

                        data_img_T = Variable(data_test['img_full']).to(DEVICE)
                        labelT = Variable(data_test['label_full']).to(DEVICE)
                        labelT = F.interpolate(labelT.unsqueeze(1).float(), size=(128, 128))
                        labelT = labelT.squeeze(1)  # 尺寸变为 [14, 512, 512]

                        seg_target_pred, _, _ = netStudent.forward(data_img_T)

                        TCELoss = cross_entropy(seg_target_pred['outUp'], labelT.long())

                        # update metric
                        TScore = {'LossT': TCELoss.item(), 'TCET': TCELoss.item()}
                        target_pred = torch.argmax(seg_target_pred['outUp'].detach(), dim=1)
                        current_scoreT = running_metric.update(target_pred, labelT)

                        record['TCET'] += TCELoss.item()
                        record['LossT'] = record['TCET']

                        # current_scoreT = running_metric.confuseMT(pr=target_pred.cpu().numpy(),
                        #                                          gt=labelT.cpu().numpy())  # 更新
                        TScore.update(current_scoreT)
                        # valMessageT = visualizer.print_current_scores(opt.phase, epoch, i, t_data_len, Score)
                        MessageT = visualizer.print_current_sorces_Seg(opt.phase + '-student', epoch, i, t_data_len,
                                                                       TScore)

                        tbar.set_description(MessageT)
                        if i > 10 and ttest:
                            break

            lossAvg = record['LossT'] / i
            CES_lossAvg = record['SCET'] / i
            DGAN_LossAvg = record['DGANT'] / i
            CET_lossAvg = record['TCET'] / i
            target_scores = running_metric.calculate_metrics()
            IterTargetScore = {'Loss': lossAvg, 'TCE': CET_lossAvg}
            IterTargetScore.update(target_scores)

            messageT = visualizer.print_sorces_Seg('target out' + '-student', epoch, IterTargetScore)
            cfg.TRAINLOG.LOGTXT.write('\n================ [%s] Target Test (%s) ================\n' % (
                cfg.TRAINLOG.DATA_NAMES[kk], time.strftime("%c")))

            cfg.TRAINLOG.LOGTXT.write(messageT + '\n')

            exel_out = 'T-' + cfg.TRAINLOG.DATA_NAMES[kk][
                0] + '-student', epoch, lossAvg, CES_lossAvg, DGAN_LossAvg, CET_lossAvg, \
                       IterTargetScore['acc'].item(), IterTargetScore['mIoU'].item(), IterTargetScore['mf1'].item(), \
                       IterTargetScore['preacc'][0].item(), IterTargetScore['preacc'][1].item(), \
                       IterTargetScore['preacc'][
                           2].item(), \
                       IterTargetScore['preacc'][3].item(), IterTargetScore['preacc'][4].item(), \
                       IterTargetScore['preacc'][
                           5].item(), \
                       IterTargetScore['preIou'][0].item(), IterTargetScore['preIou'][1].item(), \
                       IterTargetScore['preIou'][
                           2].item(), \
                       IterTargetScore['preIou'][3].item(), IterTargetScore['preIou'][4].item(), \
                       IterTargetScore['preIou'][
                           5].item(),

            cfg.TRAINLOG.EXCEL_LOG['wsT-' + cfg.TRAINLOG.DATA_NAMES[kk][0]].append(exel_out)

            figure_T_metrics_ema = T_metrics.set_figure(metric_dict=figure_T_metrics_ema,
                                                        acc=IterTargetScore['acc'].item(),
                                                        miou=IterTargetScore['mIoU'],
                                                        mf1=IterTargetScore['mf1'], preacc=IterTargetScore['preacc'],
                                                        premiou=IterTargetScore['preIou'],
                                                        CES_lossAvg=CES_lossAvg,
                                                        )

            save_pickle(figure_T_metrics, "./log/%s/%s/fig_T.pkl" % (name, time_now))
            save_pickle(figure_T_metrics_ema, "./log/%s/%s/fig_T_ema.pkl" % (name, time_now))

            cfg.TRAINLOG.EXCEL_LOG.save('./log/%s/%s/log.xlsx' % (name, time_now))
            for kk in range(len(cfg.TRAINLOG.DATA_NAMES)):
                if kk == opt.s:
                    continue
                print('target:', cfg.TRAINLOG.DATA_NAMES[kk])
                TagerMetricNum += 1
                t_loader = t_loaderTestDict[cfg.TRAINLOG.DATA_NAMES[kk]]
                t_data = t_loader.load_data()
                t_data_len = len(t_data)
                tbar = tqdm(range(t_data_len))
                iter_t = iter(t_data)
                running_metric.clear()
                opt.phase = 'target'
                netStudent.eval()
                # netTeacher.eval()
                with torch.no_grad():
                    record = {'LossT': 0, 'SCET': 0, 'DGANT': 0, 'TCET': 0}
                    for i in tbar:
                        data_test = next(iter_t)

                        data_img_T = Variable(data_test['img_full']).to(DEVICE)
                        labelT = Variable(data_test['label_full']).to(DEVICE)
                        labelT = F.interpolate(labelT.unsqueeze(1).float(), size=(128, 128))
                        labelT = labelT.squeeze(1)  # 尺寸变为 [14, 512, 512]

                        # seg_target_pred,_,FeatTT = netStudent.forward(data_img_T,DomainLabel=1,ProtoInput=PG)
                        seg_target_pred,_,FeatTT = netStudent.forward(data_img_T)

                        fig, axs = plt.subplots(opt.batch_size, 2+6, figsize=(20, 2 * opt.batch_size))  # b行2列

                        for ii in range(opt.batch_size):
                            # print('FeatTT[-1]',FeatTT[-1].shape)
                            img1 = axs[ii, 0].imshow(FeatTT['Weight'][-1][ii].detach().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
                            axs[ii, 0].set_title('Max Weight:%.2f-%.2f' % (FeatTT['Weight'][-1][ii].min(), FeatTT['Weight'][-1][ii].max()))
                            axs[ii, 0].axis('off')
                            cbar1 = fig.colorbar(img1, ax=axs[ii, 0], extend='both', fraction=0.046, pad=0.04)
                            cbar1.set_ticks([0, 0.25, 0.5, 0.75, 1])
                            cbar1.set_label('Probability')

                            img2 = axs[ii, 1].imshow(FeatTT['Weight'][-2][ii].detach().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
                            # axs[ii, 1].set_title('Mean Weight')
                            axs[ii, 1].set_title('Mean Weight:%.2f-%.2f' % (FeatTT['Weight'][-2][ii].min(), FeatTT['Weight'][-2][ii].max()))
                            axs[ii, 1].axis('off')
                            cbar2 = fig.colorbar(img2, ax=axs[ii, 1], extend='both', fraction=0.046, pad=0.04)
                            cbar2.set_ticks([0, 0.25, 0.5, 0.75, 1])
                            cbar2.set_label('Probability')
                            for jj in range(2,8):
                                img3 = axs[ii, jj].imshow(FeatTT['Weight'][-3][ii,jj].detach().cpu().numpy(), cmap='jet',
                                                         vmin=0, vmax=1)
                                axs[ii, jj].set_title('Weight:%.2f-%.2f' % (
                                        FeatTT['Weight'][-3][ii,jj].min(), FeatTT['Weight'][-3][ii,jj].max()))
                                axs[ii, jj].axis('off')
                                cbar3 = fig.colorbar(img3, ax=axs[ii, jj], extend='both', fraction=0.046, pad=0.04)
                                cbar3.set_ticks([0, 0.25, 0.5, 0.75, 1])
                                cbar3.set_label('Probability')

                        # cbar = fig.colorbar(axs[ii, 1], ax=axs[ii, 1].ravel().tolist(), extend='both', fraction=0.01, pad=0.04)
                        # cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
                        # cbar.set_label('Probability')
                        plt.tight_layout()
                        plt.savefig('./outimg/5-6Weightshow3/image_%d.png' % i)
                        plt.clf()
                        display.clear_output(wait=True)
                        display.display(plt.gcf())

                        TCELoss = cross_entropy(seg_target_pred['outUp'], labelT.long())

                        # update metric
                        TScore = {'LossT': TCELoss.item(), 'TCET': TCELoss.item()}
                        target_pred = torch.argmax(seg_target_pred['outUp'].detach(), dim=1)
                        current_scoreT = running_metric.update(target_pred, labelT)

                        record['TCET'] += TCELoss.item()
                        record['LossT'] = record['TCET']

                        # current_scoreT = running_metric.confuseMT(pr=target_pred.cpu().numpy(),
                        #                                          gt=labelT.cpu().numpy())  # 更新
                        TScore.update(current_scoreT)
                        # valMessageT = visualizer.print_current_scores(opt.phase, epoch, i, t_data_len, Score)
                        MessageT = visualizer.print_current_sorces_Seg(opt.phase+'-student', epoch, i, t_data_len, TScore)

                        tbar.set_description(MessageT)
                        if i > 10 and ttest:
                            break

            save_pickle(figure_T_metrics, "./log/%s/%s/fig_T.pkl" % (name, time_now))
            save_pickle(figure_T_metrics_ema, "./log/%s/%s/fig_T_ema.pkl" % (name, time_now))

            cfg.TRAINLOG.EXCEL_LOG.save('./log/%s/%s/log.xlsx' % (name, time_now))





    print('================ Training Completed (%s) ================\n' % time.strftime("%c"))
    cfg.TRAINLOG.LOGTXT.write('\n================ Training Completed (%s) ================\n' % time.strftime("%c"))
    plotFigureSegDA(figure_train_metrics, figure_val_metrics_ema, opt.num_epochs + opt.num_decay_epochs, name, opt.model_type,
                 time_now)
    plotFigureSegDATarget(figure_T_metrics_ema, opt.num_epochs + opt.num_decay_epochs, name, opt.model_type,
                 time_now)
    time_end = time.strftime("%Y%m%d-%H_%M", time.localtime())
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    # if scheduler:
    #     print('Training Start lr:', lr, '  Training Completion lr:', scheduler.get_last_lr())
    print('Training Start Time:', cfg.TRAINLOG.STARTTIME, '  Training Completion Time:', time_end, '  Total Epoch Num:',
          epoch)
    print('saved path:', './log/{}/{}'.format(name, time_now))
    cfg.TRAINLOG.LOGTXT.write(
        'Training Start Time:' + cfg.TRAINLOG.STARTTIME + '  Training Completion Time:' + time_end + 'Total Epoch Num:' + str(
            epoch) + '\n')
