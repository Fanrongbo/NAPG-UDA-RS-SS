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
from IPython import display

# from model.deeplab import DeepLabV3
# from model.deeplabv3plus import DeepLabV3Plus
# from model.deeplabv3plusPrototypeShow import DeepLabV3Plus
from model.deeplabv3plusPrototypeShow import DeepLabV3PlusSimGlobalLinearKL3,DeepLabV3PlusSimGlobalLinearKL3neibor,SemanticSegmentationVarianceLoss,SemanticSegmentationCosineLoss
from modelDA.discriminator import FCDiscriminator,FCDiscriminatorLow,FCDiscriminatorHigh,FCDiscriminatorLowMask,FCDiscriminatorHighMask
from util.func import prob_2_entropy
from sklearn.preprocessing import StandardScaler
from util.ValidateVisualization_tool2 import *
from util.ProtoDistValidate import *
from data.DataLoadAppendix import *
from torchinfo import summary

def normalize(logit):
    # mean = logit.mean(dim=-1, keepdims=True)
    # stdv = logit.std(dim=-1, keepdims=True)
    mean = logit.mean(dim=(2, 3), keepdim=True)
    stdv = logit.std(dim=(2, 3), keepdim=True)
    return (logit - mean) / (1e-7 + stdv)
def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    # loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none")

    loss_kd *= temperature**2
    return loss_kd
if __name__ == '__main__':
    # time.sleep(60*200)
    gpu_id = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    opt = TrainOptions().parse(save=False, gpu=gpu_id)
    opt.num_epochs = 35
    opt.batch_size = 10
    opt.use_ce_loss = True
    opt.use_hybrid_loss = False
    opt.num_decay_epochs = 0
    opt.dset = 'mmlab'
    opt.model_type = 'deeplabv3'
    opt.augmentations=True
    cfg.DA.NUM_DOMAINS_BN = 1
    ttest = True

    saveLast = False

    # name = opt.dset + '-DA/' + opt.model_type + '-GAN-Onekey'
    # name = opt.dset + '-zNewWeightCos/' + opt.model_type + '-teacherStudent'
    name = opt.dset + '-z-EX/' + opt.model_type + '-step3-KL'

    opt.LChannel = False
    opt.dataroot = opt.dataroot + '/' + opt.dset
    opt.s = 0
    opt.t = 1
    cfg.DA.S = opt.s
    cfg.DA.T = opt.t
    if opt.dset == 'mmlab':
        # cfg.TRAINLOG.DATA_NAMES = ['potsdamRGB6', 'vailhingen6']
        # cfg.TRAINLOG.DATA_NAMES = ['va-poRGB', 'potsdamRGB6']
        cfg.TRAINLOG.DATA_NAMES = ['vailhingen6', 'potsdamIRRG6']#potsdamRGB6Full,potsdamRGB6Full

        # cfg.TRAINLOG.DATA_NAMES = ['vailhingen6', 'potsdamRGB6']
    default_rcParams = plt.rcParams.copy()

    Nepoch = 10
    opt.load_pretrain = True
    if opt.load_pretrain:
        #

        saveroot = '/datab/frb-project/AttenDA2/zmain/log/mmlab-z-EX/deeplabv3-step2-Neibor/20240730-02_39_va-po-noNeibor-noCompare-noCycleGAN-N=111122-RGB-Load111100-Psort-pseudo-Detach-IR-nomask-noKLBest/'
        save_path = saveroot + '/savemodel/_41_acc-0.9484_miou-0.8494_mf1-0.9116.pth'
    else:
        saveroot = None
        save_path = None

    cfg.TRAINLOG.EXCEL_LOGSheet = ['wsTrain', 'wsVal','wsVal-Similarity','wsVal-CP','wsVal-GP']
    for wsN in range(len(cfg.TRAINLOG.DATA_NAMES)):
        if wsN == opt.s:
            continue
        else:
            cfg.TRAINLOG.EXCEL_LOGSheet.append('wsT-' + cfg.TRAINLOG.DATA_NAMES[wsN][0])
            cfg.TRAINLOG.EXCEL_LOGSheet.append('wsT-' + cfg.TRAINLOG.DATA_NAMES[wsN][0] + '-Similarity')
            cfg.TRAINLOG.EXCEL_LOGSheet.append('wsT-' + cfg.TRAINLOG.DATA_NAMES[wsN][0] + '-CP')
            cfg.TRAINLOG.EXCEL_LOGSheet.append('wsT-' + cfg.TRAINLOG.DATA_NAMES[wsN][0] + '-GP')

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
        '-RGB-noNorm',
        time.localtime())
    filename_main = os.path.basename(__file__)
    start_epoch, epoch_iter = MakeRecordFloder(name, time_now, opt, filename_main, opt.load_pretrain, saveroot)
    train_metrics = setFigureDASeg(num_class=6)
    figure_train_metrics = train_metrics.initialize_figure()


    print('\n########## Load the Source Dataset #################')
    opt.phase = 'train'
    train_loader = CreateDataLoader(opt)
    print("[%s] dataset [%s] was created successfully! Num= %d" %
          (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.s], len(train_loader)))
    cfg.TRAINLOG.LOGTXT.write("[%s] dataset [%s] was created successfully! Num= %d" %
                              (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.s], len(train_loader)) + '\n')

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
    n_cluster=2
    netStudent = DeepLabV3PlusSimGlobalLinearKL3neibor(num_classes=6, args=opt2,n_cluster=n_cluster).to(DEVICE)
    netTeacher = DeepLabV3PlusSimGlobalLinearKL3neibor(num_classes=6, args=opt2,n_cluster=n_cluster).to(DEVICE)

    # device_ids = [0, 1]  # id为0和1的两块显卡
    # model = torch.nn.DataParallel(net, device_ids=device_ids)
    print('DEVICE:', DEVICE)
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(netStudent.parameters(), lr=opt.lr,
                                    momentum=0.9,
                                    weight_decay=5e-4)
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(netStudent.parameters(), lr=opt.lr/5,
                                     betas=(0.5, 0.999))
        # optimizer = torch.optim.Adam(netStudent.parameters(), lr=opt.lr,
        #                              betas=(0.5, 0.999))

    else:
        raise NotImplemented(opt.optimizer)

    source_label = 0
    target_label = 1
    LEARNING_RATE_D = 1e-4
    model_D1 = FCDiscriminatorHighMask(num_classes=6)
    model_D1.to(DEVICE)
    model_D2 = FCDiscriminatorHighMask(num_classes=6)
    model_D2.to(DEVICE)
    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=LEARNING_RATE_D, betas=(0.9, 0.99))
    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=LEARNING_RATE_D, betas=(0.9, 0.99))
    input_size = (1, 6, 32, 32)

    summary(model_D2, input_size=input_size, col_names=["input_size", "output_size", "num_params", "mult_adds"])

    bce_loss = torch.nn.functional.binary_cross_entropy
    if opt.load_pretrain:
        modelL_state_dict, modelGAN_state_dict, modelGAN2_state_dict, optimizer_state_dict,Centerdict = tool.load_ckptGANCenter(save_path)#,Centerdict
        if modelL_state_dict is not None:
            model_utils.init_weights(netStudent, modelL_state_dict, None, False)
            model_utils.init_weights(netTeacher, modelL_state_dict, None, False)

        # if optimizer_state_dict is not None:
        #     optimizer.load_state_dict(optimizer_state_dict)
        if modelGAN_state_dict is not None:
            model_utils.init_weights(model_D1, modelGAN_state_dict, None, False)
        if modelGAN2_state_dict is not None:
            model_utils.init_weights(model_D2, modelGAN2_state_dict, None, False)
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
    weightPicPathsave='./log/'+'/'+name+'/'+time_now+'/'+'WightPic/savedata/'
    os.makedirs(weightPicPathsave)
    weightPicPathsave = './log/' + '/' + name + '/' + time_now + '/' + 'WightPic/savepic/'
    os.makedirs(weightPicPathsave)

    outFeatSize=32
    outClassiferSize=128
    t_loader = t_loaderTestDict[cfg.TRAINLOG.DATA_NAMES[opt.t]]

    GetIntCentriod_sourceST=GetIntCentriodDPST(opt=opt,dataloaderS=train_loader,dataloaderT=t_loader,
                                           device=DEVICE,Osize=outFeatSize,n_cluster=[1,1,1,2,2,2])
    GetIntCentriod_sourceT=GetIntCentriodDPT(opt=opt,dataloaderS=train_loader,dataloaderT=t_loader,
                                           device=DEVICE,Osize=outFeatSize,n_cluster=[1,1,1,2,2,2])

    ########Val Set
    Vailidate_Val=Vailidate(opt=opt, device=DEVICE,Osize=outClassiferSize)
    figure_val_metrics=Vailidate_Val.figure_metrics
    Vailidate_ValSim = VailidateSim(opt=opt, device=DEVICE,Osize=outFeatSize,n_cluster=n_cluster,CatFlag=False)
    figure_val_Sim_metrics=Vailidate_ValSim.figure_metrics


    ProtoDist_Global_ProtoVal=ProtoDistTest(opt=opt,device=DEVICE,outFeatSize=outFeatSize,
                                            outClassiferSize=outClassiferSize,n_cluster=n_cluster,CatFlag=False)
    ProtoDist_Current_ProtoVal=ProtoDistTest(opt=opt,device=DEVICE,outFeatSize=outFeatSize,
                                             outClassiferSize=outClassiferSize,n_cluster=n_cluster,CatFlag=False)

    #####Target Set
    Vailidate_Target = Vailidate(opt=opt, device=DEVICE,Osize=outClassiferSize)
    figure_target_metrics = Vailidate_Target.figure_metrics
    Vailidate_TargetSim = VailidateSim(opt=opt, device=DEVICE,Osize=outFeatSize,n_cluster=n_cluster,CatFlag=False)
    figure_TargetSim_metrics = Vailidate_TargetSim.figure_metrics

    # figure_target_metrics = Vailidate_Target.figure_metrics

    ProtoDist_Global_ProtoTarget = ProtoDistTest(opt=opt, device=DEVICE,outFeatSize=outFeatSize,
                                                 outClassiferSize=outClassiferSize,n_cluster=n_cluster,CatFlag=False)
    ProtoDist_Current_ProtoTarget = ProtoDistTest(opt=opt, device=DEVICE,outFeatSize=outFeatSize,
                                                  outClassiferSize=outClassiferSize,n_cluster=n_cluster,CatFlag=False)
    ####Plot Predict Visualization
    showIndex = random.sample(range(len(t_loader)//opt.batch_size + 1), 1)
    savepath = './log/' + '/' + name + '/' + time_now + '/'
    PredictVisualizaion_Target=PredictVisualizaion(device=DEVICE,Osize=outFeatSize,savepath=savepath,
                                                   showIndex=showIndex,n_cluster=n_cluster,CatFlag=False)
    SimilarityVisualizaion_Target=SimilarityVis(device=DEVICE,Osize=outFeatSize,savepath=savepath,showIndex=showIndex,
                                                n_cluster=n_cluster,outFeatSize=outFeatSize,CatFlag=False)
    Osize=32
    varLoss=SemanticSegmentationVarianceLoss(num_classes=6)
    distLoss=SemanticSegmentationCosineLoss(num_classes=6)
    for epoch in range(20, opt.num_epochs + opt.num_decay_epochs + 1):
        #
        if epoch<5:
            PrototypeScatST=GetIntCentriod_sourceST.getClusterKmean(model=netStudent,iterFlag=ttest,STFlag=False,confidence_threshold=None)
            print('PrototypeScat', PrototypeScatST.shape)
        else:
            PrototypeScatST = GetIntCentriod_sourceST.getClusterKmean(model=netStudent, iterFlag=ttest, STFlag=True,confidence_threshold=1.0-(epoch)*0.01)
            PrototypeScatT = GetIntCentriod_sourceT.getClusterKmean(model=netStudent, iterFlag=ttest, STFlag=True,
                                                                      confidence_threshold=1.0 - (epoch) * 0.01)
            print('PrototypeScatST', PrototypeScatST.shape)
            print('PrototypeScatT', PrototypeScatT.shape)



        #########################Train####################
        opt.phase = 'train'
        # Load Data
        train_data = train_loader.load_data()
        iter_source = iter(train_data)
        train_data_len = len(train_data)
        len_source_loader = train_data_len
        # Load Data
        Tt_loader = t_loaderDict[cfg.TRAINLOG.DATA_NAMES[cfg.DA.T]]
        Tt_data = Tt_loader.load_data()
        iter_target = iter(Tt_data)
        Tt_data_len = len(Tt_data)
        len_target_loader = Tt_data_len

        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % len(train_loader)
        running_metric.clear()
        netStudent.train()
        netTeacher.eval()
        model_D1.train()
        model_D2.train()

        tbar = tqdm(range(len_target_loader - 1))
        train_data_len = len_target_loader
        record = {'LossT': 0, 'SCET': 0, 'DGANT': 0, 'TCET': 0}
        outSize = 128
        for i in tbar:
            try:
                Sdata = next(iter_source)
            except:
                iter_source = iter(train_data)
            try:
                Tdata = next(iter_target)
            except:
                iter_target = iter(Tt_data)

            epoch_iter += opt.batch_size
            ############## Forward Pass ######################

            images = Variable(Sdata['img_full']).to(DEVICE)
            labelSori = Variable(Sdata['label_full']).to(DEVICE)
            # labels = F.interpolate(labelSori.unsqueeze(1).float(), size=(outSize, outSize), mode='nearest')
            # labelsP = F.interpolate(labelSori.unsqueeze(1).float(), size=(32, 32), mode='nearest')
            labels = F.interpolate(labelSori.unsqueeze(1).float(), size=(outSize, outSize))
            labelsP = F.interpolate(labelSori.unsqueeze(1).float(), size=(32, 32))
            labelS = labels.squeeze(1)  # 尺寸变为 [14, 512, 512]

            target_image = Variable(Tdata['img_full']).to(DEVICE)
            labelT = Variable(Tdata['label_full']).to(DEVICE)
            labelt = F.interpolate(labelT.unsqueeze(1).float(), size=(outSize, outSize)).squeeze(1)


            target_params = Tdata['params']
            target_weak_params = Tdata['weak_params']
            target_imageS = Variable(Tdata['img_strong']).to(DEVICE)
            for param in model_D1.parameters():
                param.requires_grad = False
            for param in model_D2.parameters():
                param.requires_grad = False
            for param in netTeacher.parameters():
                param.requires_grad = False
            optimizer.zero_grad()
            optimizer_D1.zero_grad()
            optimizer_D2.zero_grad()
            with torch.no_grad():
                seg_predS, PS, FeatS = netTeacher.forward(images, DomainLabel=0, ProtoInput=PrototypeScatST)
            seg_predT, PT, FeatT = netStudent.forward(target_image, DomainLabel=0, ProtoInput=PrototypeScatT)

            ########softmax
            SSoft = F.softmax(FeatS['cat'], dim=1)
            SSoft = SSoft.view(SSoft.size(0), SSoft.size(1) // (n_cluster), (n_cluster),
                               SSoft.size(2),
                               SSoft.size(3))
            SSoft = SSoft.max(dim=2)[0]

            TSoft = F.softmax(FeatT['cat'], dim=1)
            TSoft = TSoft.view(TSoft.size(0), TSoft.size(1) // (n_cluster), (n_cluster),
                               TSoft.size(2),
                               TSoft.size(3))
            TSoft = TSoft.max(dim=2)[0]

            D_out1 = model_D1(prob_2_entropy(F.softmax(seg_predT['outUp'], dim=1)))
            print('SSoft',SSoft.shape)
            D_out2_S = model_D2(prob_2_entropy(SSoft))
            D_out2_T = model_D2(prob_2_entropy(TSoft))

            loss_GT = 0.001 * bce_loss(D_out1,
                                       Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).to(
                                           DEVICE)) + \
                      0.001*0.5 * (bce_loss(D_out2_S,
                                       Variable(torch.FloatTensor(D_out2_S.data.size()).fill_(source_label)).to(
                                           DEVICE))+bce_loss(D_out2_S,
                                       Variable(torch.FloatTensor(D_out2_T.data.size()).fill_(source_label)).to(
                                           DEVICE)))
            (loss_GT).backward()
            # train with source
            for param in model_D1.parameters():
                param.requires_grad = True
            for param in model_D2.parameters():
                param.requires_grad = True
            D_out1S = model_D1(prob_2_entropy(F.softmax(seg_predS['outUp'], dim=1)).detach())
            loss_DS = bce_loss(D_out1S,
                               Variable(torch.FloatTensor(D_out1S.data.size()).fill_(source_label)).to(DEVICE))
            record['DGANT'] += loss_DS.item()
            D_out2S = model_D2(prob_2_entropy(F.softmax(seg_predS['outUp'], dim=1).detach()))

            # D_out2S = model_D2(prob_2_entropy(SSoft).detach())
            loss_DS2 = bce_loss(D_out2S,
                                Variable(torch.FloatTensor(D_out2S.data.size()).fill_(source_label)).to(DEVICE))
            record['DGANT'] += loss_DS2.item()
            loss_DS.backward()
            loss_DS2.backward()

            # train with target
            D_out1T = model_D1(prob_2_entropy(F.softmax(seg_predT['outUp'], dim=1)).detach())
            loss_DT = bce_loss(D_out1T,
                               Variable(torch.FloatTensor(D_out1T.data.size()).fill_(target_label)).to(DEVICE))
            record['DGANT'] += loss_DT.item()
            # D_out2T = model_D2(prob_2_entropy(TSoft).detach())
            D_out2T_S = model_D2(prob_2_entropy(SSoft).detach())
            D_out2T_T = model_D2(prob_2_entropy(TSoft).detach())

            loss_DT2 = 0.5*(bce_loss(D_out2T_S,
                                Variable(torch.FloatTensor(D_out2T_S.data.size()).fill_(target_label)).to(DEVICE))\
                       +bce_loss(D_out2T_T,
                                Variable(torch.FloatTensor(D_out2T_T.data.size()).fill_(target_label)).to(DEVICE)))
            record['DGANT'] += loss_DT2.item()
            loss_DT.backward()
            loss_DT2.backward()
            with torch.no_grad():
                # seg_predS, featS, _ = netStudent.forward(images, DomainLabel=0)
                teacher_out, _, teacherFeat = netTeacher.forward(target_image, DomainLabel=0,
                                                                 ProtoInput=PrototypeScatST)
                # emalp = torch.argmax(ema_outOut.detach(), dim=1).unsqueeze(1).float()#([5,6, 128, 128])
                teacher_out_affine = full2weak(teacherFeat['asspF'],
                                               target_weak_params)  # N*256*H*W torch.Size([5, 1, 128, 128])
                teacher_out_affine = label_strong_T(teacher_out_affine, target_params,
                                                    padding=250, scale=1)  # torch.Size([5, 1, 128, 128])
                mask = torch.ones_like(teacher_out_affine, dtype=torch.bool)  # 首先创建一个全True的掩码
                mask[teacher_out_affine == 250] = 0  # 将等于250的位置设置为False
                maskIn = F.interpolate(mask.float(), size=(Osize, Osize), mode='nearest').squeeze(0)
                # adjusted_teacher_probs = teacher_out_affine.clone()  # 复制一份教师网络的输出
                # adjusted_teacher_probs[~mask] = 0  # 将不参与计算的位置设置为0
                teacher_out_affine = F.interpolate(teacher_out_affine, size=(Osize, Osize), mode='bilinear',
                                                   align_corners=True)
                ema_out_teacher = F.softmax(teacher_out_affine, dim=1)

                mask_reduced_prod = torch.prod(maskIn, dim=1, keepdim=True)
            # 3.targetS_out(强变换目标域)-> log_softmax->student
            student_out, _, student_Feat = netStudent.forward(target_imageS, maskParam=mask_reduced_prod, DomainLabel=0,
                                                              ProtoInput=PrototypeScatT)
            # student = F.log_softmax(student_Feat['asspFW'], dim=1)
            # loss_consist = F.kl_div(student, ema_out_teacher, reduction='none')
            loss_consist = kd_loss(student_Feat['asspF'], teacher_out_affine,2,False)

            loss_consist = loss_consist * maskIn
            loss_consist = loss_consist.mean()
            loss_consist.backward()


            optimizer.step()
            optimizer_D1.step()
            optimizer_D2.step()

            # predict
            cd_predSo = torch.argmax(seg_predT['outUp'].detach(), dim=1)
            current_score = running_metric.update(cd_predSo, labelt)
            #############################Discrimination##################################
            # train with source
            TCELoss = torch.tensor([0.0]).to(DEVICE)
            SCELoss = torch.tensor([0.0]).to(DEVICE)
            lossT = TCELoss.item() + SCELoss.item()

            Score = {'LossT': lossT, 'SCET': SCELoss.item(), 'TCET': TCELoss.item()}
            record['LossT'] += lossT

            # current_score = running_metric.confuseMS(pr=cd_predSo.cpu().numpy(), gt=labelS.cpu().numpy())
            Score.update(current_score)
            trainMessage = visualizer.print_current_sorces_Seg(opt.phase, epoch, i, train_data_len, Score)
            # print('trainMessage',trainMessage)
            tbar.set_description(trainMessage)
            if i > 10 and ttest:
                break

        lossAvg = record['LossT'] / i
        CES_lossAvg = record['SCET'] / i
        DGAN_LossAvg = record['DGANT'] / i
        CET_lossAvg = record['TCET'] / i
        train_scores = running_metric.calculate_metrics()
        IterScore = {'LossT': lossAvg, 'SCE': CES_lossAvg, 'DGAN': DGAN_LossAvg, 'TCE': CET_lossAvg}
        IterScore.update(train_scores)
        message = visualizer.print_sorces_Seg(opt.phase, epoch, IterScore)
        # messageT, core_dictT = running_metric.get_scoresT()
        cfg.TRAINLOG.LOGTXT.write(message + '\n')

        exel_out = opt.phase, epoch, lossAvg, CES_lossAvg, DGAN_LossAvg, CET_lossAvg, \
                   IterScore['acc'].item(), IterScore['mIoU'].item(), IterScore['mf1'].item(), \
                   IterScore['preacc'][0].item(), IterScore['preacc'][1].item(), IterScore['preacc'][2].item(), \
                   IterScore['preacc'][3].item(), IterScore['preacc'][4].item(), IterScore['preacc'][5].item(), \
                   IterScore['preIou'][0].item(), IterScore['preIou'][1].item(), IterScore['preIou'][2].item(), \
                   IterScore['preIou'][3].item(), IterScore['preIou'][4].item(), IterScore['preIou'][5].item() \
            # 0,0,0,0
        # core_dictT['accT'], core_dictT['unchgT'], core_dictT['chgT'], core_dictT['mF1T']
        cfg.TRAINLOG.EXCEL_LOG['wsTrain'].append(exel_out)
        # cfg.TRAINLOG.EXCEL_LOG.save('./log/%s/%s/log.xlsx' % (name, time_now))

        figure_train_metrics = train_metrics.set_figure(metric_dict=figure_train_metrics,
                                                        acc=IterScore['acc'].item(),
                                                        miou=IterScore['mIoU'],
                                                        mf1=IterScore['mf1'], preacc=IterScore['preacc'],
                                                        premiou=IterScore['preIou'], lossAvg=lossAvg,
                                                        CES_lossAvg=CES_lossAvg,
                                                        DGAN_LossAvg=DGAN_LossAvg, CET_lossAvg=CET_lossAvg)

        ###############    Val  #####################val
        #Set PrototypeScat
        opt.phase = 'val'
        figure_val_metrics=Vailidate_Val.main(model=netTeacher,figure_metrics=figure_val_metrics,epoch=epoch,cfg=cfg,
                                         iterFlag=ttest,dataloader=train_loader,currentOpt=opt,ProtoInput=PrototypeScatST)
        ###############    Val Similarity  #####################val
        opt.phase = 'val'
        figure_val_Sim_metrics=Vailidate_ValSim.main(model=netTeacher,figure_metrics=figure_val_Sim_metrics,epoch=epoch,cfg=cfg,
                                         iterFlag=ttest,dataloader=train_loader,currentOpt=opt,ProtoInput=PrototypeScatST)
        #####################################Global Proto####################3
        # opt.phase = 'val'
        # ProtoDist_Global_ProtoVal.main(model=netTeacher,inputProto=PrototypeScatST,epoch=epoch,cfg=cfg,
        #                                iterFlag=True,dataloader=train_loader,currentOpt=opt)
        # #####################################Current Proto####################3
        # opt.phase = 'val'
        # ProtoDist_Current_ProtoVal.main(model=netTeacher,inputProto=None,epoch=epoch,cfg=cfg,
        #                                 iterFlag=True,dataloader=train_loader,currentOpt=opt)

        if saveLast:
            if epoch == opt.num_epochs:

                save_str = './log/%s/%s/savemodel/_%d_acc-%.4f_miou-%.4f_mf1-%.4f.pth' \
                           % (name, time_now, epoch + 1, Vailidate_Val.val_scores['acc'], Vailidate_Val.val_scores['mIoU'], Vailidate_Val.val_scores['mf1'])
                tool.save_ckptGANCenter(network=[netStudent,model_D1,model_D2,None],Center=[None,None,None], optimizer=optimizer, save_str=save_str)
        else:
            save_str = './log/%s/%s/savemodel/_%d_acc-%.4f_miou-%.4f_mf1-%.4f.pth' \
                       % (name, time_now, epoch + 1, Vailidate_Val.val_scores['acc'], Vailidate_Val.val_scores['mIoU'], Vailidate_Val.val_scores['mf1'])
            tool.save_ckptGANCenter(network=[netStudent, model_D1, model_D2,None], Center=[None,None,None], optimizer=optimizer,
                                    save_str=save_str)

        save_pickle(figure_train_metrics, "./log/%s/%s/fig_train.pkl" % (name, time_now))
        save_pickle(figure_val_metrics, "./log/%s/%s/fig_val.pkl" % (name, time_now))

        # end of epoch
        iter_end_time = time.time()
        cfg.TRAINLOG.EXCEL_LOG.save('./log/%s/%s/log.xlsx' % (name, time_now))
        if epoch % 1 == 0:

            #######Target Classifier
            for kk in range(len(cfg.TRAINLOG.DATA_NAMES)):
                if kk == opt.s:
                    continue
                print('target:', cfg.TRAINLOG.DATA_NAMES[kk])
                t_loader = t_loaderTestDict[cfg.TRAINLOG.DATA_NAMES[kk]]
                opt.phase = 'target'
                # Set PrototypeScat
                figure_target_metrics = Vailidate_Target.main(model=netStudent, figure_metrics=figure_target_metrics,
                                                        epoch=epoch, cfg=cfg,
                                                        iterFlag=False, dataloader=t_loader, currentOpt=opt,ProtoInput=PrototypeScatT)#00
            #######Target Classifier-ProSimilarity
            for kk in range(len(cfg.TRAINLOG.DATA_NAMES)):
                if kk == opt.s:
                    continue
                print('target:', cfg.TRAINLOG.DATA_NAMES[kk])
                t_loader = t_loaderTestDict[cfg.TRAINLOG.DATA_NAMES[kk]]
                opt.phase = 'target'
                # Set PrototypeScat
                figure_TargetSim_metrics = Vailidate_TargetSim.main(model=netStudent,
                                                              figure_metrics=figure_TargetSim_metrics,
                                                              epoch=epoch, cfg=cfg,
                                                              iterFlag=False, dataloader=t_loader, currentOpt=opt,
                                                              ProtoInput=PrototypeScatT)  # 00Notice whether input Global prptotype
            ############################Global Prototype Target
            for kk in range(len(cfg.TRAINLOG.DATA_NAMES)):
                if kk == opt.s:
                    continue
                print('target-GP:', cfg.TRAINLOG.DATA_NAMES[kk])
                t_loader = t_loaderTestDict[cfg.TRAINLOG.DATA_NAMES[kk]]
                opt.phase = 'target'
                ProtoDist_Global_ProtoTarget.main(model=netStudent, inputProto=PrototypeScatT, epoch=epoch, cfg=cfg,
                                               iterFlag=False, dataloader=t_loader, currentOpt=opt)
            ############################Current Prototype Target
            for kk in range(len(cfg.TRAINLOG.DATA_NAMES)):
                if kk == opt.s:
                    continue
                print('target-CP:', cfg.TRAINLOG.DATA_NAMES[kk])
                t_loader = t_loaderTestDict[cfg.TRAINLOG.DATA_NAMES[kk]]
                opt.phase = 'target'
                ProtoDist_Current_ProtoTarget.main(model=netStudent, inputProto=None, epoch=epoch, cfg=cfg,
                                               iterFlag=False, dataloader=t_loader, currentOpt=opt)
            save_pickle(figure_target_metrics, "./log/%s/%s/fig_T.pkl" % (name, time_now))
            save_pickle(figure_TargetSim_metrics, "./log/%s/%s/fig_T_ema.pkl" % (name, time_now))
            cfg.TRAINLOG.EXCEL_LOG.save('./log/%s/%s/log.xlsx' % (name, time_now))

            #####Draw diff predict
            # for kk in range(len(cfg.TRAINLOG.DATA_NAMES)):
            #     if kk == opt.s:
            #         continue
            #     print('target:', cfg.TRAINLOG.DATA_NAMES[kk])
            #     t_loader = t_loaderTestDict[cfg.TRAINLOG.DATA_NAMES[kk]]
            #     PredictVisualizaion_Target.main(netStudent,inputProto=PrototypeScat,epoch=1,dataloader=t_loader,currentOpt=opt)
            for kk in range(len(cfg.TRAINLOG.DATA_NAMES)):
                if kk == opt.s:
                    continue
                print('target:', cfg.TRAINLOG.DATA_NAMES[kk])
                t_loader = t_loaderTestDict[cfg.TRAINLOG.DATA_NAMES[kk]]
                opt.phase = 'target'
                SimilarityVisualizaion_Target.main(netStudent,inputProto=PrototypeScatT,epoch=epoch,dataloader=t_loader,currentOpt=opt)

    print('================ Training Completed (%s) ================\n' % time.strftime("%c"))
    cfg.TRAINLOG.LOGTXT.write('\n================ Training Completed (%s) ================\n' % time.strftime("%c"))
    plt.rcParams.update(default_rcParams)

    plotFigureSegDA(figure_train_metrics, figure_val_metrics, opt.num_epochs + opt.num_decay_epochs, name, opt.model_type,
                 time_now)
    plotFigureSegDATarget(figure_target_metrics, opt.num_epochs + opt.num_decay_epochs, name, opt.model_type,
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
