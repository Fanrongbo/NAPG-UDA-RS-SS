import torch
import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.datasets import VOCSegmentation
import torchvision.models as models
import torch.nn.functional as F
from model.RSP.resnet import ResNet
# 定义DeepLab v3模型
class DeepLabV3PlusPrototype(nn.Module):
    def __init__(self, num_classes=21,args=None):
        super(DeepLabV3PlusPrototype, self).__init__()
        # self.resnet = models.resnet101(weights = models.ResNet101_Weights.DEFAULT)

        if args.mode == 'imp':
            pretrained = '../pretrain_model/resnet50-19c8e357.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'rsp_120':
            pretrained = '../RS_CLS_finetune/output/resnet_50_224/epoch120/millionAID_224_None/0.0005_0.05_192/resnet/100/ckpt.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'rsp_300':
            self.resnet = ResNet(args)
            # pretrained = '/data/project_frb/SegDA/Segonly/model/RSP/pretrain/rsp-resnet-50-ckpt.pth'
            # pretrained_weights = torch.load(pretrained)
            # self.resnet.load_state_dict(pretrained_weights)
            # pretrained = '../RS_CLS_finetune/output/resnet_50_224/epoch300/millionAID_224_None/0.0005_0.05_128/resnet/100/ckpt.pth'
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode=='nopre':
            self.resnet = models.resnet50(weights=None)
        # self.resnet = models.resnet50(weights=False)
        # # 加载你的本地权重文件
        # weight_path = '/data/project_frb/SegDA/Segonly/model/RSP/pretrain/rsp-resnet-50-ckpt.pth'
        # state_dict = torch.load(weight_path)
        # # 更新模型的状态字典
        # self.resnet.load_state_dict(state_dict)
        self.num_classes=num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        # self.decoder = Decoder(num_classes=num_classes)
        self.classifier = nn.Conv2d(256, num_classes, 1)
        self.prototypeN = 1

        for i in range(num_classes):
            setattr(self, f'query_{i}', nn.Conv2d(256 * self.prototypeN, 128, kernel_size=1))
            setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
    def forward(self, x,DomainLabel=0):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)#[10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)

        # print('assp',x.shape,x3.shape)
        # print('assp_features',assp_features.shape)
        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        with torch.no_grad():
            pseudo_out = self.classifier(assp_features)#([10, 6, 16, 16])
        #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
        #     print('pseudo_out',pseudo_out.shape)
            mask = torch.argmax(pseudo_out, dim=1).unsqueeze(1)
        #
        prototypes = []
        if DomainLabel == 0:
            # 计算每个类别的原型
            for i in range(self.num_classes):
                class_mask = (mask == i).float()
                if class_mask.sum() > 0:  # 确保类别在批次中存在
                    # print('assp_features',assp_features.shape,class_mask.shape)
                    # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                    prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]) + 1e-5)#([10, 256])
                    prototypes.append(prototype)
                else:
                    prototypes.append(zero_prototype)

        # 将原型输入到对应的query层
        total_weight=0
        for i, prototype in enumerate(prototypes):
            if prototype is not None:
                query_output = getattr(self, f'query_{i}')(prototype.unsqueeze(-1).unsqueeze(-1))
                query_output = query_output.view(query_output.size(0), -1,
                                                 query_output.size(1))  # Reshape to [10, 1, 128]
                key_output = getattr(self, f'key_{i}')(assp_features)
                key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]

                similarity = torch.bmm(query_output, key_output)  # Perform batch matrix multiplication
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                               assp_features.size(3))#([10, 1, 16, 16])
                attn_weights = F.softmax(similarity, dim=1)#[10, 1, 16, 16])
                total_weight=total_weight+attn_weights
        # print('total_weight',total_weight.shape)
        assp_weighted=assp_features*(total_weight/self.num_classes)
        x = self.classifier(assp_weighted)  # ([10, 6, 16, 16])

        xup = nn.functional.interpolate(x, size=(h//4, w//4), mode='bilinear', align_corners=True)
        assp = nn.functional.interpolate(assp_features, size=(h//4, w//4), mode='bilinear', align_corners=True)

        # print('prototypes',x.shape)
        concatenated_prototypes = torch.cat([p.unsqueeze(1) for p in prototypes], dim=1)#[10, 6, 256])
        # print('concatenated_prototypes',concatenated_prototypes.shape)

        return xup,concatenated_prototypes,[assp,assp_weighted]
class DeepLabV3PlusMultiPrototypeSingleKey(nn.Module):
    def __init__(self, num_classes=21,args=None,n_cluster=1):
        super(DeepLabV3PlusMultiPrototypeSingleKey, self).__init__()
        # self.resnet = models.resnet101(weights = models.ResNet101_Weights.DEFAULT)

        if args.mode == 'imp':
            pretrained = '../pretrain_model/resnet50-19c8e357.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'rsp_120':
            pretrained = '../RS_CLS_finetune/output/resnet_50_224/epoch120/millionAID_224_None/0.0005_0.05_192/resnet/100/ckpt.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'rsp_300':
            self.resnet = ResNet(args)
            # pretrained = '/data/project_frb/SegDA/Segonly/model/RSP/pretrain/rsp-resnet-50-ckpt.pth'
            # pretrained_weights = torch.load(pretrained)
            # self.resnet.load_state_dict(pretrained_weights)
            # pretrained = '../RS_CLS_finetune/output/resnet_50_224/epoch300/millionAID_224_None/0.0005_0.05_128/resnet/100/ckpt.pth'
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode=='nopre':
            self.resnet = models.resnet50(weights=None)
        # self.resnet = models.resnet50(weights=False)
        # # 加载你的本地权重文件
        # weight_path = '/data/project_frb/SegDA/Segonly/model/RSP/pretrain/rsp-resnet-50-ckpt.pth'
        # state_dict = torch.load(weight_path)
        # # 更新模型的状态字典
        # self.resnet.load_state_dict(state_dict)
        self.num_classes=num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim=128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),)
        #     nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(),
        # nn.ConvTranspose2d(64, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
        # nn.ReLU()
        #     )
        # self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        # self.decoder = Decoder(num_classes=num_classes)
        self.classifier = nn.Conv2d(featDim, num_classes, 1)

        self.n_cluster=n_cluster
        self.prototypeN = 1
        self.key=nn.Conv2d(featDim, 128, kernel_size=1)
        for i in range(num_classes):
            setattr(self, f'query_{i}', nn.Conv2d(featDim * self.prototypeN, 128, kernel_size=1))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
    def forward(self, x,DomainLabel=0,maskParam=None,ProtoInput=None):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)#[10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        with torch.no_grad():
            pseudo_out = self.classifier(assp_features)#([10, 6, 16, 16])
        #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
            mask = torch.argmax(pseudo_out, dim=1).unsqueeze(1)
        prototypes = []
        if DomainLabel == 0:
            # 计算每个类别的原型
            for i in range(self.num_classes):
                class_mask = (mask == i).float()
                if maskParam is not None:
                    # print('class_mask * maskParam',class_mask.shape , maskParam.shape)
                    class_mask = class_mask * maskParam  # Apply maskParam
                if class_mask.sum() > 0:  # 确保类别在批次中存在
                    # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                    prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]) + 1e-5)#([10, 256])
                    for pp in range(self.n_cluster):
                        # print('prototype',prototype.shape)
                        prototypes.append(prototype.unsqueeze(1))
                else:
                    for pp in range(self.n_cluster):
                        prototypes.append(zero_prototype.unsqueeze(1))
        elif DomainLabel==1 and ProtoInput!=None:
            prototypes=ProtoInput
        # print('lem',len(prototypes))
        # 将原型输入到对应的query层
        total_weight=0
        key_output = self.key(assp_features)
        key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
        # print('key_output',key_output.shape)
        prototypesOut=[]
        similarityList=[]
        for i, prototype in enumerate(prototypes):
            if prototype is not None:
                # print('prototype',prototype.shape)
                query_output = getattr(self, f'query_{i//self.n_cluster}')(prototype[:,0,:].unsqueeze(-1).unsqueeze(-1))
                query_output = query_output.view(query_output.size(0), -1,
                                                 query_output.size(1))  # Reshape to [10, 1, 128]
                prototypesOut.append(query_output.squeeze(1))
                similarity = torch.bmm(query_output, key_output)  # Perform batch matrix multiplication#torch.Size([10, 1, 256])
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                               assp_features.size(3))#([10, 1, 16, 16])
                similarityList.append(similarity)

        similarityCat=torch.cat(similarityList,dim=1)
        # print('similarityCat',similarityCat.shape)
        similarityWeiht=F.softmax(similarityCat/similarityCat.size(1)*self.n_cluster,dim=1)
        # print('similarityWeiht',similarityWeiht.shape)
        similarityWeihtMean=similarityWeiht.mean(dim=1)

        similarityWeihtMax,_=torch.max(similarityWeiht,dim=1)
        # print('similarityWeiht',similarityWeiht.shape,similarityWeihtMax.shape)

        assp_weighted=assp_features*similarityWeihtMean.unsqueeze(1)

        x = self.classifier(assp_weighted)  # ([10, 6, 16, 16])
        # print(x.shape)
        xup = nn.functional.interpolate(x, size=(h//4, w//4), mode='bilinear', align_corners=True)
        # assp = nn.functional.interpolate(assp_features, size=(h//4, w//4), mode='bilinear', align_corners=True)

        # print('prototypes',x.shape)
        concatenated_prototypes = torch.cat([p.unsqueeze(1) for p in prototypesOut], dim=1)#[10, 6, 256])
        # print('concatenated_prototypes',concatenated_prototypes.shape)
        # print('concatenated_prototypes',concatenated_prototypes.shape)
        return {'out':x,'outUp':xup},{'CurrentPorotype':None,'GetProto':concatenated_prototypes,'query':prototypes},{'asspF':assp_features,'asspFW':assp_weighted,'cat':similarityCat,
                                                                                                                     'Weight':[similarityCat,similarityWeihtMean,similarityWeihtMax]}

        # return xup,concatenated_prototypes,[assp_features,assp_weighted,similarityCat,similarityWeihtMean,similarityWeihtMax]

class DeepLabV3PlusMultiPrototypeSingleKeyBN(nn.Module):
    def __init__(self, num_classes=21,args=None,n_cluster=1):
        super(DeepLabV3PlusMultiPrototypeSingleKeyBN, self).__init__()
        # self.resnet = models.resnet101(weights = models.ResNet101_Weights.DEFAULT)

        if args.mode == 'imp':
            pretrained = '../pretrain_model/resnet50-19c8e357.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'rsp_120':
            pretrained = '../RS_CLS_finetune/output/resnet_50_224/epoch120/millionAID_224_None/0.0005_0.05_192/resnet/100/ckpt.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'rsp_300':
            self.resnet = ResNet(args)
            # pretrained = '/data/project_frb/SegDA/Segonly/model/RSP/pretrain/rsp-resnet-50-ckpt.pth'
            # pretrained_weights = torch.load(pretrained)
            # self.resnet.load_state_dict(pretrained_weights)
            # pretrained = '../RS_CLS_finetune/output/resnet_50_224/epoch300/millionAID_224_None/0.0005_0.05_128/resnet/100/ckpt.pth'
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode=='nopre':
            self.resnet = models.resnet50(weights=None)
        # self.resnet = models.resnet50(weights=False)
        # # 加载你的本地权重文件
        # weight_path = '/data/project_frb/SegDA/Segonly/model/RSP/pretrain/rsp-resnet-50-ckpt.pth'
        # state_dict = torch.load(weight_path)
        # # 更新模型的状态字典
        # self.resnet.load_state_dict(state_dict)
        self.num_classes=num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim=128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),)
        #     nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(),
        # nn.ConvTranspose2d(64, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
        # nn.ReLU()
        #     )
        # self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        # self.decoder = Decoder(num_classes=num_classes)
        self.classifier = nn.Conv2d(featDim, num_classes, 1)

        self.n_cluster=n_cluster
        self.prototypeN = 1
        self.key=nn.Conv2d(featDim, 128, kernel_size=1)
        self.bn_key=nn.BatchNorm2d(128)
        for i in range(num_classes):
            setattr(self, f'query_{i}', nn.Conv2d(featDim * self.prototypeN, 128, kernel_size=1))
            setattr(self, f'bn_query_{i}', nn.BatchNorm2d(128))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
    def forward(self, x,DomainLabel=0,maskParam=None,ProtoInput=None):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)#[10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        with torch.no_grad():
            pseudo_out = self.classifier(assp_features)#([10, 6, 16, 16])
        #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
            mask = torch.argmax(pseudo_out, dim=1).unsqueeze(1)
        prototypes = []
        if DomainLabel == 0:
            # 计算每个类别的原型
            for i in range(self.num_classes):
                class_mask = (mask == i).float()
                if maskParam is not None:
                    # print('class_mask * maskParam',class_mask.shape , maskParam.shape)
                    class_mask = class_mask * maskParam  # Apply maskParam
                if class_mask.sum() > 0:  # 确保类别在批次中存在
                    # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                    prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]) + 1e-5)#([10, 256])
                    for pp in range(self.n_cluster):
                        # print('prototype',prototype.shape)
                        prototypes.append(prototype.unsqueeze(1))
                else:
                    for pp in range(self.n_cluster):
                        prototypes.append(zero_prototype.unsqueeze(1))
        elif DomainLabel==1 and ProtoInput!=None:
            prototypes=ProtoInput
        # print('lem',len(prototypes))
        # 将原型输入到对应的query层
        total_weight=0
        key_output = self.bn_key(self.key(assp_features))

        key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
        # print('key_output',key_output.shape)
        prototypesOut=[]
        similarityList=[]
        for i, prototype in enumerate(prototypes):
            if prototype is not None:
                # print('prototype',prototype.shape)
                query_output = getattr(self, f'query_{i//self.n_cluster}')(prototype[:,0,:].unsqueeze(-1).unsqueeze(-1))
                query_output = getattr(self, f'bn_query_{i//self.n_cluster}')(query_output)
                query_output = query_output.view(query_output.size(0), -1,
                                                 query_output.size(1))  # Reshape to [10, 1, 128]
                prototypesOut.append(query_output.squeeze(1))

                similarity = torch.bmm(query_output, key_output)  # Perform batch matrix multiplication#torch.Size([10, 1, 256])

                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                               assp_features.size(3))#([10, 1, 16, 16])
                similarityList.append(similarity)

        similarityCat=torch.cat(similarityList,dim=1)
        # print('similarityCat',similarityCat.shape)
        similarityWeiht=F.softmax(similarityCat/similarityCat.size(1)*self.n_cluster,dim=1)
        similarityWeiht=similarityWeiht.mean(dim=1)
        similarityWeihtmax,_=torch.max(similarityWeiht,dim=1)
        assp_weighted=assp_features*similarityWeiht.unsqueeze(1)

        x = self.classifier(assp_weighted)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h//4, w//4), mode='bilinear', align_corners=True)
        # assp = nn.functional.interpolate(assp_features, size=(h//4, w//4), mode='bilinear', align_corners=True)

        concatenated_prototypes = torch.cat([p.unsqueeze(1) for p in prototypesOut], dim=1)#[10, 6, 256])

        return xup,concatenated_prototypes,[assp_features,assp_weighted,similarityWeiht,similarityWeihtmax]
class DeepLabV3PlusPrototypeSingleKey(nn.Module):
    def __init__(self, num_classes=21,args=None):
        super(DeepLabV3PlusPrototypeSingleKey, self).__init__()
        # self.resnet = models.resnet101(weights = models.ResNet101_Weights.DEFAULT)

        if args.mode == 'imp':
            pretrained = '../pretrain_model/resnet50-19c8e357.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'rsp_120':
            pretrained = '../RS_CLS_finetune/output/resnet_50_224/epoch120/millionAID_224_None/0.0005_0.05_192/resnet/100/ckpt.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'rsp_300':
            self.resnet = ResNet(args)
            # pretrained = '/data/project_frb/SegDA/Segonly/model/RSP/pretrain/rsp-resnet-50-ckpt.pth'
            # pretrained_weights = torch.load(pretrained)
            # self.resnet.load_state_dict(pretrained_weights)
            # pretrained = '../RS_CLS_finetune/output/resnet_50_224/epoch300/millionAID_224_None/0.0005_0.05_128/resnet/100/ckpt.pth'
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode=='nopre':
            self.resnet = models.resnet50(weights=None)
        # self.resnet = models.resnet50(weights=False)
        # # 加载你的本地权重文件
        # weight_path = '/data/project_frb/SegDA/Segonly/model/RSP/pretrain/rsp-resnet-50-ckpt.pth'
        # state_dict = torch.load(weight_path)
        # # 更新模型的状态字典
        # self.resnet.load_state_dict(state_dict)
        self.num_classes=num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim=128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),)
        #     nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(),
        # nn.ConvTranspose2d(64, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
        # nn.ReLU()
        #     )
        # self.aspp = ASPPModule(in_channels=2048, out_channels=256)

        # self.decoder = Decoder(num_classes=num_classes)
        self.classifier = nn.Conv2d(featDim, num_classes, 1)


        self.prototypeN = 1
        self.key=nn.Conv2d(featDim, 128, kernel_size=1)
        for i in range(num_classes):
            setattr(self, f'query_{i}', nn.Conv2d(featDim * self.prototypeN, 128, kernel_size=1))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
    def forward(self, x,DomainLabel=0,maskParam=None,ProtoInput=None):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)#[10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)

        # print('assp',x.shape,x3.shape)
        # print('assp_features',assp_features.shape)
        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        with torch.no_grad():
            pseudo_out = self.classifier(assp_features)#([10, 6, 16, 16])
        #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
        #     print('pseudo_out',pseudo_out.shape)
            mask = torch.argmax(pseudo_out, dim=1).unsqueeze(1)
        #
        prototypes = []
        if DomainLabel == 0:
            # 计算每个类别的原型
            for i in range(self.num_classes):
                class_mask = (mask == i).float()
                if maskParam is not None:
                    # print('class_mask * maskParam',class_mask.shape , maskParam.shape)
                    class_mask = class_mask * maskParam  # Apply maskParam

                if class_mask.sum() > 0:  # 确保类别在批次中存在
                    # print('assp_features',assp_features.shape,class_mask.shape)
                    # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                    prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]) + 1e-5)#([10, 256])
                    prototypes.append(prototype)
                else:
                    prototypes.append(zero_prototype)
        elif DomainLabel==1 and ProtoInput!=None:
            prototypes=ProtoInput
        # 将原型输入到对应的query层
        total_weight=0
        key_output = self.key(assp_features)
        key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
        # print('key_output',key_output.shape)
        prototypesOut=[]
        similarityList=[]
        for i, prototype in enumerate(prototypes):
            if prototype is not None:
                query_output = getattr(self, f'query_{i}')(prototype.unsqueeze(-1).unsqueeze(-1))
                query_output = query_output.view(query_output.size(0), -1,
                                                 query_output.size(1))  # Reshape to [10, 1, 128]
                prototypesOut.append(query_output.squeeze(1))
                similarity = torch.bmm(query_output, key_output)  # Perform batch matrix multiplication#torch.Size([10, 1, 256])
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                               assp_features.size(3))#([10, 1, 16, 16])
                # print('key_output',key_output.shape,query_output.shape,similarity.shape)
                similarityList.append(similarity)
        similarityCat=torch.cat(similarityList,dim=1)

        similarityWeiht=F.softmax(similarityCat/similarityCat.size(1),dim=1)
        similarityWeiht=similarityWeiht.mean(dim=1)
        # similarityWeiht,_=torch.max(similarityWeiht,dim=1)
        assp_weighted=assp_features*similarityWeiht.unsqueeze(1)

        x = self.classifier(assp_weighted)  # ([10, 6, 16, 16])
        # print(x.shape)
        xup = nn.functional.interpolate(x, size=(h//4, w//4), mode='bilinear', align_corners=True)
        # assp = nn.functional.interpolate(assp_features, size=(h//4, w//4), mode='bilinear', align_corners=True)

        # print('prototypes',x.shape)
        concatenated_prototypes = torch.cat([p.unsqueeze(1) for p in prototypesOut], dim=1)#[10, 6, 256])
        # print('concatenated_prototypes',concatenated_prototypes.shape)

        return xup,concatenated_prototypes,[assp_features,assp_weighted]
class DeepLabV3PlusPrototypeMultiKey(nn.Module):
    def __init__(self, num_classes=21,args=None):
        super(DeepLabV3PlusPrototypeMultiKey, self).__init__()
        # self.resnet = models.resnet101(weights = models.ResNet101_Weights.DEFAULT)

        if args.mode == 'imp':
            pretrained = '../pretrain_model/resnet50-19c8e357.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'rsp_120':
            pretrained = '../RS_CLS_finetune/output/resnet_50_224/epoch120/millionAID_224_None/0.0005_0.05_192/resnet/100/ckpt.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'rsp_300':
            self.resnet = ResNet(args)
            # pretrained = '/data/project_frb/SegDA/Segonly/model/RSP/pretrain/rsp-resnet-50-ckpt.pth'
            # pretrained_weights = torch.load(pretrained)
            # self.resnet.load_state_dict(pretrained_weights)
            # pretrained = '../RS_CLS_finetune/output/resnet_50_224/epoch300/millionAID_224_None/0.0005_0.05_128/resnet/100/ckpt.pth'
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode=='nopre':
            self.resnet = models.resnet50(weights=None)
        # self.resnet = models.resnet50(weights=False)
        # # 加载你的本地权重文件
        # weight_path = '/data/project_frb/SegDA/Segonly/model/RSP/pretrain/rsp-resnet-50-ckpt.pth'
        # state_dict = torch.load(weight_path)
        # # 更新模型的状态字典
        # self.resnet.load_state_dict(state_dict)
        self.num_classes=num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        # self.decoder = Decoder(num_classes=num_classes)
        self.classifier = nn.Conv2d(256, num_classes, 1)
        self.prototypeN = 1
        # self.key=nn.Conv2d(256, 128, kernel_size=1)
        for i in range(num_classes):
            setattr(self, f'query_{i}', nn.Conv2d(256 * self.prototypeN, 128, kernel_size=1))
            setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
    def forward(self, x,DomainLabel=0,maskParam=None,ProtoInput=None):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)#[10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)

        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        with torch.no_grad():
            pseudo_out = self.classifier(assp_features)#([10, 6, 16, 16])
        #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
            mask = torch.argmax(pseudo_out, dim=1).unsqueeze(1)
        prototypes = []
        if DomainLabel == 0:
            # 计算每个类别的原型
            for i in range(self.num_classes):
                class_mask = (mask == i).float()
                if maskParam is not None:
                    class_mask = class_mask * maskParam  # Apply maskParam
                if class_mask.sum() > 0:  # 确保类别在批次中存在
                    # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                    prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]) + 1e-5)#([10, 256])
                    prototypes.append(prototype)
                else:
                    prototypes.append(zero_prototype)
        elif DomainLabel==1 and ProtoInput!=None:
            prototypes=ProtoInput
        # 将原型输入到对应的query层
        total_weight=0
        # key_output = self.key(assp_features)
        # key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
        prototypesOut = []
        similarityList = []
        for i, prototype in enumerate(prototypes):
            if prototype is not None:
                query_output = getattr(self, f'query_{i}')(prototype.unsqueeze(-1).unsqueeze(-1))
                query_output = query_output.view(query_output.size(0), -1,
                                                 query_output.size(1))  # Reshape to [10, 1, 128]
                prototypesOut.append(query_output.squeeze(1))
                key_output = getattr(self, f'key_{i}')(assp_features)
                key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
                similarity = torch.bmm(query_output,
                                       key_output)  # Perform batch matrix multiplication#torch.Size([10, 1, 256])
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                             assp_features.size(3))  # ([10, 1, 16, 16])
                similarityList.append(similarity)
        similarityCat = torch.cat(similarityList, dim=1)

        similarityWeiht = F.softmax(similarityCat, dim=1)
        similarityWeiht, _ = torch.max(similarityWeiht, dim=1)
        assp_weighted = assp_features * similarityWeiht.unsqueeze(1)
        x = self.classifier(assp_weighted)  # ([10, 6, 16, 16])

        xup = nn.functional.interpolate(x, size=(h//4, w//4), mode='bilinear', align_corners=True)
        # assp = nn.functional.interpolate(assp_features, size=(h//4, w//4), mode='bilinear', align_corners=True)

        # print('prototypes',x.shape)
        concatenated_prototypes = torch.cat([p.unsqueeze(1) for p in prototypesOut], dim=1)#[10, 6, 256])
        # print('concatenated_prototypes',concatenated_prototypes.shape)

        return xup,concatenated_prototypes,[assp_features,assp_weighted]
class DeepLabV3PlusPrototypeIter(nn.Module):
    def __init__(self, num_classes=21,args=None):
        super(DeepLabV3PlusPrototypeIter, self).__init__()
        # self.resnet = models.resnet101(weights = models.ResNet101_Weights.DEFAULT)

        if args.mode == 'imp':
            pretrained = '../pretrain_model/resnet50-19c8e357.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'rsp_120':
            pretrained = '../RS_CLS_finetune/output/resnet_50_224/epoch120/millionAID_224_None/0.0005_0.05_192/resnet/100/ckpt.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'rsp_300':
            self.resnet = ResNet(args)
            # pretrained = '/data/project_frb/SegDA/Segonly/model/RSP/pretrain/rsp-resnet-50-ckpt.pth'
            # pretrained_weights = torch.load(pretrained)
            # self.resnet.load_state_dict(pretrained_weights)
            # pretrained = '../RS_CLS_finetune/output/resnet_50_224/epoch300/millionAID_224_None/0.0005_0.05_128/resnet/100/ckpt.pth'
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode=='nopre':
            self.resnet = models.resnet50(weights=None)
        # self.resnet = models.resnet50(weights=False)
        # # 加载你的本地权重文件
        # weight_path = '/data/project_frb/SegDA/Segonly/model/RSP/pretrain/rsp-resnet-50-ckpt.pth'
        # state_dict = torch.load(weight_path)
        # # 更新模型的状态字典
        # self.resnet.load_state_dict(state_dict)
        self.num_classes=num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        # self.decoder = Decoder(num_classes=num_classes)
        self.classifier = nn.Conv2d(256, num_classes, 1)
        self.prototypeN = 1
        self.key=nn.Conv2d(256, 128, kernel_size=1)
        for i in range(num_classes):
            setattr(self, f'query_{i}', nn.Conv2d(256 * self.prototypeN, 128, kernel_size=1))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
    def forward(self, x,DomainLabel=0):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)#[10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)

        # print('assp',x.shape,x3.shape)
        # print('assp_features',assp_features.shape)
        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        with torch.no_grad():
            pseudo_out = self.classifier(assp_features)#([10, 6, 16, 16])
        #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
        #     print('pseudo_out',pseudo_out.shape)
            mask = torch.argmax(pseudo_out, dim=1).unsqueeze(1)
        #
        prototypes = []
        if DomainLabel == 0:
            # 计算每个类别的原型
            for i in range(self.num_classes):
                class_mask = (mask == i).float()
                if class_mask.sum() > 0:  # 确保类别在批次中存在
                    # print('assp_features',assp_features.shape,class_mask.shape)
                    # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                    prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]) + 1e-5)#([10, 256])
                    prototypes.append(prototype)
                else:
                    prototypes.append(zero_prototype)

        # 将原型输入到对应的query层
        total_weight=0
        key_output = self.key(assp_features)
        key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
        queryProto=[]
        for i, prototype in enumerate(prototypes):
            if prototype is not None:
                query_output = getattr(self, f'query_{i}')(prototype.unsqueeze(-1).unsqueeze(-1))
                if prototype !=zero_prototype:
                    queryProto.append(query_output)
                else:
                    queryProto.append(torch.zeros_like(query_output,device=query_output.device))

                query_output = query_output.view(query_output.size(0), -1,
                                                 query_output.size(1))  # Reshape to [10, 1, 128]
                # key_output = getattr(self, f'key_{i}')(assp_features)

                similarity = torch.bmm(query_output, key_output)  # Perform batch matrix multiplication
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                               assp_features.size(3))#([10, 1, 16, 16])
                attn_weights = F.softmax(similarity, dim=1)#[10, 1, 16, 16])
                total_weight=total_weight+attn_weights
        # print('total_weight',total_weight.shape)
        assp_weighted=assp_features*(total_weight/self.num_classes)
        x = self.classifier(assp_weighted)  # ([10, 6, 16, 16])

        xup = nn.functional.interpolate(x, size=(h//4, w//4), mode='bilinear', align_corners=True)
        assp = nn.functional.interpolate(assp_features, size=(h//4, w//4), mode='bilinear', align_corners=True)

        # print('prototypes',x.shape)
        # concatenated_prototypes = torch.cat([p.unsqueeze(1) for p in prototypes], dim=1)#[10, 6, 256])
        concatenated_prototypes = torch.cat([p.unsqueeze(1) for p in queryProto], dim=1)#[10, 6, 256])

        # print('concatenated_prototypes',concatenated_prototypes.shape)

        return xup,concatenated_prototypes
class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) Module
    """

    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super(ASPPModule, self).__init__()
        modules = []
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        self.convs = nn.ModuleList(modules)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(len(atrous_rates) * out_channels + out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res.append(F.interpolate(self.pool(x), size=x.shape[2:], mode='bilinear', align_corners=False))
        res = torch.cat(res, dim=1)
        return self.project(res)

