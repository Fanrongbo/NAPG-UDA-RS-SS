import torch
import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.datasets import VOCSegmentation
import torchvision.models as models
import torch.nn.functional as F
from model.RSP.resnet import ResNet
from util.ProtoDistValidate import *
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

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
                nn.ReLU(inplace=False)
            ))

        self.convs = nn.ModuleList(modules)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        self.project = nn.Sequential(
            nn.Conv2d(len(atrous_rates) * out_channels + out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res.append(F.interpolate(self.pool(x), size=x.shape[2:], mode='bilinear', align_corners=False))
        res = torch.cat(res, dim=1)
        return self.project(res)
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=21,args=None,n_cluster=1):
        super(DeepLabV3Plus, self).__init__()
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
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode=='nopre':
            self.resnet = models.resnet50(weights=None)
        self.num_classes=num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim=128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),)
        self.classifier = nn.Conv2d(featDim, num_classes, 1)
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

        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h//4, w//4), mode='bilinear', align_corners=True)

        return {'out': x,'outUp': xup},{'CurrentPorotype':None,'GetProto': None,'query': None},\
               {'asspF':assp_features,'asspFW':None,'cat':None,'Weight': [None,None]}


class DeepLabV3PlusSim(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSim, self).__init__()
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

        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode == 'nopre':
            self.resnet = models.resnet50(weights=None)
        self.num_classes = num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), )
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
        self.n_cluster = n_cluster
        self.prototypeN = 1
        self.key = nn.Conv2d(featDim, 128, kernel_size=1)
        for i in range(num_classes):
            setattr(self, f'query_{i}', nn.Conv2d(featDim * self.prototypeN, 128, kernel_size=1))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
        self.normP=nn.LayerNorm(1024)
    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None,getPFlag=False):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)  # [10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        if getPFlag:
            return assp_features
        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        prototypes = []
        # if DomainLabel == 0:
        if ProtoInput == None:
            with torch.no_grad():
                pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
                #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
                mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
            # 计算每个类别的原型

            for i in range(self.num_classes):
                class_mask = (mask == i).float()
                if maskParam is not None:
                    class_mask = class_mask * maskParam  # Apply maskParam
                if class_mask.sum() > 0:  # 确保类别在批次中存在
                    # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                    prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                                class_mask.sum(dim=[2, 3]) + 1e-5)  # ([10, 256])
                    prototypes.append(prototype.unsqueeze(1))
                else:
                    # for pp in range(self.n_cluster):
                    prototypes.append(zero_prototype.unsqueeze(1))
            prototypes = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze(
                -1)  # prototypes torch.Size([10, 6, 128, 1, 1])
        # elif DomainLabel==1 and ProtoInput!=None:
        elif ProtoInput != None:
            prototypes = ProtoInput
            # prototypes = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze(-1)
        query_outputList = []
        # prototypesOut = []
        similarityList = []
        key_output = assp_features
        key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
        for ii in range(prototypes.size(1)):
            prototype = prototypes[:, ii]
            if prototype is not None:
                # query_output = getattr(self, f'query_{i//(len(prototypes)//6)}')(prototype[:,0,:].unsqueeze(-1).unsqueeze(-1))
                query_output = prototype.view(prototype.size(0), -1, prototype.size(1))  # Reshape to [10, 1, 128]
                # prototypesOut.append(query_output.squeeze(1))
                # similarity = torch.abs(
                #     torch.bmm(query_output, key_output) / 128)  # Perform batch matrix multiplication#([10, 1, 1024])
                # Normalize along the channel dimension
                # query_output = F.normalize(query_output, p=2,dim=2)
                # key_output = F.normalize(key_output, p=2, dim=2)
                # Perform batch matrix multiplication#([10, 1, 1024])
                similarity = self.normP(torch.matmul(query_output, key_output))
                # similarity = torch.matmul(query_output, key_output)
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                             assp_features.size(3))  # ([10, 1, 16, 16])
                similarityList.append(similarity)
                query_outputList.append(query_output)

        query_outputcat = torch.cat(query_outputList, dim=1).unsqueeze(-1).unsqueeze(-1)
        similarityCat = torch.cat(similarityList, dim=1)
        similarityWeiht = F.softmax(similarityCat, dim=1)
        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        assp_weighted = assp_features * similarityWeihtMax.unsqueeze(1)
        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': xup}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,
                'Weight': [similarityWeiht, similarityWeihtMax]}
class DeepLabV3PlusSimCat(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimCat, self).__init__()
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

        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode == 'nopre':
            self.resnet = models.resnet50(weights=None)
        self.num_classes = num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), )
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
        self.n_cluster = n_cluster
        self.prototypeN = 1
        self.key = nn.Conv2d(featDim, 128, kernel_size=1)
        for i in range(num_classes):
            setattr(self, f'query_{i}', nn.Conv2d(featDim * self.prototypeN, 128, kernel_size=1))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
        self.normP=nn.LayerNorm(1024)
    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None,getPFlag=False):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)  # [10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        if getPFlag:
            return assp_features

        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        prototypes = []
        # if DomainLabel == 0:
        # if ProtoInput == None:
        with torch.no_grad():
            pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
            #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
            mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
        # 计算每个类别的原型

        for i in range(self.num_classes):
            class_mask = (mask == i).float()
            if maskParam is not None:
                class_mask = class_mask * maskParam  # Apply maskParam
            if class_mask.sum() > 0:  # 确保类别在批次中存在
                # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                            class_mask.sum(dim=[2, 3]) + 1e-5)  # ([10, 256])
                prototypes.append(prototype.unsqueeze(1))
            else:
                # for pp in range(self.n_cluster):
                prototypes.append(zero_prototype.unsqueeze(1))
            # print(ProtoInput.shape,ProtoInput[:,i*self.n_cluster:(i+1)*self.n_cluster,:,0,0].shape,prototype.unsqueeze(1).shape)
            prototypes.append(ProtoInput[:,i*self.n_cluster:(i+1)*self.n_cluster,:,0,0])
        prototypes = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze(-1)  # prototypes t([10, 6, 128, 1, 1])
    # elif DomainLabel==1 and ProtoInput!=None:
    #     elif ProtoInput != None:
    #         prototypes = ProtoInput
            # prototypes = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze(-1)
        query_outputList = []
        # prototypesOut = []
        similarityList = []
        key_output = assp_features
        key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
        for ii in range(prototypes.size(1)):
            prototype = prototypes[:, ii]
            if prototype is not None:
                # query_output = getattr(self, f'query_{i//(len(prototypes)//6)}')(prototype[:,0,:].unsqueeze(-1).unsqueeze(-1))
                query_output = prototype.view(prototype.size(0), -1, prototype.size(1))  # Reshape to [10, 1, 128]
                # prototypesOut.append(query_output.squeeze(1))
                # similarity = torch.abs(
                #     torch.bmm(query_output, key_output) / 128)  # Perform batch matrix multiplication#([10, 1, 1024])
                # Normalize along the channel dimension
                # query_output = F.normalize(query_output, p=2,dim=2)
                # key_output = F.normalize(key_output, p=2, dim=2)
                # Perform batch matrix multiplication#([10, 1, 1024])
                similarity = self.normP(torch.matmul(query_output, key_output))
                # similarity = torch.matmul(query_output, key_output)
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                             assp_features.size(3))  # ([10, 1, 16, 16])
                similarityList.append(similarity)
                query_outputList.append(query_output)

        query_outputcat = torch.cat(query_outputList, dim=1).unsqueeze(-1).unsqueeze(-1)
        similarityCat = torch.cat(similarityList, dim=1)
        # print('similarityCat',similarityCat.shape)
        similarityWeiht = F.softmax(similarityCat, dim=1)
        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        assp_weighted = assp_features * similarityWeihtMax.unsqueeze(1)
        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': xup}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,
                'Weight': [similarityWeiht, similarityWeihtMax]}

class DeepLabV3PlusSimCatLinear(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimCatLinear, self).__init__()
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
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode == 'nopre':
            self.resnet = models.resnet50(weights=None)
        self.num_classes = num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), )

        self.classifier = nn.Conv2d(featDim, num_classes, 1)
        self.n_cluster = n_cluster
        self.prototypeN = 1
        # self.key = nn.Conv2d(featDim, 128, kernel_size=1)
        for i in range(num_classes):
            # setattr(self, f'query_{i}', nn.Conv2d(featDim * self.prototypeN, 128, kernel_size=1))
            setattr(self, f'a_{i}', nn.Parameter(torch.ones(1), requires_grad=True))
            setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1), requires_grad=True))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
        self.normP=nn.LayerNorm(1024)
    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None,getPFlag=False):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)  # [10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        if getPFlag:
            return assp_features
        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        prototypes = []
        # if DomainLabel == 0:
        # if ProtoInput == None:
        with torch.no_grad():
            pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
            #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
            mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
            # 计算每个类别的原型
            for i in range(self.num_classes):
                class_mask = (mask == i).float()
                if maskParam is not None:
                    class_mask = class_mask * maskParam  # Apply maskParam
                if class_mask.sum() > 0:  # 确保类别在批次中存在
                    prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                                class_mask.sum(dim=[2, 3]) + 1e-5).detach()  # ([10, 256])
                    prototypes.append(prototype.unsqueeze(1))
                else:
                    prototypes.append(zero_prototype.unsqueeze(1))
                prototypes.append(ProtoInput[:,i*self.n_cluster:(i+1)*self.n_cluster,:,0,0])
            prototypes = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze(-1)  # prototypes t([10, 6, 128, 1, 1])
    # elif DomainLabel==1 and ProtoInput!=None:
    #     elif ProtoInput != None:
    #         prototypes = ProtoInput
            # prototypes = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze(-1)
        query_outputList = []
        # prototypesOut = []
        similarityList = []
        key_output = assp_features
        key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
        for ii in range(prototypes.size(1)):
            prototype = prototypes[:, ii]
            if prototype is not None:
                #######key
                # key_output=getattr(self, f'a_{ii//(prototypes.size(1)//6)}') * key_output + getattr(self, f'b_{ii//((prototypes.size(1)//6))}')
                # query_output = prototype.view(prototype.size(0), -1, prototype.size(1))  # Reshape to [10, 1, 128]
                ########query
                query_output = getattr(self, f'a_{ii // (prototypes.size(1) // 6)}') * prototype + getattr(self,
                                                                                                           f'b_{ii // ((prototypes.size(1) // 6))}')
                query_output = prototype.view(query_output.size(0), -1, query_output.size(1))  # Reshape to [10, 1, 128]

                # prototypesOut.append(query_output.squeeze(1))
                # similarity = torch.abs(
                #     torch.bmm(query_output, key_output) / 128)  # Perform batch matrix multiplication#([10, 1, 1024])
                # Normalize along the channel dimension
                # query_output = F.normalize(query_output, p=2,dim=2)
                # key_output = F.normalize(key_output, p=2, dim=2)
                # Perform batch matrix multiplication#([10, 1, 1024])
                similarity = self.normP(torch.matmul(query_output, key_output))
                # similarity = torch.matmul(query_output, key_output)
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                             assp_features.size(3))  # ([10, 1, 16, 16])
                similarityList.append(similarity)
                query_outputList.append(query_output)

        query_outputcat = torch.cat(query_outputList, dim=1).unsqueeze(-1).unsqueeze(-1)
        similarityCat = torch.cat(similarityList, dim=1)
        # print('similarityCat',similarityCat.shape)
        similarityWeiht = F.softmax(similarityCat, dim=1)
        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        assp_weighted = assp_features * similarityWeihtMax.unsqueeze(1)
        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': xup}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,
                'Weight': [similarityWeiht, similarityWeihtMax]}
class DeepLabV3PlusSimGlobalLinear(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimGlobalLinear, self).__init__()
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
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode == 'nopre':
            self.resnet = models.resnet50(weights=None)
        self.num_classes = num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), )

        self.classifier = nn.Conv2d(featDim, num_classes, 1)

        self.n_cluster = n_cluster
        self.prototypeN = 1
        # self.key = nn.Conv2d(featDim, featDim, kernel_size=3, padding=1)
        for i in range(num_classes):
            # setattr(self, f'query_{i}', nn.Conv2d(featDim * self.prototypeN, 128, kernel_size=1))
            setattr(self, f'a_{i}', nn.Parameter(torch.ones(1), requires_grad=True))
            setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1), requires_grad=True))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
        self.normP=nn.LayerNorm(1024)
    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None,getPFlag=False):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)  # [10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        if getPFlag:
            return assp_features
        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        prototypes = []
        if ProtoInput == None:
            with torch.no_grad():
                pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
                #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
                mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
            # 计算每个类别的原型
            for i in range(self.num_classes):
                class_mask = (mask == i).float()
                if maskParam is not None:
                    class_mask = class_mask * maskParam  # Apply maskParam
                if class_mask.sum() > 0:  # 确保类别在批次中存在
                    # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                    prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                                class_mask.sum(dim=[2, 3]) + 1e-5)  # ([10, 256])
                    prototypes.append(prototype.unsqueeze(1))
                else:
                    # for pp in range(self.n_cluster):
                    prototypes.append(zero_prototype.unsqueeze(1))
            prototypes = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze(
                -1)  # prototypes torch.Size([10, 6, 128, 1, 1])
        # elif DomainLabel==1 and ProtoInput!=None:
        elif ProtoInput != None:
            prototypes = ProtoInput
        query_outputList = []
        # prototypesOut = []
        similarityList = []
        # key_output = self.key(assp_features)
        key_output=assp_features
        # print('key_output',key_output.shape,assp_features.shape)
        key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
        for ii in range(prototypes.size(1)):
            prototype = prototypes[:, ii]
            if prototype is not None:
                #######key
                # key_output=getattr(self, f'a_{ii//(prototypes.size(1)//6)}') * key_output + getattr(self, f'b_{ii//((prototypes.size(1)//6))}')
                # query_output = prototype.view(prototype.size(0), -1, prototype.size(1))  # Reshape to [10, 1, 128]
                ########query
                # query_output = getattr(self, f'query_{ii // (prototypes.size(1) // 6)}') (prototype)
                query_output = getattr(self, f'a_{ii // (prototypes.size(1) // 6)}') * prototype + \
                               getattr(self,f'b_{ii // ((prototypes.size(1) // 6))}')
                query_output = query_output.view(query_output.size(0), -1, query_output.size(1))  # Reshape to [10, 1, 128]
                # prototypesOut.append(query_output.squeeze(1))
                # similarity = torch.abs(
                #     torch.bmm(query_output, key_output) / 128)  # Perform batch matrix multiplication#([10, 1, 1024])
                # Normalize along the channel dimension
                # query_output = F.normalize(query_output, p=2,dim=2)
                # key_output = F.normalize(key_output, p=2, dim=2)
                # Perform batch matrix multiplication#([10, 1, 1024])

                similarity = self.normP(torch.matmul(query_output, key_output))
                # similarity = torch.matmul(query_output, key_output)
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                             assp_features.size(3))  # ([10, 1, 16, 16])
                similarityList.append(similarity)
                query_outputList.append(query_output)

        query_outputcat = torch.cat(query_outputList, dim=1).unsqueeze(-1).unsqueeze(-1)
        # print('query_outputcat',query_outputcat.shape,prototypes.shape,GlobalProto_transOut.shape)
        similarityCat = torch.cat(similarityList, dim=1)#([10, 30, 32, 32])
        # print('similarityCat',similarityCat.shape)
        # print('similarityCat',similarityCat.shape)
        similarityWeiht = F.softmax(similarityCat, dim=1)#torch.Size([10, 30, 32, 32])
        # print('similarityCat',similarityCat.shape)

        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        assp_weighted = assp_features * similarityWeihtMax.unsqueeze(1)
        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': xup}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,
                'Weight': [similarityWeiht, similarityWeihtMax]}
class DeepLabV3PlusSimGlobalLinearKL(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimGlobalLinearKL, self).__init__()
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
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode == 'nopre':
            self.resnet = models.resnet50(weights=None)
        self.num_classes = num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), )

        self.classifier = nn.Conv2d(featDim, num_classes, 1)

        self.n_cluster = n_cluster
        self.prototypeN = 1
        # self.key = nn.Conv2d(featDim, featDim, kernel_size=3, padding=1)
        for i in range(num_classes):
            # setattr(self, f'query_{i}', nn.Conv2d(featDim * self.prototypeN, 128, kernel_size=1))
            setattr(self, f'a_{i}', nn.Parameter(torch.ones(1), requires_grad=True))
            setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1), requires_grad=True))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
        self.normP=nn.LayerNorm(128)
    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None,getPFlag=False):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)  # [10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        if getPFlag:
            return assp_features
        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        prototypes = []
        if ProtoInput == None:
            with torch.no_grad():
                pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
                #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
                mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
            # 计算每个类别的原型
                for i in range(self.num_classes):
                    class_mask = (mask == i).float()
                    if maskParam is not None:
                        class_mask = class_mask * maskParam  # Apply maskParam
                    if class_mask.sum() > 0:  # 确保类别在批次中存在
                        # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                        prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                                    class_mask.sum(dim=[2, 3]) + 1e-5)  # ([10, 256])
                        prototypes.append(prototype.unsqueeze(1))
                    else:
                        # for pp in range(self.n_cluster):
                        prototypes.append(zero_prototype.unsqueeze(1))
                prototypesC = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze( -1)  # prototypes torch.Size([10, 6, 128, 1, 1])
                prototypes = prototypesC
        elif ProtoInput != None:
            prototypes = ProtoInput
        query_outputList = []
        # prototypesOut = []
        similarityList = []
        # key_output = self.key(assp_features)
        key_output=assp_features
        # print('key_output',key_output.shape,assp_features.shape)
        # key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
        # jj=0
        # for c in range(self.num_classes):
        #     if ProtoInput == None:
        #         prototype = prototypes[:, jj]
        #         jj = jj + 1
        #         if prototype is not None:
        #             query_output = getattr(self, f'a_{c}') * prototype + getattr(self, f'b_{c}')
        #             deep_features_normalized = F.normalize(key_output, p=2, dim=1)  # [B, 128, H, W]
        #             proto_features_normalized = F.normalize(query_output, p=2, dim=1)  # [B, 128, 1, 1]
        #             query_output = query_output.view(query_output.size(0), -1,
        #                                              query_output.size(1))  # Reshape to [10, 1, 128]
        #             similarity = (deep_features_normalized * proto_features_normalized)  # [B, H, W]
        #             similarity = similarity.sum(dim=1).unsqueeze(1)  # [B, H, W]
        #             similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
        #                                          assp_features.size(3))  # ([10, 1, 16, 16])
        #             similarityList.append(similarity)
        #             query_output = query_output.view(query_output.size(0), 1, 128
        #                                              )  # Reshape to [10, 1, 128]
        #             # print('query_outputC', query_output.shape)
        #             query_outputList.append(query_output)
        #     else:
        #         for ii in range(self.n_cluster):
        #             prototype = prototypes[:, jj]
        #             jj = jj + 1
        #             if prototype is not None:
        #                 query_output = getattr(self, f'a_{c}') * prototype + getattr(self, f'b_{c}')
        #                 deep_features_normalized = F.normalize(key_output, p=2, dim=1)  # [B, 128, H, W]
        #                 proto_features_normalized = F.normalize(query_output, p=2, dim=1)  # [B, 128, 1, 1]
        #                 query_output = query_output.view(query_output.size(0), -1,
        #                                                  query_output.size(1))  # Reshape to [10, 1, 128]
        #                 similarity = (deep_features_normalized * proto_features_normalized)  # [B, H, W]
        #                 similarity = similarity.sum(dim=1).unsqueeze(1)  # [B, H, W]
        #                 similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
        #                                              assp_features.size(3))  # ([10, 1, 16, 16])
        #                 similarityList.append(similarity)
        #                 query_output = query_output.view(query_output.size(0), 1,
        #                                                  128)  # Reshape to [10, 1, 128]
        #                 query_outputList.append(query_output)
        # print('jj',jj)
        for ii in range(prototypes.size(1)):
            prototype = prototypes[:, ii]
            if prototype is not None:
                #######key
                # key_output=getattr(self, f'a_{ii//(prototypes.size(1)//6)}') * key_output + getattr(self, f'b_{ii//((prototypes.size(1)//6))}')
                # query_output = prototype.view(prototype.size(0), -1, prototype.size(1))  # Reshape to [10, 1, 128]
                ########query
                # query_output = getattr(self, f'query_{ii // (prototypes.size(1) // 6)}') (prototype)
                # query_output = getattr(self, f'a_{ii // (prototypes.size(1) // 6)}') * prototype + \
                #                getattr(self,f'b_{ii // ((prototypes.size(1) // 6))}')
                # query_output = query_output
                # proto_features = query_output.unsqueeze(-1).unsqueeze(-1)  # [B, 128, 1, 1]

                # print('query_output',query_output.shape)
                # 对特征进行softmax转换为概率分布
                # proto_features = F.softmax(query_output, dim=1)  # [B, 128, 1, 1]
                # deep_features = F.softmax(key_output, dim=1)  # [B, 128, H, W]
                # print('similarity', deep_features.shape,proto_features.shape)
                # similarity = F.kl_div(deep_features.log(), proto_features, reduction='none') # [B, 128, H, W]
                deep_features_normalized = F.normalize(key_output, p=2, dim=1)  # [B, 128, H, W]
                proto_features_normalized = F.normalize(prototype, p=2, dim=1)  # [B, 128, 1, 1]
                similarity = (deep_features_normalized * proto_features_normalized) # [B, H, W]

                similarity = similarity.sum(dim=1).unsqueeze(1) # [B, H, W]

                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                             assp_features.size(3))  # ([10, 1, 16, 16])
                similarityList.append(similarity)
                query_output = prototype.view(prototype.size(0), -1,
                                                 prototype.size(1))  # Reshape to [10, 1, 128]
                query_outputList.append(query_output)

        query_outputcat = torch.cat(query_outputList, dim=1).unsqueeze(-1).unsqueeze(-1)
        # print('query_outputcat',query_outputcat.shape,prototypes.shape,GlobalProto_transOut.shape)
        similarityCat = torch.cat(similarityList, dim=1)#([10, 30, 32, 32])
        similarityWeiht = F.softmax(similarityCat, dim=1)#torch.Size([10, 30, 32, 32])
        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        assp_weighted = assp_features * similarityWeihtMax.unsqueeze(1)
        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': xup}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,
                'Weight': [similarityWeiht, similarityWeihtMax]}

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []
        b.append(self.resnet.conv1)
        b.append(self.resnet.bn1)
        b.append(self.resnet.layer1)
        b.append(self.resnet.layer2)
        b.append(self.resnet.layer3)
        b.append(self.resnet.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k
    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.aspp.parameters())
        b.append(self.classifier.parameters())
        # b.append(self.layer7.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self):
        learning_rate=2.5e-4
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * learning_rate}]
# from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
class DeepLabV3PlusSimGlobalLinearKL2(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimGlobalLinearKL2, self).__init__()
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
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode == 'nopre':
            self.resnet = models.resnet50(weights=None)
        self.num_classes = num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(featDim),
            nn.ReLU() )

        self.classifier = nn.Conv2d(featDim, num_classes, 1)

        self.n_cluster = n_cluster
        self.prototypeN = 1
        # self.key = nn.Conv2d(featDim, featDim, kernel_size=3, padding=1)
        # for i in range(num_classes):
        #     setattr(self, f'query_{i}', nn.Conv2d(featDim * self.prototypeN, 128, kernel_size=1))
        #     setattr(self, f'a_{i}', nn.Parameter(torch.ones(1), requires_grad=True))
        #     setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1), requires_grad=True))
        #     # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
        # self.normP=nn.LayerNorm(128)
    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None,getPFlag=False):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)  # [10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        if getPFlag:
            return assp_features
        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        prototypes = []
        if ProtoInput == None:
            with torch.no_grad():
                pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
                mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
            # 计算每个类别的原型
                for i in range(self.num_classes):
                    class_mask = (mask == i).float()
                    if maskParam is not None:
                        class_mask = class_mask * maskParam  # Apply maskParam
                    if class_mask.sum() > 0:  # 确保类别在批次中存在
                        prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                                    class_mask.sum(dim=[2, 3]) + 1e-5)  # ([10, 256])
                        prototypes.append(prototype.unsqueeze(1))
                    else:
                        prototypes.append(zero_prototype.unsqueeze(1))
                prototypesC = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze( -1)  # prototypes torch.Size([10, 6, 128, 1, 1])
                prototypes = prototypesC
        elif ProtoInput != None:
            prototypes = ProtoInput
        query_outputList = []
        similarityList = []
        # key_output = self.key(assp_features)
        key_output=assp_features
        for ii in range(prototypes.size(1)):
            prototype = prototypes[:, ii]
            if prototype is not None:

                deep_features_normalized = F.normalize(key_output, p=2, dim=1)  # [B, 128, H, W]
                proto_features_normalized = F.normalize(prototype, p=2, dim=1)  # [B, 128, 1, 1]
                similarity = (deep_features_normalized * proto_features_normalized) # [B, H, W]

                similarity = similarity.sum(dim=1).unsqueeze(1) # [B, H, W]

                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                             assp_features.size(3))  # ([10, 1, 16, 16])


                similarityList.append(similarity)
                query_output = prototype.view(prototype.size(0), -1,
                                                 prototype.size(1))  # Reshape to [10, 1, 128]
                query_outputList.append(query_output)

        query_outputcat = torch.cat(query_outputList, dim=1).unsqueeze(-1).unsqueeze(-1)
        similarityCat = torch.cat(similarityList, dim=1)#([10, 30, 32, 32])
        similarityWeiht = F.softmax(similarityCat, dim=1)#torch.Size([10, 30, 32, 32])
        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        assp_weighted = assp_features * similarityWeihtMax.unsqueeze(1)
        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': xup}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,
                'Weight': [similarityWeiht, similarityWeihtMax]}

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []
        b.append(self.resnet.conv1)
        b.append(self.resnet.bn1)
        b.append(self.resnet.layer1)
        b.append(self.resnet.layer2)
        b.append(self.resnet.layer3)
        b.append(self.resnet.layer4)
        # for i in range(len(b)):
        #     for j in b[i].modules():
        #         jj = 0
        #         for k in j.parameters():
        #             jj += 1
        #             if k.requires_grad:
        #                 yield k
        for module in b:
            for param in module.parameters():
                if param.requires_grad:
                    yield param

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.aspp)
        b.append(self.classifier)
        # b.append(self.layer7.parameters())
        for module in b:
            for param in module.parameters():
                if param.requires_grad:
                    yield param

        # for j in range(len(b)):
        #     for i in b[j]:
        #         yield i

    def optim_parameters(self,learning_rate):
        # learning_rate=0.0001
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': learning_rate/10},
                {'params': self.get_10x_lr_params(), 'lr': learning_rate}]

class DeepLabV3PlusSimGlobalLinearKL3(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimGlobalLinearKL3, self).__init__()
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
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode == 'nopre':
            self.resnet = models.resnet50(weights=None)
        self.num_classes = num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(featDim),
            nn.ReLU(), )

        self.classifier = nn.Conv2d(featDim, num_classes, 1)
        self.n_cluster = n_cluster
        self.prototypeN = 1
        # self.key = nn.Conv2d(featDim, featDim, kernel_size=3, padding=1)
        # self.sc=nn.Conv2d(featDim, featDim, kernel_size=3,  padding=1)
        Sim=nn.Sequential(
            nn.Conv2d(featDim, featDim, kernel_size=3,  padding=1),
            # nn.ConvTranspose2d(featDim, featDim, kernel_size=3, stride=4, padding=1, output_padding=1),

            # nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(featDim),
            # nn.ReLU(),
            # nn.Conv2d(featDim, 1, kernel_size=1)
            )
        for i in range(num_classes):
            # setattr(self, f'sim_{i}', nn.Conv2d(featDim, featDim, kernel_size=3,  padding=1))
            setattr(self, f'sim_{i}', Sim)

            # setattr(self, f'a_{i}', nn.Parameter(torch.ones(1), requires_grad=True))
            # setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1), requires_grad=True))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
        # self.normP=nn.LayerNorm(128)
    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None,getPFlag=False):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        assp_features = self.aspp(x4)  # [10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        if getPFlag:
            return assp_features
        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        prototypes = []
        if ProtoInput == None:
            with torch.no_grad():
                pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
                #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
                mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
            # 计算每个类别的原型
                for i in range(self.num_classes):
                    class_mask = (mask == i).float()
                    if maskParam is not None:
                        class_mask = class_mask * maskParam  # Apply maskParam
                    if class_mask.sum() > 0:  # 确保类别在批次中存在
                        prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                                    class_mask.sum(dim=[2, 3]) + 1e-5)  # ([10, 256])
                        prototypes.append(prototype.unsqueeze(1))
                    else:
                        # for pp in range(self.n_cluster):
                        prototypes.append(zero_prototype.unsqueeze(1))
                prototypesC = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze( -1)  # prototypes torch.Size([10, 6, 128, 1, 1])
                prototypes = prototypesC
        elif ProtoInput != None:
            prototypes = ProtoInput
        query_outputList = []
        # prototypesOut = []
        similarityList = []
        # key_output = self.key(assp_features)
        key_output=assp_features
        for ii in range(prototypes.size(1)):
            prototype = prototypes[:, ii]
            if prototype is not None:
                ######key
                # key_output=getattr(self, f'a_{ii//(prototypes.size(1)//6)}') * key_output + getattr(self, f'b_{ii//((prototypes.size(1)//6))}')
                # query_output = prototype.view(prototype.size(0), -1, prototype.size(1))  # Reshape to [10, 1, 128]
                #######query

                deep_features_normalized = F.normalize(key_output, p=2, dim=1)  # [B, 128, H, W]
                proto_features_normalized = F.normalize(prototype, p=2, dim=1)  # [B, 128, 1, 1]
                similarity = (deep_features_normalized * proto_features_normalized) # [B, H, W]
                similarity = getattr(self, f'sim_{ii // (prototypes.size(1) // 6)}') (similarity)
                # similarity=self.sc(similarity)
                similarity = similarity.sum(dim=1).unsqueeze(1) # [B, H, W]
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                             assp_features.size(3))  # ([10, 1, 16, 16])
                similarityList.append(similarity)
                query_output = prototype.view(prototype.size(0), -1,
                                                 prototype.size(1))  # Reshape to [10, 1, 128]
                query_outputList.append(query_output)

        query_outputcat = torch.cat(query_outputList, dim=1).unsqueeze(-1).unsqueeze(-1)
        # print('query_outputcat',query_outputcat.shape,prototypes.shape,GlobalProto_transOut.shape)
        similarityCat = torch.cat(similarityList, dim=1)#([10, 30, 32, 32])
        similarityWeiht = F.softmax(similarityCat, dim=1)#torch.Size([10, 30, 32, 32])
        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        assp_weighted = assp_features * similarityWeihtMax.unsqueeze(1)
        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': xup}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,
                'Weight': [similarityWeiht, similarityWeihtMax]}

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []
        b.append(self.resnet.conv1)
        b.append(self.resnet.bn1)
        b.append(self.resnet.layer1)
        b.append(self.resnet.layer2)
        b.append(self.resnet.layer3)
        b.append(self.resnet.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k
    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.aspp.parameters())
        b.append(self.classifier.parameters())
        # b.append(self.layer7.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self):
        learning_rate=2.5e-4
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * learning_rate}]

def compute_gradient_map_weight(similarity_map,features,neiborWindow):
    B, C, H, W = similarity_map.shape
    G_x = torch.abs(similarity_map[:, :, :, 1:] - similarity_map[:, :, :, :-1])  # [B, 6, H, W-2]
    G_y = torch.abs(similarity_map[:, :, 1:, :] - similarity_map[:, :, :-1, :])  # [B, 6, H-2, W]
    print(G_x.shape,G_y.shape)
    G_x = F.pad(G_x, (1, 1, 0, 0), mode='constant', value=0)  # [B, 6, H, W]
    G_y = F.pad(G_y, (0, 0, 1, 1), mode='constant', value=0)  # [B, 6, H, W]
    G = torch.sqrt(G_x ** 2 + G_y ** 2)

    w_up = 1 / (1 + F.pad(G_y[:, :, :-(neiborWindow // 2), :], (0, 0, neiborWindow // 2, 0), mode='constant',
                          value=0))
    w_down = 1 / (
                1 + F.pad(G_y[:, :, neiborWindow // 2:, :], (0, 0, 0, neiborWindow // 2), mode='constant', value=0))
    w_left = 1 / (1 + F.pad(G_x[:, :, :, :-(neiborWindow // 2)], (neiborWindow // 2, 0, 0, 0), mode='constant',
                            value=0))
    w_right = 1 / (
                1 + F.pad(G_x[:, :, :, neiborWindow // 2:], (0, neiborWindow // 2, 0, 0), mode='constant', value=0))
    w = torch.cat([w_up.unsqueeze(1), w_down.unsqueeze(1), w_left.unsqueeze(1), w_right.unsqueeze(1)],
                  dim=1)  # [B, 4, 6, H, W]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    padded_features = F.pad(features, (1, 1, 1, 1), mode='constant', value=0)
    shifted_features = []
    for di, dj in directions:
        shifted = padded_features[:, :, 1 + di:H + 1 + di, 1 + dj:W + 1 + dj]
        shifted_features.append(shifted)
    shifted_features = torch.stack(shifted_features, dim=1)  # [B, 4, C, H, W]

    central_features = features.unsqueeze(1).expand(-1, 4, -1, -1, -1)  # [B, 4, C, H, W]
    local_similarities = F.cosine_similarity(central_features, shifted_features, dim=2)  # [B, 4, H, W]
    weighted_similarities = (1 - local_similarities.unsqueeze(2)) * w  # [B, 6, 4, H, W]
    weighted_loss = weighted_similarities.sum(dim=2).mean()


    return weighted_loss



def compute_gradient_map_weight2(similarity_map,features,N_win_list,DEVICE,mask=None):
    B, C, H, W = features.shape

    # weighted_losses = []
    weighted_losses = torch.tensor([0.0]).to(DEVICE)

    N=0
    for idx, N_win in enumerate(N_win_list):
        if N_win == 0:
            # weighted_losses=weighted_losses+
            # weighted_losses.append(torch.tensor(0.0, requires_grad=True))
            continue
        N=N+1
        grad_left, grad_right, grad_up, grad_down=compute_gradients(input_tensor=similarity_map[:, idx:idx + 1, :, :],N=N_win)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        padded_features = F.pad(features, (1, 1, 1, 1), mode='constant', value=0)
        shifted_features = []
        for di, dj in directions:
            shifted = padded_features[:, :, 1 + di:H + 1 + di, 1 + dj:W + 1 + dj]
            shifted_features.append(shifted)
        shifted_features = torch.stack(shifted_features, dim=1)  # [B, 4, C, H, W]
        central_features = features.unsqueeze(1).expand(-1, 4, -1, -1, -1).detach()  # [B, 4, C, H, W]
        S_local = F.cosine_similarity(central_features, shifted_features, dim=2)  # [B, 4, H, W]
        w_up = 1 / (1 + abs(grad_up))
        w_down = 1 / (1 + abs(grad_down))
        w_left = 1 / (1 + abs(grad_left))
        w_right = 1 / (1 + abs(grad_right))
        w_up[:, :, :N_win, :] = 0
        w_down[:, :, -N_win:, :] = 0
        w_left[:, :, :, :N_win] = 0
        w_right[:, :, :, -N_win:] = 0
        w = torch.cat([w_up.unsqueeze(1), w_down.unsqueeze(1), w_left.unsqueeze(1), w_right.unsqueeze(1)],
                      dim=1)  # [B, 4, 1, H, W]
        if mask ==None:
            weighted_loss=(1 - S_local.unsqueeze(2)) * w
            weighted_loss = weighted_loss.sum(dim=2).mean()

        else:
            # mask=mask.unsqueeze(1).unsqueeze(2)
            # print('mask',mask.shape,S_local.shape,w.shape)
            weighted_loss = (1 - S_local.unsqueeze(2)) * w  # [10, 4, 1, 32, 32]
            # print('weighted_loss',weighted_loss.shape)
            # print(weighted_loss.sum(dim=2).shape,mask.shape,mask.sum())
            weighted_loss = weighted_loss.sum(dim=2) * (mask)  # mask [10,32,32]
            # print('weighted_loss1', weighted_loss.shape)
            weighted_loss = weighted_loss.sum() / (mask.sum() + 1)
        # weighted_loss=(1 - S_local.unsqueeze(2)) * w
        # weighted_loss = weighted_loss.sum(dim=2).mean()

        weighted_losses=weighted_losses+weighted_loss
    weighted_losses=weighted_losses/N
    return weighted_losses
def compute_gradients(input_tensor, N=1):
    B, C, H, W = input_tensor.size()

    # Define padding for each direction
    pad_left = F.pad(input_tensor, (N, 0, 0, 0))[:, :, :, :-N]
    pad_right = F.pad(input_tensor, (0, N, 0, 0))[:, :, :, N:]
    pad_up = F.pad(input_tensor, (0, 0, N, 0))[:, :, :-N, :]
    pad_down = F.pad(input_tensor, (0, 0, 0, N))[:, :, N:, :]

    # Compute gradients
    grad_left = input_tensor - pad_left
    grad_right = pad_right - input_tensor
    grad_up = input_tensor - pad_up
    grad_down = pad_down - input_tensor
    return grad_left, grad_right, grad_up, grad_down
def compute_gradient_map_weight3(similarity_map,features,N_win_list,mask=None,lp=None,DEVICE=None):
    # print('lp',lp.shape)
    B, _, H, W = features.shape
    # print(C)
    mask2 = torch.zeros(B, 4, H, W).to(DEVICE)
    # weighted_losses = []
    weighted_losses = torch.tensor([0.0]).to(DEVICE)
    # weighted_losses = torch.tensor([0.0])
    N=0
    grad_left_list=[]
    grad_right_list=[]
    grad_up_list=[]
    grad_down_list=[]
    w_list=[]
    S_local_list=[]
    # grad_left
    for idx, N_win in enumerate(N_win_list):
        if N_win == 0:
            # weighted_losses=weighted_losses+
            # weighted_losses.append(torch.tensor(0.0, requires_grad=True))
            continue
        N=N+1
        grad_left, grad_right, grad_up, grad_down=compute_gradients(input_tensor=similarity_map[:, idx:idx + 1, :, :],N=N_win)
        grad_left_list.append(grad_left)
        grad_right_list.append(grad_right)
        grad_up_list.append(grad_up)
        grad_down_list.append(grad_down)
        # print(idx,'grad_left',grad_left.shape)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        padded_features = F.pad(features, (1, 1, 1, 1), mode='constant', value=0)
        shifted_features = []
        for di, dj in directions:
            shifted = padded_features[:, :, 1 + di:H + 1 + di, 1 + dj:W + 1 + dj]
            shifted_features.append(shifted)
        shifted_features = torch.stack(shifted_features, dim=1)  # [B, 4, C, H, W]
        central_features = features.unsqueeze(1).expand(-1, 4, -1, -1, -1)  # [B, 4, C, H, W]
        S_local = F.cosine_similarity(central_features, shifted_features, dim=2)  # [B, 4, H, W]
        S_local_list.append(S_local.unsqueeze(2))
        # print('S_local',S_local.shape)#[2, 4, 64, 64]
        w_up = 1 / (1 + abs(grad_up))
        w_down = 1 / (1 + abs(grad_down))
        w_left = 1 / (1 + abs(grad_left))
        w_right = 1 / (1 + abs(grad_right))
        w_up[:, :, :N_win, :] = 0
        w_down[:, :, -N_win:, :] = 0
        w_left[:, :, :, :N_win] = 0
        w_right[:, :, :, -N_win:] = 0
        w = torch.cat([w_up.unsqueeze(1), w_down.unsqueeze(1), w_left.unsqueeze(1), w_right.unsqueeze(1)],
                      dim=1)  # [B, 4, 1, H, W]
        w_list.append(w)

    wCat = torch.cat(w_list, dim=2)  # [B, 4, N, H, W]
    S_local_cat = torch.cat(S_local_list, dim=2)  # [B, 4, N, H, W]
    # print('S_local_cat', S_local_cat[0, 2, :, 0, 0], S_local_cat[0, 0, 2, 0, 0])
    grad_left = torch.cat(grad_left_list, dim=1)  # [B, N, H, W]
    grad_right = torch.cat(grad_right_list, dim=1)  # [B, N, H, W]
    grad_up = torch.cat(grad_up_list, dim=1)  # [B, N, H, W]
    grad_down = torch.cat(grad_down_list, dim=1)  # [B, N, H, W]

    # 生成 mask2
    lp = lp.to(torch.int64)  # 确保 lp 是 int64 类型
    # lp_expanded = lp.expand(-1, 1, -1, -1)  # 扩展 lp 以便于张量操作
    lp_expanded = lp
    grad_all_directions = torch.stack([grad_left, grad_right, grad_up, grad_down], dim=1)  # [B, 4, N, H, W]
    grad_cList = []
    S_local_cList = []
    wCat_List = []
    for direction in range(4):
        grad_current_direction = grad_all_directions[:, direction, :, :, :]  # [B, 6, H, W]
        S_local_ = S_local_cat[:, direction, :, :, :]  # [B, 6, H, W]Similarity
        w_ = wCat[:, direction, :, :, :]  # [B, 6, H, W]

        grad_c = grad_current_direction.gather(1, lp_expanded)  # [B, 1, H, W]
        S_local_c = S_local_.gather(1, lp_expanded)  # [B, 1, H, W]
        w_c = w_.gather(1, lp_expanded)  # [B, 1, H, W]

        S_local_cList.append(S_local_c)
        # print('S_local_c',S_local_c.shape)
        is_min = (grad_current_direction >= grad_c).all(dim=1, keepdim=True)  # [B, 1, H, W]

        mask2[:, direction, :, :] = is_min.squeeze(1).float()
        grad_cList.append(grad_c)
        wCat_List.append(w_c)

        # grad_sum = grad_current_direction.sum(dim=1, keepdim=True) - grad_c  # [B, 1, H, W]
        # print('grad_sum',grad_sum.shape)
        # mask2[:, direction, :, :] = (grad_c < grad_sum).squeeze(1).float()

    # grad_cCat = torch.cat(grad_cList, dim=1)
    S_local_cCat = torch.cat(S_local_cList, dim=1)
    wCat_cCat = torch.cat(wCat_List, dim=1)

    if mask is None:
        weighted_loss = (1 - S_local_cCat) * wCat_cCat * mask2
        weighted_loss = weighted_loss
        weighted_loss = weighted_loss.sum() / (mask2.sum() + 1)
    else:
        weighted_loss = (1 - S_local_cCat) * wCat_cCat
        maskAll = mask * mask2
        weighted_loss = weighted_loss * maskAll
        weighted_loss = weighted_loss.sum() / (maskAll.sum() + 1)

    # for i in range(N):
    #     if mask == None:
    #         weighted_loss = (1 - S_local_cat[:,:,i].unsqueeze(2)) * wCat[:,:,i].unsqueeze(2)*mask2[:,:].unsqueeze(2)
    #         # print(wCat[:,:,i].unsqueeze(2).shape,mask2[:,:,i].shape,weighted_loss.shape)
    #         weighted_loss = weighted_loss.sum(dim=2)
    #         weighted_loss = weighted_loss.sum() / (mask2.sum() + 1)
    #     else:
    #         weighted_loss = (1 - S_local_cat[:,:,i].unsqueeze(2)) * wCat[:,:,i].unsqueeze(2)# [10, 4, 1, 32, 32]
    #         weighted_loss = weighted_loss.sum(dim=2)
    #         maskAll=mask*mask2[:,:]
    #
    #         weighted_loss = weighted_loss * (maskAll)  # mask [10,32,32]
    #         weighted_loss = weighted_loss.sum() / (maskAll.sum() + 1)
    #     weighted_losses = weighted_losses + weighted_loss
    #
    # weighted_losses=weighted_losses/N
    return weighted_loss
class GradientMapWeight(nn.Module):
    def __init__(self, n_clusters, device):
        super(GradientMapWeight, self).__init__()
        self.n_clusters = n_clusters
        self.device = device

    # def compute_gradients(self, input_tensor, N):
    #     # Placeholder function: replace with actual gradient computation logic
    #     grad_left = torch.randn_like(input_tensor)
    #     grad_right = torch.randn_like(input_tensor)
    #     grad_up = torch.randn_like(input_tensor)
    #     grad_down = torch.randn_like(input_tensor)
    #     return grad_left, grad_right, grad_up, grad_down

    def compute_gradients(self,input_tensor, N=1):
        B, C, H, W = input_tensor.size()

        # Define padding for each direction
        pad_left = F.pad(input_tensor, (N, 0, 0, 0))[:, :, :, :-N]
        pad_right = F.pad(input_tensor, (0, N, 0, 0))[:, :, :, N:]
        pad_up = F.pad(input_tensor, (0, 0, N, 0))[:, :, :-N, :]
        pad_down = F.pad(input_tensor, (0, 0, 0, N))[:, :, N:, :]

        # Compute gradients
        grad_left = input_tensor - pad_left
        grad_right = pad_right - input_tensor
        grad_up = input_tensor - pad_up
        grad_down = pad_down - input_tensor
        return grad_left, grad_right, grad_up, grad_down
    def forward(self, similarity_map, features, N_win_list, mask=None, lp=None, Proto=None, name=None):
        n_clusterN = self.n_clusters
        B, _, H, W = features.shape

        # Initialize mask2 and weighted_losses
        mask2 = torch.zeros(B, 4, H, W, requires_grad=False).to(self.device)
        weighted_losses = torch.tensor([0.0]).to(self.device)
        N = 0

        # Lists to store gradients and weights
        grad_left_list = []
        grad_right_list = []
        grad_up_list = []
        grad_down_list = []
        w_list = []
        S_local_list = []

        # Iterate over N_win_list
        for idx, N_win in enumerate(N_win_list):
            if N_win == 0:
                continue
            N += 1
            # Compute gradients
            grad_left, grad_right, grad_up, grad_down = self.compute_gradients(
                input_tensor=similarity_map[:, idx:idx + 1, :, :], N=N_win)
            grad_left_list.append(grad_left)
            grad_right_list.append(grad_right)
            grad_up_list.append(grad_up)
            grad_down_list.append(grad_down)

            # Define directions for padding and shifting features
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            padded_features = F.pad(features, (1, 1, 1, 1), mode='constant', value=0)
            shifted_features = []
            for di, dj in directions:
                shifted = padded_features[:, :, 1 + di:H + 1 + di, 1 + dj:W + 1 + dj]
                shifted_features.append(shifted)
            shifted_features = torch.stack(shifted_features, dim=1)  # [B, 4, C, H, W]

            # Normalize features
            deep_features_normalized = F.normalize(features, p=2, dim=1)  # [B, 128, H, W]
            similaritylist = []
            for jj in range(n_clusterN):
                proto_features_normalized = F.normalize(Proto[:, idx * n_clusterN + jj], p=2, dim=1)  # [B, 128, 1, 1]
                similarity = (deep_features_normalized * proto_features_normalized).mean(dim=1).unsqueeze(1)  # [B, H, W]
                similaritylist.append(similarity)
            similaritycat = torch.cat(similaritylist, dim=1)
            similaritymax, similarityindex = similaritycat.max(dim=1)

            Proto_flat = Proto[:, idx * n_clusterN:idx * n_clusterN + n_clusterN, :, 0, 0]
            similarityindex_flat = similarityindex.view(B, -1)  # [B, 32*32]
            Pc_selected_flat = Proto_flat.gather(1, similarityindex_flat.unsqueeze(2).expand(-1, -1, 128))  # [B, 32*32, 128]
            Pc_selected = Pc_selected_flat.view(B, H, W, 128).permute(0, 3, 1, 2)  # [B, 128, 32, 32]
            Pc_selected_expanded = Pc_selected.unsqueeze(1).expand(-1, 4, -1, -1, -1)  # [B, 4, 128, 32, 32]
            # print('S_local6', idx, N_win,similaritycat.shape, similarityindex_flat.min(), similarityindex_flat.max(),similaritymax.min(),similaritymax.max())

            S_local = F.cosine_similarity(Pc_selected_expanded, shifted_features, dim=2)  # [B, 4, 32, 32]
            S_local_list.append(S_local.unsqueeze(2))

            # Compute weights for gradients
            w_up = 1 / (1 + abs(grad_up))
            w_down = 1 / (1 + abs(grad_down))
            w_left = 1 / (1 + abs(grad_left))
            w_right = 1 / (1 + abs(grad_right))
            w_up[:, :, :N_win, :] = 0
            w_down[:, :, -N_win:, :] = 0
            w_left[:, :, :, :N_win] = 0
            w_right[:, :, :, -N_win:] = 0
            w = torch.cat([w_up.unsqueeze(1), w_down.unsqueeze(1), w_left.unsqueeze(1), w_right.unsqueeze(1)], dim=1)  # [B, 4, 1, H, W]
            w_list.append(w)

        # Concatenate all the collected gradients and weights
        wCat = torch.cat(w_list, dim=2)  # [B, 4, N, H, W]
        # print('wCat',wCat.shape)
        S_local_cat = torch.cat(S_local_list, dim=2)  # [B, 4, N, H, W]
        grad_left = torch.cat(grad_left_list, dim=1)  # [B, N, H, W]
        grad_right = torch.cat(grad_right_list, dim=1)  # [B, N, H, W]
        grad_up = torch.cat(grad_up_list, dim=1)  # [B, N, H, W]
        grad_down = torch.cat(grad_down_list, dim=1)  # [B, N, H, W]

        # Ensure lp is int64 type
        lp = lp.to(torch.int64)
        lp_expanded = lp.expand(-1, 1, -1, -1)

        grad_all_directions = abs(torch.stack([grad_left, grad_right, grad_up, grad_down], dim=1))  # [B, 4, N, H, W]
        grad_cList = []
        S_local_cList = []
        wCat_List = []

        # Compute masks and gather gradients, similarity, and weights
        for direction in range(4):
            grad_current_direction = grad_all_directions[:, direction, :, :, :]  # [B, 6, H, W]
            S_local_ = S_local_cat[:, direction, :, :, :]  # [B, 6, H, W]
            w_ = wCat[:, direction, :, :, :]  # [B, 6, H, W]
            grad_c = grad_current_direction.gather(1, lp_expanded)  # [B, 1, H, W]
            S_local_c = S_local_.gather(1, lp_expanded)  # [B, 1, H, W]
            w_c = w_.gather(1, lp_expanded)  # [B, 1, H, W]
            S_local_cList.append(S_local_c)
            is_min = (grad_current_direction >= grad_c).all(dim=1, keepdim=True)  # [B, 1, H, W]
            mask2[:, direction, :, :] = is_min.squeeze(1).float().detach()
            grad_cList.append(grad_c)
            wCat_List.append(w_c)
        grad_cCat = torch.cat(grad_cList, dim=1)

        if name is not None:
            for ii in range(len(name)):
                path = name[ii].split('/')[-1]
                # print(path)
                nameList = ['area32_512_0_1024_512.png', 'area32_512_512_1024_1024.png',
                            'area32_512_1024_1024_1536.png', 'area32_512_1536_1024_2048.png',
                            'area32_512_2043_1024_2555.png', 'area33_0_0_512_512.png', 'area33_0_512_512_1024.png',
                            'area33_0_1024_512_1536.png',
                            'area33_0_1536_512_2048.png', 'area33_0_2043_512_2555.png']
                if path in nameList:
                    print('oook-', path)
                    np.save('./outimg/mask/%s_mask.npy' % (path.split('.')[0]), mask2.detach().cpu().numpy())
                    np.save('./outimg/mask/%s_grad_c.npy' % (path.split('.')[0]), grad_cCat.detach().cpu().numpy())
        S_local_cCat = torch.cat(S_local_cList, dim=1)
        wCat_cCat = torch.cat(wCat_List, dim=1)
        # print('S_local_cCat',S_local_cCat.shape)
        # Calculate weighted loss
        if mask is None:

            weighted_loss = (1 - S_local_cCat) * wCat_cCat * mask2
            weighted_loss = weighted_loss.sum() / (mask2.sum() + 1)
        else:
            weighted_loss = (1 - S_local_cCat) * wCat_cCat
            maskAll = mask * mask2
            weighted_loss = weighted_loss * maskAll
            weighted_loss = weighted_loss.sum() / (maskAll.sum() + 1)

        return weighted_loss
class GradientMapWeight2(nn.Module):
    def __init__(self, n_clusters, device):
        super(GradientMapWeight2, self).__init__()
        self.n_clusters = n_clusters
        self.device = device
    def compute_gradients(self,input_tensor, N=1):
        # B, C, H, W = input_tensor.size()

        # Define padding for each direction
        pad_left = F.pad(input_tensor, (N, 0, 0, 0))[:, :, :, :-N]
        pad_right = F.pad(input_tensor, (0, N, 0, 0))[:, :, :, N:]
        pad_up = F.pad(input_tensor, (0, 0, N, 0))[:, :, :-N, :]
        pad_down = F.pad(input_tensor, (0, 0, 0, N))[:, :, N:, :]

        # # Compute gradients
        grad_left = abs(input_tensor - pad_left)/N
        # print('grad_left',grad_left.shape)
        grad_right = abs(pad_right - input_tensor)/N
        grad_up = abs(input_tensor - pad_up)/N
        grad_down = abs(pad_down - input_tensor)/N

        return grad_left, grad_right, grad_up, grad_down
    def forward(self, similarityCat, features, N_win_list, mask=None, lp=None, Proto=None, name=None,data=None,SimP=None,maskp=None):
        # similarityCatMax = similarityCat.view(similarityCat.size(0), similarityCat.size(1) // (self.n_clusters),
        #                                       (self.n_clusters),
        #                                       similarityCat.size(2),
        #                                       similarityCat.size(3))
        # similarityCatMax = similarityCatMax.max(dim=2)[0]
        # print('similarityCat',similarityCat.shape,similarityCatMax.shape)
        n_clusterN = self.n_clusters
        B, _, H, W = features.shape
        # Initialize mask2 and weighted_losses
        # weighted_losses = torch.tensor([0.0]).to(self.device)
        N = 0
        # Lists to store gradients and weights
        grad_left_list = []
        grad_right_list = []
        grad_up_list = []
        grad_down_list = []
        w_list = []
        S_local_list = []

        # Iterate over N_win_list
        for idx, N_win in enumerate(N_win_list):
            if N_win == 0:
                continue
            N += 1
            # Compute gradients
            # Define directions for padding and shifting features
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            padded_features = F.pad(features, (N_win, N_win, N_win, N_win), mode='constant', value=0)
            shifted_features = []
            for di, dj in directions:
                shifted = padded_features[:, :, 1 + di:H + 1 + di, 1 + dj:W + 1 + dj]
                shifted_features.append(shifted)
            shifted_features = torch.stack(shifted_features, dim=1)  # [B, 4, C, H, W]
            # # Normalize features
            # deep_features_normalized = F.normalize(features, p=2, dim=1)  # [B, 128, H, W]
            # similaritylist = []
            # for jj in range(n_clusterN):
            #     proto_features_normalized = F.normalize(Proto[:, idx * n_clusterN + jj], p=2, dim=1)  # [B, 128, 1, 1]
            #     similarity = (deep_features_normalized * proto_features_normalized).sum(dim=1).unsqueeze(1)  # [B, H, W]
            #     # similarity=F.cosine_similarity(features,Proto[:, idx * n_clusterN + jj]).unsqueeze(1)
            #     similaritylist.append(similarity)
            # similaritycat = torch.cat(similaritylist, dim=1)
            # similaritymax, similarityindex = similaritycat.max(dim=1)
            # print('similaritymax',similaritymax.shape)
            similarityc=similarityCat[:, idx * n_clusterN:idx * n_clusterN + n_clusterN]
            similaritymax, similarityindex = similarityc.max(dim=1)
            # print('similaritymax',similaritymax.shape,similarityCatMax[:,idx].shape,(similaritymax == similarityCatMax[:,idx]).all())
            grad_left, grad_right, grad_up, grad_down = self.compute_gradients(input_tensor=similaritymax.unsqueeze(1), N=N_win)

            grad_left_list.append(grad_left)
            grad_right_list.append(grad_right)
            grad_up_list.append(grad_up)
            grad_down_list.append(grad_down)

            Proto_flat = Proto[:, idx * n_clusterN:idx * n_clusterN + n_clusterN, :, 0, 0]
            similarityindex_flat = similarityindex.view(B, -1)  # [B, 32*32]
            Pc_selected_flat = Proto_flat.gather(1, similarityindex_flat.unsqueeze(2).expand(-1, -1, 128))  # [B, 32*32, 128]
            Pc_selected = Pc_selected_flat.view(B, H, W, 128).permute(0, 3, 1, 2)  # [B, 128, 32, 32]
            Pc_selected_expanded = Pc_selected.unsqueeze(1).expand(-1, 4, -1, -1, -1)  # [B, 4, 128, 32, 32]
            # print('S_local6', idx, N_win,similaritycat.shape, similarityindex_flat.min(), similarityindex_flat.max(),similaritymax.min(),similaritymax.max())
            # Pc_selected_expanded_normalized = F.normalize(Pc_selected_expanded, p=2, dim=1)  # [B, 128, H, W]
            # shifted_features_normalized= F.normalize(Pc_selected_expanded, p=2, dim=1)  # [B, 128, H, W]
            # central_features = features.unsqueeze(1).expand(-1, 4, -1, -1, -1).detach()  # [B, 4, C, H, W]

            S_local = F.cosine_similarity(Pc_selected_expanded, shifted_features, dim=2)  # [B, 4, 32, 32]
            # S_local=(Pc_selected_expanded_normalized*shifted_features_normalized).sum(dim=2)
            # print('S_local1',S_local1.shape,S_local.shape,)
            # print((S_local1==S_local).all())
            S_local_list.append(S_local.unsqueeze(2))
            # Compute weights for gradients
            w_up = 1 / (1 + abs(grad_up))
            w_down = 1 / (1 + abs(grad_down))
            w_left = 1 / (1 + abs(grad_left))
            w_right = 1 / (1 + abs(grad_right))
            w_up[:, :, :N_win, :] = 0
            w_down[:, :, -N_win:, :] = 0
            w_left[:, :, :, :N_win] = 0
            w_right[:, :, :, -N_win:] = 0
            w = torch.cat([w_left.unsqueeze(1), w_right.unsqueeze(1), w_up.unsqueeze(1), w_down.unsqueeze(1)], dim=1)  # [B, 4, 1, H, W]
            w_list.append(w)

        # Concatenate all the collected gradients and weights
        wCat = torch.cat(w_list, dim=2)  # [B, 4, N, H, W]
        # print('wCat',wCat.shape)
        S_local_cat = torch.cat(S_local_list, dim=2)  # [B, 4, N, H, W]
        grad_left = torch.cat(grad_left_list, dim=1)  # [B, N, H, W]
        grad_right = torch.cat(grad_right_list, dim=1)  # [B, N, H, W]
        grad_up = torch.cat(grad_up_list, dim=1)  # [B, N, H, W]
        grad_down = torch.cat(grad_down_list, dim=1)  # [B, N, H, W]

        # Ensure lp is int64 type
        lp = lp.to(torch.int64)
        lp_expanded = lp.expand(-1, 1, -1, -1)#[1, 1, 32, 32])
        # print('lp_expanded',lp_expanded.shape)
        grad_all_directions = (torch.stack([grad_left, grad_right, grad_up, grad_down], dim=1))  # [B, 4, N, H, W]
        grad_cList = []
        S_local_cList = []
        wCat_List = []
        mask2 = torch.zeros(B, 4, H, W, requires_grad=False).to(self.device)
        minL=[]
        # Compute masks and gather gradients, similarity, and weights
        for direction in range(4):
            grad_current_direction = grad_all_directions[:, direction, :, :, :]  # [B, 6, H, W]
            # grad_current_directionMin=grad_current_direction-grad_current_direction.min(dim=1, keepdim=True)[0]
            # grad_current_directionMin=grad_current_directionMin/(grad_current_directionMin.max(dim=1, keepdim=True)[0]+1e-7)
            # print('grad_current_direction',grad_current_direction[0,:,10,10])
            grad_c = grad_current_direction.gather(1, lp_expanded)  # [B, 1, H, W]
            grad_cList.append(grad_c)
            min_vals, _ = grad_current_direction.min(dim=1, keepdim=True)  # [B, 1, H, W]
            minL.append(min_vals)
            # print( (grad_c <= min_vals).float().shape, (grad_c <= min_vals).float().sum())
            # print('min_vals',min_vals.shape,grad_c.shape)
            mask2[:, direction, :, :] = (grad_c ==min_vals).all(dim=1, keepdim=True).float().squeeze(1).detach()  # [B, 1, H, W]
            # is_min = (grad_current_direction <= grad_c).all(dim=1, keepdim=True)  # [B, 1, H, W]
            # mask2[:, direction, :, :] = is_min.squeeze(1).float().detach()
            w_ = wCat[:, direction, :, :, :]  # [B, 6, H, W]
            w_c = w_.gather(1, lp_expanded)  # [B, 1, H, W]
            wCat_List.append(w_c)

            # print('grad_current_direction',grad_current_direction.shape,grad_c.shape)#[10, 6, 32, 32] [10, 1, 32, 32]
            # is_min = (grad_current_direction >= grad_c).all(dim=1, keepdim=True)  # [B, 1, H, W]
            # mask2[:, direction, :, :] = is_min.squeeze(1).float().detach()

            S_local_ = S_local_cat[:, direction, :, :, :]  # [B, 6, H, W]
            S_local_c = S_local_.gather(1, lp_expanded)  # [B, 1, H, W]
            S_local_cList.append(S_local_c)
        minLcat=torch.cat(minL,dim=1)
        grad_cCat = torch.cat(grad_cList, dim=1)
        mask2[:, :, 0, :] = 0
        mask2[:, :, -1, :] = 0
        mask2[:, :, :, 0] = 0
        mask2[:, :, :, -1] = 0
        minLcat[:, :, 0, :] = 0
        minLcat[:, :, -1, :] = 0
        minLcat[:, :, :, 0] = 0
        minLcat[:, :, :, -1] = 0
        grad_cCat[:, :, 0, :] = 0
        grad_cCat[:, :, -1, :] = 0
        grad_cCat[:, :, :, 0] = 0
        grad_cCat[:, :, :, -1] = 0
        wCat_cCat = torch.cat(wCat_List, dim=1)

        if name is not None:
            ProtoG=SimP[0]
            FeatTTG=SimP[1]
            prototype_features_transposed = ProtoG['query'].squeeze(-1).squeeze(-1)  # [B, 128, 6]
            deep_features_reshaped = FeatTTG['asspF'].reshape(FeatTTG['asspF'].size(0), FeatTTG['asspF'].size(1),
                                                              -1)  # [B, 128, H*W]
            prototype_features_transposed = F.normalize(prototype_features_transposed, p=2,
                                                        dim=2)  # Normalize along the channel dimension
            deep_features_reshaped = F.normalize(deep_features_reshaped, p=2,
                                                 dim=2)  # Normalize along the channel dimension
            similarity = torch.matmul(prototype_features_transposed, deep_features_reshaped)  # [B, 6, H*W]([10, 6, 1024])

            similarity_reshaped = similarity.reshape(FeatTTG['asspF'].size(0), ProtoG['query'].size(1),
                                                     FeatTTG['asspF'].size(2), FeatTTG['asspF'].size(3))  # [B, 6, H, W]
            # G_pred = torch.argmax(similarity_reshaped.detach(), dim=1) // (
            #     self.n_clusters)  #############!!!!!!!!!!!!!!!!!!!!!!
            # distancesSoftmax = F.softmax(similarity_reshaped, dim=1)
            # distancesSoftmax = distancesSoftmax.view(distancesSoftmax.size(0),
            #                                          distancesSoftmax.size(1) // (self.n_clusters),
            #                                          (self.n_clusters),
            #                                          distancesSoftmax.size(2), distancesSoftmax.size(3))
            # distancesSoftmax = distancesSoftmax.sum(dim=2)
            for ii in range(len(name)):
                path = name[ii].split('/')[-1]
                # print(path)
                # nameList = ['area32_512_0_1024_512.png', 'area32_512_512_1024_1024.png',
                #             'area32_512_1024_1024_1536.png', 'area32_512_1536_1024_2048.png',
                #             'area32_512_2043_1024_2555.png', 'area33_0_0_512_512.png', 'area33_0_512_512_1024.png',
                #             'area33_0_1024_512_1536.png',
                #             'area33_0_1536_512_2048.png', 'area33_0_2043_512_2555.png']
                nameList = ['7_8_5488_512_6000_1024.png', '7_8_5488_5488_6000_6000.png',
                            '7_9_0_0_512_512.png', '7_9_0_1024_512_1536.png',
                            '7_9_0_1536_512_2048.png', '7_9_0_2048_512_2560.png', '7_9_0_2560_512_3072.png',
                            '7_9_0_3072_512_3584.png',
                            '7_9_0_3584_512_4096.png', '7_9_0_4096_512_4608.png']
                if path in nameList:
                    print('oook-', path)
                    np.save('./outimg/%s/%s_mask.npy' % (maskp,path.split('.')[0]), mask2.detach().cpu().numpy())
                    np.save('./outimg/%s/%s_grad_c.npy' % (maskp,path.split('.')[0]), grad_cCat.detach().cpu().numpy())
                    np.save('./outimg/%s/%s_w_c.npy' % (maskp,path.split('.')[0]), wCat_cCat.detach().cpu().numpy())
                    np.save('./outimg/%s/%s_min_c.npy' % (maskp,path.split('.')[0]), minLcat.detach().cpu().numpy())
                    np.save('./outimg/%s/%s_image.npy' % (maskp,path.split('.')[0]), data[0].detach().cpu().numpy())
                    np.save('./outimg/%s/%s_label.npy' % (maskp,path.split('.')[0]), data[1].detach().cpu().numpy())
                    np.save('./outimg/%s/%s_pseudo.npy' % (maskp,path.split('.')[0]), data[2].detach().cpu().numpy())
                    np.save('./outimg/%s/%s_similarity_reshaped.npy' % (maskp,path.split('.')[0]), similarity_reshaped.detach().cpu().numpy())


        # mask2=torch.ones_like(mask2,requires_grad=False).to(self.device)
        S_local_cCat = torch.cat(S_local_cList, dim=1)
        # print('S_local_cCat',S_local_cCat.shape)
        # Calculate weighted loss
        # print('S_local_cCat',S_local_cCat.min(),S_local_cCat.max())#[0,1]
        if mask is None:
            weighted_loss = (1 - S_local_cCat) * wCat_cCat
            weighted_loss = weighted_loss.mean()
            # weighted_loss = (1 - S_local_cCat) * wCat_cCat * mask2
            # weighted_loss = weighted_loss.sum() / (mask2.sum() + 1)
        else:

            weighted_loss = (1 - S_local_cCat) * wCat_cCat
            maskAll = mask
            weighted_loss = weighted_loss * maskAll
            weighted_loss = weighted_loss.sum() / (maskAll.sum() + 1)

        return weighted_loss
def compute_gradient_map_weight3_P(similarity_map, features, N_win_list, mask=None, lp=None, DEVICE=None, Proto=None, name=None):
    """
    Compute gradient map with weighted loss and consistency estimation.

    Parameters:
    similarity_map (torch.Tensor): Similarity map tensor of shape [B, C, H, W].
    features (torch.Tensor): Feature tensor of shape [B, C, H, W].
    N_win_list (list): List of window sizes for gradient computation.
    mask (torch.Tensor, optional): Optional mask tensor. Default is None.
    lp (torch.Tensor, optional): Label tensor. Default is None.
    DEVICE (torch.device, optional): Device for tensor operations. Default is None.
    Proto (torch.Tensor, optional): Prototype tensor. Default is None.
    name (str, optional): Name for saving mask2. Default is None.

    Returns:
    torch.Tensor: Weighted loss value.
    """
    n_clusterN = Proto.size(1) // 6
    B, _, H, W = features.shape

    # Initialize mask2 and weighted_losses
    mask2 = torch.zeros(B, 4, H, W, requires_grad=False).to(DEVICE)
    weighted_losses = torch.tensor([0.0]).to(DEVICE)
    N = 0

    # Lists to store gradients and weights
    grad_left_list = []
    grad_right_list = []
    grad_up_list = []
    grad_down_list = []
    w_list = []
    S_local_list = []

    # Iterate over N_win_list
    for idx, N_win in enumerate(N_win_list):
        if N_win == 0:
            continue
        N += 1

        # Compute gradients
        grad_left, grad_right, grad_up, grad_down = compute_gradients(
            input_tensor=similarity_map[:, idx:idx + 1, :, :], N=N_win)
        grad_left_list.append(grad_left)
        grad_right_list.append(grad_right)
        grad_up_list.append(grad_up)
        grad_down_list.append(grad_down)

        # Define directions for padding and shifting features
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        padded_features = F.pad(features, (1, 1, 1, 1), mode='constant', value=0)
        shifted_features = []
        for di, dj in directions:
            shifted = padded_features[:, :, 1 + di:H + 1 + di, 1 + dj:W + 1 + dj]
            shifted_features.append(shifted)
        shifted_features = torch.stack(shifted_features, dim=1)  # [B, 4, C, H, W]

        # Normalize features
        deep_features_normalized = F.normalize(features, p=2, dim=1)  # [B, 128, H, W]
        similaritylist = []
        for jj in range(n_clusterN):
            proto_features_normalized = F.normalize(Proto[:, idx * n_clusterN + jj], p=2, dim=1)  # [B, 128, 1, 1]
            similarity = (deep_features_normalized * proto_features_normalized).mean(dim=1).unsqueeze(1)  # [B, H, W]
            similaritylist.append(similarity)
        similaritycat = torch.cat(similaritylist, dim=1)
        # print('similaritylist',similaritycat.shape)

        similaritymax, similarityindex = similaritycat.max(dim=1)
        # print('similarityindex',similarityindex)
        Proto_flat = Proto[:, idx * n_clusterN:idx * n_clusterN + n_clusterN, :, 0, 0]
        similarityindex_flat = similarityindex.view(B, -1)  # [B, 32*32]
        Pc_selected_flat = Proto_flat.gather(1,
                                             similarityindex_flat.unsqueeze(2).expand(-1, -1, 128))  # [B, 32*32, 128]
        Pc_selected = Pc_selected_flat.view(B, H, W, 128).permute(0, 3, 1, 2)  # [B, 128, 32, 32]
        Pc_selected_expanded = Pc_selected.unsqueeze(1).expand(-1, 4, -1, -1, -1)  # [B, 4, 128, 32, 32]
        # Pc_selected_expanded_normalized = F.normalize(Pc_selected_expanded, p=2, dim=1)
        # shifted_features_normalized = F.normalize(shifted_features, p=2, dim=1)
        # print('S_local6',idx, N_win,Pc_selected_expanded.shape,shifted_features.shape,Proto_flat.shape,
        #       similarityindex_flat.shape,similarityindex_flat.min(), similarityindex_flat.max())
        print('S_local6',idx, N_win,similarityindex_flat.min(), similarityindex_flat.max())
        # S_local=(Pc_selected_expanded_normalized*shifted_features_normalized).mean(dim=2)
        S_local = F.cosine_similarity(Pc_selected_expanded, shifted_features, dim=2)  # [B, 4, 32, 32]

        S_local_list.append(S_local.unsqueeze(2))

        # Compute weights for gradients
        w_up = 1 / (1 + abs(grad_up))
        w_down = 1 / (1 + abs(grad_down))
        w_left = 1 / (1 + abs(grad_left))
        w_right = 1 / (1 + abs(grad_right))
        w_up[:, :, :N_win, :] = 0
        w_down[:, :, -N_win:, :] = 0
        w_left[:, :, :, :N_win] = 0
        w_right[:, :, :, -N_win:] = 0
        w = torch.cat([w_up.unsqueeze(1), w_down.unsqueeze(1), w_left.unsqueeze(1), w_right.unsqueeze(1)],
                      dim=1)  # [B, 4, 1, H, W]
        w_list.append(w)

    # Concatenate all the collected gradients and weights
    wCat = torch.cat(w_list, dim=2)  # [B, 4, N, H, W]
    S_local_cat = torch.cat(S_local_list, dim=2)  # [B, 4, N, H, W]
    grad_left = torch.cat(grad_left_list, dim=1)  # [B, N, H, W]
    grad_right = torch.cat(grad_right_list, dim=1)  # [B, N, H, W]
    grad_up = torch.cat(grad_up_list, dim=1)  # [B, N, H, W]
    grad_down = torch.cat(grad_down_list, dim=1)  # [B, N, H, W]

    # Ensure lp is int64 type
    lp = lp.to(torch.int64)
    lp_expanded = lp.expand(-1, 1, -1, -1)

    grad_all_directions = torch.stack([grad_left, grad_right, grad_up, grad_down], dim=1)  # [B, 4, N, H, W]
    grad_cList = []
    S_local_cList = []
    wCat_List = []

    # Compute masks and gather gradients, similarity, and weights
    for direction in range(4):
        grad_current_direction = grad_all_directions[:, direction, :, :, :]  # [B, 6, H, W]
        S_local_ = S_local_cat[:, direction, :, :, :]  # [B, 6, H, W]
        w_ = wCat[:, direction, :, :, :]  # [B, 6, H, W]
        grad_c = grad_current_direction.gather(1, lp_expanded)  # [B, 1, H, W]
        S_local_c = S_local_.gather(1, lp_expanded)  # [B, 1, H, W]
        w_c = w_.gather(1, lp_expanded)  # [B, 1, H, W]
        S_local_cList.append(S_local_c)
        is_min = (grad_current_direction >= grad_c).all(dim=1, keepdim=True)  # [B, 1, H, W]
        mask2[:, direction, :, :] = is_min.squeeze(1).float().detach()
        grad_cList.append(grad_c)
        wCat_List.append(w_c)

    grad_cCat = torch.cat(grad_cList, dim=1)
    S_local_cCat = torch.cat(S_local_cList, dim=1)
    wCat_cCat = torch.cat(wCat_List, dim=1)

    # Calculate weighted loss
    if mask is None:
        weighted_loss = (1 - S_local_cCat) * wCat_cCat * mask2
        weighted_loss = weighted_loss.sum() / (mask2.sum() + 1)
    else:
        weighted_loss = (1 - S_local_cCat) * wCat_cCat
        maskAll = mask * mask2
        weighted_loss = weighted_loss * maskAll
        weighted_loss = weighted_loss.sum() / (maskAll.sum() + 1)

    return weighted_loss

def compute_gradient_map_weight3_Pcopy(similarity_map,features,N_win_list,mask=None,lp=None,DEVICE=None,Proto=None,name=None):
    n_clusterN=Proto.size(1)//6
    B, _, H, W = features.shape
    # print(C)
    mask2 = torch.zeros(B, 4,  H, W)
    mask2 = torch.tensor(mask2,requires_grad=False).to(DEVICE)
    # weighted_losses = []
    weighted_losses = torch.tensor([0.0]).to(DEVICE)
    # weighted_losses = torch.tensor([0.0])
    N=0
    grad_left_list=[]
    grad_right_list=[]
    grad_up_list=[]
    grad_down_list=[]
    w_list=[]
    S_local_list=[]
    # grad_left
    for idx, N_win in enumerate(N_win_list):
        if N_win == 0:
            # weighted_losses=weighted_losses+
            # weighted_losses.append(torch.tensor(0.0, requires_grad=True))
            continue
        N=N+1
        grad_left, grad_right, grad_up, grad_down=compute_gradients(input_tensor=similarity_map[:, idx:idx + 1, :, :],N=N_win)
        grad_left_list.append(grad_left)
        grad_right_list.append(grad_right)
        grad_up_list.append(grad_up)
        grad_down_list.append(grad_down)
        # print(idx,'grad_left',grad_left.shape)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        padded_features = F.pad(features, (1, 1, 1, 1), mode='constant', value=0)
        shifted_features = []
        for di, dj in directions:
            shifted = padded_features[:, :, 1 + di:H + 1 + di, 1 + dj:W + 1 + dj]
            shifted_features.append(shifted)
        shifted_features = torch.stack(shifted_features, dim=1)  # [B, 4, C, H, W]
        deep_features_normalized = F.normalize(features, p=2, dim=1)  # [B, 128, H, W]
        similaritylist=[]
        for jj in range(n_clusterN):

            proto_features_normalized = F.normalize(Proto[:,idx*n_clusterN+jj], p=2, dim=1)  # [B, 128, 1, 1]
            similarity = (deep_features_normalized * proto_features_normalized).sum(dim=1).unsqueeze(1)  # [B, H, W]
            similaritylist.append(similarity)
        similaritycat=torch.cat(similaritylist,dim=1)
        similaritymax,similarityindex=similaritycat.max(dim=1)
        Proto_flat = Proto[:, idx * n_clusterN:idx * n_clusterN + n_clusterN,:,0,0]
        similarityindex_flat = similarityindex.view(B, -1)  # [B, 32*32]
        Pc_selected_flat = Proto_flat.gather(1,
                                             similarityindex_flat.unsqueeze(2).expand(-1, -1, 128))  # [B, 32*32, 128]
        Pc_selected = Pc_selected_flat.view(B, H, W, 128).permute(0, 3, 1, 2)  # [B, 128, 32, 32]
        Pc_selected_expanded = Pc_selected.unsqueeze(1).expand(-1, 4, -1, -1, -1)  # [B, 4, 128, 32, 32]
        S_local = F.cosine_similarity(Pc_selected_expanded, shifted_features, dim=2)  # [B, 4, 32, 32]

        S_local_list.append(S_local.unsqueeze(2))
        # print('S_local',S_local.shape)#[2, 4, 64, 64]
        w_up = 1 / (1 + abs(grad_up))
        w_down = 1 / (1 + abs(grad_down))
        w_left = 1 / (1 + abs(grad_left))
        w_right = 1 / (1 + abs(grad_right))
        w_up[:, :, :N_win, :] = 0
        w_down[:, :, -N_win:, :] = 0
        w_left[:, :, :, :N_win] = 0
        w_right[:, :, :, -N_win:] = 0
        w = torch.cat([w_up.unsqueeze(1), w_down.unsqueeze(1), w_left.unsqueeze(1), w_right.unsqueeze(1)],
                      dim=1)  # [B, 4, 1, H, W]
        w_list.append(w)

    wCat = torch.cat(w_list, dim=2)  # [B, 4, N, H, W]
    S_local_cat = torch.cat(S_local_list, dim=2)  # [B, 4, N, H, W]
    # print('S_local_cat', S_local_cat[0, 2, :, 0, 0], S_local_cat[0, 0, 2, 0, 0])
    grad_left = torch.cat(grad_left_list, dim=1)  # [B, N, H, W]
    grad_right = torch.cat(grad_right_list, dim=1)  # [B, N, H, W]
    grad_up = torch.cat(grad_up_list, dim=1)  # [B, N, H, W]
    grad_down = torch.cat(grad_down_list, dim=1)  # [B, N, H, W]

    # 生成 mask2
    lp = lp.to(torch.int64)  # 确保 lp 是 int64 类型
    # lp_expanded = lp.expand(-1, 1, -1, -1)  # 扩展 lp 以便于张量操作
    lp_expanded = lp
    grad_all_directions = torch.stack([grad_left, grad_right, grad_up, grad_down], dim=1)  # [B, 4, N, H, W]
    grad_cList = []
    S_local_cList = []
    wCat_List = []
    for direction in range(4):
        grad_current_direction = grad_all_directions[:, direction, :, :, :]  # [B, 6, H, W]
        S_local_ = S_local_cat[:, direction, :, :, :]  # [B, 6, H, W]
        w_ = wCat[:, direction, :, :, :]  # [B, 6, H, W]
        grad_c = grad_current_direction.gather(1, lp_expanded)  # [B, 1, H, W]
        S_local_c = S_local_.gather(1, lp_expanded)  # [B, 1, H, W]
        w_c = w_.gather(1, lp_expanded)  # [B, 1, H, W]
        S_local_cList.append(S_local_c)
        is_min = (grad_current_direction >= grad_c).all(dim=1, keepdim=True)  # [B, 1, H, W]
        mask2[:, direction, :, :] = is_min.squeeze(1).float().detach()
        grad_cList.append(grad_c)
        wCat_List.append(w_c)
        # grad_sum = grad_current_direction.sum(dim=1, keepdim=True) - grad_c  # [B, 1, H, W]
        # print('grad_sum',grad_sum.shape)
        # mask2[:, direction, :, :] = (grad_c < grad_sum).squeeze(1).float()
    if name is not None:
        for ii in range(len(name)):
            path = name[ii].split('/')[-1]
            # print(path)
            nameList=['area32_512_0_1024_512.png','area32_512_512_1024_1024.png','area32_512_1024_1024_1536.png','area32_512_1536_1024_2048.png',
                      'area32_512_2043_1024_2555.png','area33_0_0_512_512.png','area33_0_512_512_1024.png','area33_0_1024_512_1536.png',
                      'area33_0_1536_512_2048.png','area33_0_2043_512_2555.png']
            if path in nameList:
                print('oook-',path)
                np.save('./outimg/mask/%s.npy'%(path.split('.')[0]),mask2.detach().cpu().numpy())
    grad_cCat = torch.cat(grad_cList, dim=1)
    S_local_cCat = torch.cat(S_local_cList, dim=1)
    wCat_cCat = torch.cat(wCat_List, dim=1)

    if mask is None:
        weighted_loss = (1 - S_local_cCat) * wCat_cCat * mask2
        # print()
        weighted_loss = weighted_loss
        weighted_loss = weighted_loss.sum() / (mask2.sum() + 1)
    else:
        weighted_loss = (1 - S_local_cCat) * wCat_cCat
        # print(weighted_loss.shape)
        # weighted_loss = weighted_loss.sum(dim=2)
        maskAll = mask * mask2
        # print(weighted_loss.shape, maskAll.shape)
        weighted_loss = weighted_loss * maskAll
        weighted_loss = weighted_loss.sum() / (maskAll.sum() + 1)
    # for i in range(N):
    #     # print(wCat[:,:,i].shape)
    #     if mask == None:
    #         weighted_loss = (1 - S_local_cat[:,:,i].unsqueeze(2)) * wCat[:,:,i].unsqueeze(2)*mask2.unsqueeze(2)
    #         # print(wCat[:,:,i].unsqueeze(2).shape,mask2[:,:,i].shape,weighted_loss.shape)
    #         weighted_loss = weighted_loss.sum(dim=2)
    #         weighted_loss = weighted_loss.sum() / (mask2.sum() + 1)
    #
    #     else:
    #
    #         weighted_loss = (1 - S_local_cat[:,:,i].unsqueeze(2)) * wCat[:,:,i].unsqueeze(2)# [10, 4, 1, 32, 32]
    #         weighted_loss = weighted_loss.sum(dim=2)
    #         maskAll=mask*mask2
    #
    #         weighted_loss = weighted_loss * (maskAll)  # mask [10,32,32]
    #         weighted_loss = weighted_loss.sum() / (maskAll.sum() + 1)
    #     weighted_losses = weighted_losses + weighted_loss

    # weighted_losses=weighted_losses/N
    return weighted_loss
def compute_gradient_map_weight3_F(similarity_map,features,N_win_list,mask=None,lp=None,DEVICE=None,Proto=None):
    n_clusterN=Proto.size(1)//6
    B, _, H, W = features.shape
    # print(C)
    mask2 = torch.zeros(B, 4,  H, W)
    mask2 = torch.tensor(mask2,requires_grad=False).to(DEVICE)

    # weighted_losses = []
    weighted_losses = torch.tensor([0.0]).to(DEVICE)
    # weighted_losses = torch.tensor([0.0])

    N=0
    grad_left_list=[]
    grad_right_list=[]
    grad_up_list=[]
    grad_down_list=[]
    w_list=[]
    S_local_list=[]
    # grad_left
    for idx, N_win in enumerate(N_win_list):
        if N_win == 0:
            # weighted_losses=weighted_losses+
            # weighted_losses.append(torch.tensor(0.0, requires_grad=True))
            continue
        N=N+1
        grad_left, grad_right, grad_up, grad_down=compute_gradients(input_tensor=similarity_map[:, idx:idx + 1, :, :],N=N_win)
        grad_left_list.append(grad_left)
        grad_right_list.append(grad_right)
        grad_up_list.append(grad_up)
        grad_down_list.append(grad_down)
        # print(idx,'grad_left',grad_left.shape)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        padded_features = F.pad(features, (1, 1, 1, 1), mode='constant', value=0)
        shifted_features = []
        for di, dj in directions:
            shifted = padded_features[:, :, 1 + di:H + 1 + di, 1 + dj:W + 1 + dj]
            shifted_features.append(shifted)
        shifted_features = torch.stack(shifted_features, dim=1)  # [B, 4, C, H, W]
        central_features = features.unsqueeze(1).expand(-1, 4, -1, -1, -1).detach()  # [B, 4, C, H, W]
        # print('central_features',central_features.shape, shifted_features.shape)
        S_local = F.cosine_similarity(central_features, shifted_features, dim=2)  # [B, 4, H, W]
        # print('S_local',S_local.shape, shifted_features.shape)

        # directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # padded_features = F.pad(features, (1, 1, 1, 1), mode='constant', value=0)
        #
        # shifted_features = []
        # for di, dj in directions:
        #     shifted = padded_features[:, :, 1 + di:H + 1 + di, 1 + dj:W + 1 + dj]
        #     shifted_features.append(shifted)
        # shifted_features = torch.stack(shifted_features, dim=1)  # [B, 4, C, H, W]
        # deep_features_normalized = F.normalize(features, p=2, dim=1)  # [B, 128, H, W]
        # similaritylist=[]
        # for jj in range(n_clusterN):
        #
        #     proto_features_normalized = F.normalize(Proto[:,idx*n_clusterN+jj], p=2, dim=1)  # [B, 128, 1, 1]
        #     similarity = (deep_features_normalized * proto_features_normalized).sum(dim=1).unsqueeze(1)  # [B, H, W]
        #     similaritylist.append(similarity)
        # similaritycat=torch.cat(similaritylist,dim=1)
        # similaritymax,similarityindex=similaritycat.max(dim=1)
        # Proto_flat = Proto[:, idx * n_clusterN:idx * n_clusterN + n_clusterN,:,0,0]
        # similarityindex_flat = similarityindex.view(B, -1)  # [B, 32*32]
        # Pc_selected_flat = Proto_flat.gather(1,
        #                                      similarityindex_flat.unsqueeze(2).expand(-1, -1, 128))  # [B, 32*32, 128]
        # Pc_selected = Pc_selected_flat.view(B, H, W, 128).permute(0, 3, 1, 2)  # [B, 128, 32, 32]
        # Pc_selected_expanded = Pc_selected.unsqueeze(1).expand(-1, 4, -1, -1, -1)  # [B, 4, 128, 32, 32]
        # S_local = F.cosine_similarity(Pc_selected_expanded, shifted_features, dim=2)  # [B, 4, 32, 32]

        S_local_list.append(S_local.unsqueeze(2))
        # print('S_local',S_local.shape)#[2, 4, 64, 64]
        w_up = 1 / (1 + abs(grad_up))
        w_down = 1 / (1 + abs(grad_down))
        w_left = 1 / (1 + abs(grad_left))
        w_right = 1 / (1 + abs(grad_right))
        w_up[:, :, :N_win, :] = 0
        w_down[:, :, -N_win:, :] = 0
        w_left[:, :, :, :N_win] = 0
        w_right[:, :, :, -N_win:] = 0
        w = torch.cat([w_up.unsqueeze(1), w_down.unsqueeze(1), w_left.unsqueeze(1), w_right.unsqueeze(1)],
                      dim=1)  # [B, 4, 1, H, W]
        w_list.append(w)

    wCat = torch.cat(w_list, dim=2)  # [B, 4, N, H, W]
    S_local_cat = torch.cat(S_local_list, dim=2)  # [B, 4, N, H, W]
    # print('S_local_cat', S_local_cat[0, 2, :, 0, 0], S_local_cat[0, 0, 2, 0, 0])
    grad_left = torch.cat(grad_left_list, dim=1)  # [B, N, H, W]
    grad_right = torch.cat(grad_right_list, dim=1)  # [B, N, H, W]
    grad_up = torch.cat(grad_up_list, dim=1)  # [B, N, H, W]
    grad_down = torch.cat(grad_down_list, dim=1)  # [B, N, H, W]

    # 生成 mask2
    lp = lp.to(torch.int64)  # 确保 lp 是 int64 类型
    lp_expanded = lp.expand(-1, 1, -1, -1)  # 扩展 lp 以便于张量操作
    lp_expanded = lp
    grad_all_directions = torch.stack([grad_left, grad_right, grad_up, grad_down], dim=1)  # [B, 4, N, H, W]
    grad_cList = []
    S_local_cList = []
    wCat_List = []
    for direction in range(4):
        grad_current_direction = grad_all_directions[:, direction, :, :, :]  # [B, 6, H, W]
        S_local_ = S_local_cat[:, direction, :, :, :]  # [B, 6, H, W]
        w_ = wCat[:, direction, :, :, :]  # [B, 6, H, W]
        grad_c = grad_current_direction.gather(1, lp_expanded)  # [B, 1, H, W]
        S_local_c = S_local_.gather(1, lp_expanded)  # [B, 1, H, W]
        w_c = w_.gather(1, lp_expanded)  # [B, 1, H, W]
        S_local_cList.append(S_local_c)
        is_min = (grad_current_direction >= grad_c).all(dim=1, keepdim=True)  # [B, 1, H, W]
        mask2[:, direction, :, :] = is_min.squeeze(1).float().detach()
        grad_cList.append(grad_c)
        wCat_List.append(w_c)
        # grad_sum = grad_current_direction.sum(dim=1, keepdim=True) - grad_c  # [B, 1, H, W]
        # print('grad_sum',grad_sum.shape)
        # mask2[:, direction, :, :] = (grad_c < grad_sum).squeeze(1).float()

    grad_cCat = torch.cat(grad_cList, dim=1)
    S_local_cCat = torch.cat(S_local_cList, dim=1)
    wCat_cCat = torch.cat(wCat_List, dim=1)

    if mask is None:
        weighted_loss = (1 - S_local_cCat) * wCat_cCat * mask2
        # print()
        weighted_loss = weighted_loss
        weighted_loss = weighted_loss.sum() / (mask2.sum() + 1)
    else:
        weighted_loss = (1 - S_local_cCat) * wCat_cCat
        # print(weighted_loss.shape)
        # weighted_loss = weighted_loss.sum(dim=2)
        maskAll = mask * mask2
        # print(weighted_loss.shape, maskAll.shape)
        weighted_loss = weighted_loss * maskAll
        weighted_loss = weighted_loss.sum() / (maskAll.sum() + 1)
    # for i in range(N):
    #     # print(wCat[:,:,i].shape)
    #     if mask == None:
    #         weighted_loss = (1 - S_local_cat[:,:,i].unsqueeze(2)) * wCat[:,:,i].unsqueeze(2)*mask2.unsqueeze(2)
    #         # print(wCat[:,:,i].unsqueeze(2).shape,mask2[:,:,i].shape,weighted_loss.shape)
    #         weighted_loss = weighted_loss.sum(dim=2)
    #         weighted_loss = weighted_loss.sum() / (mask2.sum() + 1)
    #
    #     else:
    #
    #         weighted_loss = (1 - S_local_cat[:,:,i].unsqueeze(2)) * wCat[:,:,i].unsqueeze(2)# [10, 4, 1, 32, 32]
    #         weighted_loss = weighted_loss.sum(dim=2)
    #         maskAll=mask*mask2
    #
    #         weighted_loss = weighted_loss * (maskAll)  # mask [10,32,32]
    #         weighted_loss = weighted_loss.sum() / (maskAll.sum() + 1)
    #     weighted_losses = weighted_losses + weighted_loss

    # weighted_losses=weighted_losses/N
    return weighted_loss
def compute_gradient_map_weightP(similarity_map,features,Proto,N_win_list,DEVICE,mask=None):
    B, C, H, W = features.shape
    n_clusterN=Proto.size(1)//6
    # weighted_losses = []
    weighted_losses = torch.tensor([0.0]).to(DEVICE)
    # if mask is not None:
    #     mask = mask.unsqueeze(1)
    N=0
    for idx, N_win in enumerate(N_win_list):
        if N_win == 0:
            # weighted_losses=weighted_losses+
            # weighted_losses.append(torch.tensor(0.0, requires_grad=True))
            continue
        N=N+1
        grad_left, grad_right, grad_up, grad_down=compute_gradients(input_tensor=similarity_map[:, idx:idx + 1, :, :],N=N_win)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        padded_features = F.pad(features, (1, 1, 1, 1), mode='constant', value=0)
        shifted_features = []
        for di, dj in directions:
            shifted = padded_features[:, :, 1 + di:H + 1 + di, 1 + dj:W + 1 + dj]
            shifted_features.append(shifted)
        shifted_features = torch.stack(shifted_features, dim=1)  # [B, 4, C, H, W]
        deep_features_normalized = F.normalize(features, p=2, dim=1)  # [B, 128, H, W]
        similaritylist=[]
        for jj in range(n_clusterN):
            proto_features_normalized = F.normalize(Proto[:,idx*n_clusterN+jj], p=2, dim=1)  # [B, 128, 1, 1]
            similarity = (deep_features_normalized * proto_features_normalized).sum(dim=1).unsqueeze(1)  # [B, H, W]
            similaritylist.append(similarity)
        similaritycat=torch.cat(similaritylist,dim=1)
        similaritymax,similarityindex=similaritycat.max(dim=1)
        Proto_flat = Proto[:, idx * n_clusterN:idx * n_clusterN + n_clusterN,:,0,0]
        # print('Proto_flat',Proto_flat.shape)
        # Proto_flat = Pc.view(B, n_clusterN, 128)  # [B, 2, 128]
        similarityindex_flat = similarityindex.view(B, -1)  # [B, 32*32]
        Pc_selected_flat = Proto_flat.gather(1,
                                             similarityindex_flat.unsqueeze(2).expand(-1, -1, 128))  # [B, 32*32, 128]
        Pc_selected = Pc_selected_flat.view(B, H, W, 128).permute(0, 3, 1, 2)  # [B, 128, 32, 32]
        Pc_selected_expanded = Pc_selected.unsqueeze(1).expand(-1, 4, -1, -1, -1)  # [B, 4, 128, 32, 32]
        # print('shifted_features',shifted_features.shape,Pc_selected_expanded.shape)
        S_local = F.cosine_similarity(Pc_selected_expanded, shifted_features, dim=2)  # [B, 4, 32, 32]


        #
        # central_features = features.unsqueeze(1).expand(-1, 4, -1, -1, -1)  # [B, 4, C, H, W]
        # S_local = F.cosine_similarity(central_features, shifted_features, dim=2)  # [B, 4, H, W]
        w_up = 1 / (1 + abs(grad_up))
        w_down = 1 / (1 + abs(grad_down))
        w_left = 1 / (1 + abs(grad_left))
        w_right = 1 / (1 + abs(grad_right))
        w_up[:, :, :N_win, :] = 0
        w_down[:, :, -N_win:, :] = 0
        w_left[:, :, :, :N_win] = 0
        w_right[:, :, :, -N_win:] = 0
        w = torch.cat([w_up.unsqueeze(1), w_down.unsqueeze(1), w_left.unsqueeze(1), w_right.unsqueeze(1)],
                      dim=1)  # [B, 4, 1, H, W]
        if mask ==None:
            weighted_loss=(1 - S_local.unsqueeze(2)) * w
            weighted_loss = weighted_loss.sum(dim=2).mean()

        else:
            # mask=mask.unsqueeze(1).unsqueeze(2)
            # print('mask',mask.shape,S_local.shape,w.shape)
            weighted_loss = (1 - S_local.unsqueeze(2)) * w  # [10, 4, 1, 32, 32]

            # print('weighted_loss',weighted_loss.shape)
            # print(weighted_loss.sum(dim=2).shape,mask.shape,mask.sum())
            weighted_loss = weighted_loss.sum(dim=2) * (mask)  # mask [10,32,32]
            # print('weighted_loss1', weighted_loss.shape)
            weighted_loss = weighted_loss.sum() / (mask.sum() + 1)
        # weighted_loss=(1 - S_local.unsqueeze(2)) * w
        # weighted_loss = weighted_loss.sum(dim=2).mean()

        weighted_losses=weighted_losses+weighted_loss
    weighted_losses=weighted_losses/N
    return weighted_losses
class DeepLabV3PlusSimGlobalLinearKL3neibor(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimGlobalLinearKL3neibor, self).__init__()
        # self.resnet = models.resnet101(weights = models.ResNet101_Weights.DEFAULT)
        # if args.mode == 'imp':
        #     pretrained = '../pretrain_model/resnet50-19c8e357.pth'
        #     pretrained_weights = torch.load(pretrained)
        #     self.resnet.load_state_dict(pretrained_weights)
        # elif args.mode == 'rsp_120':
        #     pretrained = '../RS_CLS_finetune/output/resnet_50_224/epoch120/millionAID_224_None/0.0005_0.05_192/resnet/100/ckpt.pth'
        #     pretrained_weights = torch.load(pretrained)
        #     self.resnet.load_state_dict(pretrained_weights)
        # elif args.mode == 'rsp_300':
        #     self.resnet = ResNet(args)
        # elif args.mode == 'seco':
        #     pretrained = '../pretrain_model/seco_resnet50_1m.pth'
        #     pretrained_weights = torch.load(pretrained)
        #     self.resnet.load_state_dict(pretrained_weights)
        # elif args.mode == 'office':
        #     self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # elif args.mode == 'nopre':
        #     self.resnet = models.resnet50(weights=None)
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.num_classes = num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(featDim),
            nn.ReLU(), )

        self.classifier = nn.Conv2d(featDim, num_classes, 1)
        self.n_cluster = n_cluster
        self.prototypeN = 1

        Sim=nn.Sequential(
            nn.Conv2d(featDim, featDim, kernel_size=3,  padding=1),

            )
        for i in range(num_classes):
            setattr(self, f'sim_{i}', Sim)
        self.adainpro=ProAdaIN()

    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None,getPFlag=False):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        assp_features = self.aspp(x4)  # [10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        if getPFlag == True:
            # print('getPFlag',getPFlag)
            return assp_features
        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        prototypes = []
        if ProtoInput == None:
            with torch.no_grad():
                pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
                #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
                mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
            # 计算每个类别的原型
                for i in range(self.num_classes):
                    class_mask = (mask == i).float()
                    if maskParam is not None:
                        class_mask = class_mask * maskParam  # Apply maskParam
                    if class_mask.sum() > 0:  # 确保类别在批次中存在
                        prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                                    class_mask.sum(dim=[2, 3]) + 1e-5)  # ([10, 256])
                        prototypes.append(prototype.unsqueeze(1))
                    else:
                        prototypes.append(zero_prototype.unsqueeze(1))
                prototypesC = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze( -1)  # prototypes torch.Size([10, 6, 128, 1, 1])
                prototypes = prototypesC
        elif ProtoInput != None:
            prototypes = ProtoInput
        query_outputList = []
        similarityList = []
        # key_output = self.key(assp_features)
        key_output=assp_features
        for ii in range(prototypes.size(1)):
            prototype = prototypes[:, ii]
            if prototype is not None:
                ######key
                #######query
                # deep_features_normalized,proto_features_normalized=self.adainpro(key_output,prototype)
                deep_features_normalized = F.normalize(key_output, p=2, dim=1)  # [B, 128, H, W]
                proto_features_normalized = F.normalize(prototype, p=2, dim=1)  # [B, 128, 1, 1]
                similarity = (deep_features_normalized * proto_features_normalized) # [B, H, W]
                similarity = getattr(self, f'sim_{ii // (prototypes.size(1) // 6)}') (similarity)
                similarity = similarity.sum(dim=1).unsqueeze(1) # [B, H, W]
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                             assp_features.size(3))  # ([10, 1, 16, 16])
                similarityList.append(similarity)
                query_output = prototype.view(prototype.size(0), -1,
                                                 prototype.size(1))  # Reshape to [10, 1, 128]
                query_outputList.append(query_output)
        query_outputcat = torch.cat(query_outputList, dim=1).unsqueeze(-1).unsqueeze(-1)
        similarityCat = torch.cat(similarityList, dim=1)#([10, 30, 32, 32])
        if getPFlag==2:
            similarityCatMax=similarityCat
        else:
            similarityCatMax = similarityCat.view(similarityCat.size(0), similarityCat.size(1) // (self.n_cluster),
                                                  (self.n_cluster),
                                     similarityCat.size(2),
                                     similarityCat.size(3))
            similarityCatMax=similarityCatMax.max(dim=2)[0]
        # G=self.compute_gradients(similarityCatMax,kernel_size=3)#([10, 36, 32, 32])
        # G_x, G_y, G = self.compute_gradient_map(similarityCatMax,neiborWindow=5)

        # print('G',G.shape)#
        # SSoftLog = torch.log(SSoftLog.max(dim=2)[0])

        similarityWeiht = F.softmax(similarityCat, dim=1)#torch.Size([10, 30, 32, 32])
        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, index = torch.max(similarityWeiht, dim=1)
        # similarityCatMax=similarityCat[index]

        # print('similarityCatMax',similarityCatMax.shape)

        assp_weighted = assp_features * similarityWeihtMax.unsqueeze(1)
        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': xup}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,'catmax':similarityCatMax,
                'Weight': [similarityWeiht, similarityWeihtMax]}



    # def compute_gradients(self,S, kernel_size=3):
    #     B, C, W, H = S.shape
    #
    #     # 定义邻域核，计算3x3邻域的和
    #     kernel = torch.ones((1, 1, kernel_size, kernel_size), device=S.device)
    #     kernel[0, 0, kernel_size // 2, kernel_size // 2] = 0  # 中心值为0
    #
    #     # 计算梯度
    #     G = torch.zeros((B, C * C, W, H), device=S.device)
    #
    #     for i in range(C):
    #         S_i = S[:, i:i + 1, :, :]
    #         for j in range(C):
    #             S_j = S[:, j:j + 1, :, :]
    #             with torch.no_grad():
    #             # 计算 S_j 的邻域和
    #                 S_j_sum = F.conv2d(S_j, kernel, padding=kernel_size // 2,)
    #
    #             # 计算梯度
    #             gradient = torch.abs(S_j_sum - S_i)  # 使用绝对值避免负值影响
    #             G[:, i * C + j, :, :] = gradient.squeeze(1)
    #
    #     return G
    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []
        b.append(self.resnet.conv1)
        b.append(self.resnet.bn1)
        b.append(self.resnet.layer1)
        b.append(self.resnet.layer2)
        b.append(self.resnet.layer3)
        b.append(self.resnet.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k
    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.aspp.parameters())
        b.append(self.classifier.parameters())
        # b.append(self.layer7.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self):
        learning_rate=2.5e-4
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * learning_rate}]
def neighborhood_relation_loss(G, S, C):
    B, _, W, H = G.shape
    G = G.view(B, C, C, W, H)

    loss = 0.0
    for i in range(C):
        for j in range(C):
            if i != j:
                # 计算邻域关系损失，使得相似度矩阵在邻域内更加一致
                # loss += torch.mean(G[:, i, j, :, :] * S[:, i, :, :] * (1 - S[:, j, :, :]))
                # loss += torch.mean(G[:, i, j, :, :] * S[:, i, :, :] * (1/ (S[:, j, :, :]+0.0001)))
                loss += torch.mean((1/(G[:, i, j, :, :]+0.0001))* (1/(S[:, i, :, :]+0.0001)) * S[:, j, :, :])


    return loss
class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()
    def calc_mean_std(self, features, eps=1e-5):
        N, C, H, W = features.size()
        features = features.view(N, C, -1)
        mean = features.mean(dim=2).view(N, C, 1, 1)
        std = features.std(dim=2).view(N, C, 1, 1) + eps
        return mean, std

    def forward(self, content_features, style_features):
        content_mean, content_std = self.calc_mean_std(content_features)
        style_mean, style_std = self.calc_mean_std(style_features)
        normalized_features = (content_features - content_mean) / content_std
        return normalized_features * style_std + style_mean



class DeepLabV3PlusSimGlobalLinearKL3Adain(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimGlobalLinearKL3Adain, self).__init__()
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
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode == 'nopre':
            self.resnet = models.resnet50(weights=None)
        self.num_classes = num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(featDim),
            nn.ReLU(), )

        self.classifier = nn.Conv2d(featDim, num_classes, 1)
        self.n_cluster = n_cluster
        self.prototypeN = 1
        # self.key = nn.Conv2d(featDim, featDim, kernel_size=3, padding=1)
        # self.sc=nn.Conv2d(featDim, featDim, kernel_size=3,  padding=1)
        self.adain = AdaIN()
        Sim=nn.Sequential(
            nn.Conv2d(featDim, featDim, kernel_size=3,  padding=1),
            )
        for i in range(num_classes):
            # setattr(self, f'sim_{i}', nn.Conv2d(featDim, featDim, kernel_size=3,  padding=1))
            setattr(self, f'sim_{i}', Sim)
            # setattr(self, f'a_{i}', nn.Parameter(torch.ones(1), requires_grad=True))
            # setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1), requires_grad=True))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
        # self.normP=nn.LayerNorm(128)
    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None,getPFlag=False,target_features=None):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        assp_features = self.aspp(x4)  # [10, 256, 16, 16])
        self.assp_features=assp_features
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)

        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        # ProtoInput=ProtoInput.reshape()
        if DomainLabel==0 and target_features is not None:
            assp_features = self.adain(assp_features, target_features)
        # else:
        #     assp_features = self.adain(assp_features, assp_features)
        if getPFlag:
            return assp_features

        prototypes = []
        if ProtoInput == None:
            with torch.no_grad():
                pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
                # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
                mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
                # 计算每个类别的原型
                for i in range(self.num_classes):
                    class_mask = (mask == i).float()
                    if maskParam is not None:
                        class_mask = class_mask * maskParam  # Apply maskParam
                    if class_mask.sum() > 0:  # 确保类别在批次中存在
                        prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                                    class_mask.sum(dim=[2, 3]) + 1e-5)  # ([10, 256])
                        prototypes.append(prototype.unsqueeze(1))
                    else:
                        # for pp in range(self.n_cluster):
                        prototypes.append(zero_prototype.unsqueeze(1))
                prototypesC = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze( -1)  # prototypes torch.Size([10, 6, 128, 1, 1])
                prototypes = prototypesC
        elif ProtoInput != None:
            prototypes = ProtoInput
        query_outputList = []
        # prototypesOut = []
        similarityList = []
        # key_output = self.key(assp_features)
        key_output=assp_features
        for ii in range(prototypes.size(1)):
            prototype = prototypes[:, ii]
            if prototype is not None:
                ######key
                # key_output=getattr(self, f'a_{ii//(prototypes.size(1)//6)}') * key_output + getattr(self, f'b_{ii//((prototypes.size(1)//6))}')
                # query_output = prototype.view(prototype.size(0), -1, prototype.size(1))  # Reshape to [10, 1, 128]
                #######query
                deep_features_normalized = F.normalize(key_output, p=2, dim=1)  # [B, 128, H, W]
                proto_features_normalized = F.normalize(prototype, p=2, dim=1)  # [B, 128, 1, 1]
                similarity = (deep_features_normalized * proto_features_normalized) # [B, H, W]
                similarity = getattr(self, f'sim_{ii // (prototypes.size(1) // 6)}') (similarity)
                similarity = similarity.sum(dim=1).unsqueeze(1) # [B, H, W]
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                             assp_features.size(3))  # ([10, 1, 16, 16])
                similarityList.append(similarity)
                query_output = prototype.view(prototype.size(0), -1,
                                                 prototype.size(1))  # Reshape to [10, 1, 128]
                query_outputList.append(query_output)

        query_outputcat = torch.cat(query_outputList, dim=1).unsqueeze(-1).unsqueeze(-1)
        # print('query_outputcat',query_outputcat.shape,prototypes.shape,GlobalProto_transOut.shape)
        similarityCat = torch.cat(similarityList, dim=1)#([10, 30, 32, 32])
        similarityWeiht = F.softmax(similarityCat, dim=1)#torch.Size([10, 30, 32, 32])
        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        assp_weighted = assp_features * similarityWeihtMax.unsqueeze(1)
        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': xup}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,
                'Weight': [similarityWeiht, similarityWeihtMax]}

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []
        b.append(self.resnet.conv1)
        b.append(self.resnet.bn1)
        b.append(self.resnet.layer1)
        b.append(self.resnet.layer2)
        b.append(self.resnet.layer3)
        b.append(self.resnet.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k
    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.aspp.parameters())
        b.append(self.classifier.parameters())
        # b.append(self.layer7.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self):
        learning_rate=2.5e-4
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * learning_rate}]


class ProAdaIN(nn.Module):
    def __init__(self):
        super(ProAdaIN, self).__init__()

    def calc_mean_std(self, features, eps=1e-5):
        N, C, H, W = features.size()
        features = features.view(N, C, -1)
        mean = features.mean(dim=2).view(N, C, 1, 1)
        std = features.std(dim=2).view(N, C, 1, 1) + eps
        return mean, std

    def forward(self, key_output, prototype):
        N, C, H, W = key_output.size()
        # content_mean, content_std = self.calc_mean_std(content_features)
        style_mean = prototype.mean(1).view(N, 1, 1, 1)
        style_std = prototype.std(dim=1).view(N, 1, 1, 1) + 1e-5
        # style_mean, style_std = self.calc_mean_std(style_features)
        # normalized_features = (content_features - content_mean) / content_std
        # deep_features_normalized = F.normalize(key_output, p=2, dim=1)  # [B, 128, H, W]
        # proto_features_normalized = F.normalize(prototype, p=2, dim=1)  # [B, 128, 1, 1]
        content_mean, content_std = self.calc_mean_std(key_output)
        deep_features_normalized = (key_output - content_mean) / content_std
        deep_features_normalized=deep_features_normalized * style_std + style_mean
        # proto_features_normalized=(prototype-style_mean)/ style_std
        # proto_features_normalized=proto_features_normalized * style_std + style_mean
        return deep_features_normalized, prototype

class DeepLabV3PlusSimGlobalLinearKLProAdaIN(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimGlobalLinearKLProAdaIN, self).__init__()
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
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode == 'nopre':
            self.resnet = models.resnet50(weights=None)
        self.num_classes = num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(featDim),
            nn.ReLU(), )

        self.classifier = nn.Conv2d(featDim, num_classes, 1)
        self.n_cluster = n_cluster
        self.prototypeN = 1
        # self.key = nn.Conv2d(featDim, featDim, kernel_size=3, padding=1)
        # self.sc=nn.Conv2d(featDim, featDim, kernel_size=3,  padding=1)
        # self.adain = AdaIN()
        self.adainpro = ProAdaIN()

        Sim=nn.Sequential(
            nn.Conv2d(featDim, featDim, kernel_size=3,  padding=1),
            )
        for i in range(num_classes):
            # setattr(self, f'sim_{i}', nn.Conv2d(featDim, featDim, kernel_size=3,  padding=1))
            setattr(self, f'sim_{i}', Sim)
            # setattr(self, f'a_{i}', nn.Parameter(torch.ones(1), requires_grad=True))
            # setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1), requires_grad=True))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
        # self.normP=nn.LayerNorm(128)
    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None,getPFlag=False):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        assp_features = self.aspp(x4)  # [10, 256, 16, 16])
        self.assp_features=assp_features
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)

        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        # ProtoInput=ProtoInput.reshape()
        # if DomainLabel==0 and target_features is not None:
        #     assp_features = self.adain(assp_features, target_features)
        # else:
        #     assp_features = self.adain(assp_features, assp_features)
        if getPFlag:
            return assp_features

        prototypes = []
        if ProtoInput == None:
            with torch.no_grad():
                pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
                # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
                mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
                # 计算每个类别的原型
                for i in range(self.num_classes):
                    class_mask = (mask == i).float()
                    if maskParam is not None:
                        class_mask = class_mask * maskParam  # Apply maskParam
                    if class_mask.sum() > 0:  # 确保类别在批次中存在
                        prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                                    class_mask.sum(dim=[2, 3]) + 1e-5)  # ([10, 256])
                        prototypes.append(prototype.unsqueeze(1))
                    else:
                        # for pp in range(self.n_cluster):
                        prototypes.append(zero_prototype.unsqueeze(1))
                prototypesC = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze( -1)  # prototypes torch.Size([10, 6, 128, 1, 1])
                prototypes = prototypesC
        elif ProtoInput != None:
            prototypes = ProtoInput
        query_outputList = []
        # prototypesOut = []
        similarityList = []
        # key_output = self.key(assp_features)
        key_output=assp_features
        for ii in range(prototypes.size(1)):
            prototype = prototypes[:, ii]
            if prototype is not None:
                ######key
                # key_output=getattr(self, f'a_{ii//(prototypes.size(1)//6)}') * key_output + getattr(self, f'b_{ii//((prototypes.size(1)//6))}')
                # query_output = prototype.view(prototype.size(0), -1, prototype.size(1))  # Reshape to [10, 1, 128]
                #######query
                deep_features_normalized,proto_features_normalized=self.adainpro(key_output,prototype)
                deep_features_normalized = F.normalize(deep_features_normalized, p=2, dim=1)  # [B, 128, H, W]
                proto_features_normalized = F.normalize(proto_features_normalized, p=2, dim=1)  # [B, 128, 1, 1]
                similarity = (deep_features_normalized * proto_features_normalized) # [B, H, W]
                similarity = getattr(self, f'sim_{ii // (prototypes.size(1) // 6)}') (similarity)
                similarity = similarity.sum(dim=1).unsqueeze(1) # [B, H, W]
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                             assp_features.size(3))  # ([10, 1, 16, 16])
                similarityList.append(similarity)
                query_output = prototype.view(prototype.size(0), -1,
                                                 prototype.size(1))  # Reshape to [10, 1, 128]
                query_outputList.append(query_output)

        query_outputcat = torch.cat(query_outputList, dim=1).unsqueeze(-1).unsqueeze(-1)
        # print('query_outputcat',query_outputcat.shape,prototypes.shape,GlobalProto_transOut.shape)
        similarityCat = torch.cat(similarityList, dim=1)#([10, 30, 32, 32])
        similarityWeiht = F.softmax(similarityCat, dim=1)#torch.Size([10, 30, 32, 32])
        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        assp_weighted = assp_features * similarityWeihtMax.unsqueeze(1)
        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': xup}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,
                'Weight': [similarityWeiht, similarityWeihtMax]}

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []
        b.append(self.resnet.conv1)
        b.append(self.resnet.bn1)
        b.append(self.resnet.layer1)
        b.append(self.resnet.layer2)
        b.append(self.resnet.layer3)
        b.append(self.resnet.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k
    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.aspp.parameters())
        b.append(self.classifier.parameters())
        # b.append(self.layer7.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self):
        learning_rate=2.5e-4
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * learning_rate}]

class DeepLabV3PlusSimGlobalLinearKLProAdaIN2(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimGlobalLinearKLProAdaIN2, self).__init__()
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
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode == 'nopre':
            self.resnet = models.resnet50(weights=None)
        self.num_classes = num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(featDim),
            nn.ReLU(), )

        self.classifier = nn.Conv2d(featDim, num_classes, 1)
        self.n_cluster = n_cluster
        self.prototypeN = 1
        # self.key = nn.Conv2d(featDim, featDim, kernel_size=3, padding=1)
        # self.sc=nn.Conv2d(featDim, featDim, kernel_size=3,  padding=1)
        self.adain = AdaIN()
        self.adainpro = ProAdaIN()

        Sim=nn.Sequential(
            nn.Conv2d(featDim, featDim, kernel_size=3,  padding=1),
            )
        for i in range(num_classes):
            # setattr(self, f'sim_{i}', nn.Conv2d(featDim, featDim, kernel_size=3,  padding=1))
            setattr(self, f'sim_{i}', Sim)
            # setattr(self, f'a_{i}', nn.Parameter(torch.ones(1), requires_grad=True))
            # setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1), requires_grad=True))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
        # self.normP=nn.LayerNorm(128)
    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None,getPFlag=False,target_features=None):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)

        x4 = self.resnet.layer4(x3)
        assp_features = self.aspp(x4)  # [10, 256, 16, 16])
        self.assp_features=assp_features
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)

        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        # ProtoInput=ProtoInput.reshape()
        if DomainLabel==0 and target_features is not None:
            assp_featuresN = self.adain(assp_features, target_features)
        else:
            # assp_features = self.adain(assp_features, assp_features)
            assp_featuresN=assp_features
        if getPFlag:
            return assp_features

        prototypes = []
        if ProtoInput == None:
            with torch.no_grad():
                pseudo_out = self.classifier(assp_featuresN)  # ([10, 6, 16, 16])
                # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
                mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
                # 计算每个类别的原型
                for i in range(self.num_classes):
                    class_mask = (mask == i).float()
                    if maskParam is not None:
                        class_mask = class_mask * maskParam  # Apply maskParam
                    if class_mask.sum() > 0:  # 确保类别在批次中存在
                        prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                                    class_mask.sum(dim=[2, 3]) + 1e-5)  # ([10, 256])
                        prototypes.append(prototype.unsqueeze(1))
                    else:
                        # for pp in range(self.n_cluster):
                        prototypes.append(zero_prototype.unsqueeze(1))
                prototypesC = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze( -1)  # prototypes torch.Size([10, 6, 128, 1, 1])
                prototypes = prototypesC
        elif ProtoInput != None:
            prototypes = ProtoInput
        query_outputList = []
        # prototypesOut = []
        similarityList = []
        # key_output = self.key(assp_features)
        key_output=assp_features
        for ii in range(prototypes.size(1)):
            prototype = prototypes[:, ii]
            if prototype is not None:
                ######key
                # key_output=getattr(self, f'a_{ii//(prototypes.size(1)//6)}') * key_output + getattr(self, f'b_{ii//((prototypes.size(1)//6))}')
                # query_output = prototype.view(prototype.size(0), -1, prototype.size(1))  # Reshape to [10, 1, 128]
                #######query
                deep_features_normalized,proto_features_normalized=self.adainpro(key_output,prototype)
                deep_features_normalized = F.normalize(deep_features_normalized, p=2, dim=1)  # [B, 128, H, W]
                proto_features_normalized = F.normalize(proto_features_normalized, p=2, dim=1)  # [B, 128, 1, 1]
                similarity = (deep_features_normalized * proto_features_normalized) # [B, H, W]
                similarity = getattr(self, f'sim_{ii // (prototypes.size(1) // 6)}') (similarity)
                similarity = similarity.sum(dim=1).unsqueeze(1) # [B, H, W]
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                             assp_features.size(3))  # ([10, 1, 16, 16])
                similarityList.append(similarity)
                query_output = prototype.view(prototype.size(0), -1,
                                                 prototype.size(1))  # Reshape to [10, 1, 128]
                query_outputList.append(query_output)

        query_outputcat = torch.cat(query_outputList, dim=1).unsqueeze(-1).unsqueeze(-1)
        # print('query_outputcat',query_outputcat.shape,prototypes.shape,GlobalProto_transOut.shape)
        similarityCat = torch.cat(similarityList, dim=1)#([10, 30, 32, 32])
        similarityWeiht = F.softmax(similarityCat, dim=1)#torch.Size([10, 30, 32, 32])
        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        assp_weighted = assp_features * similarityWeihtMax.unsqueeze(1)
        x = self.classifier(assp_featuresN)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': xup}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,
                'Weight': [similarityWeiht, similarityWeihtMax]}

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []
        b.append(self.resnet.conv1)
        b.append(self.resnet.bn1)
        b.append(self.resnet.layer1)
        b.append(self.resnet.layer2)
        b.append(self.resnet.layer3)
        b.append(self.resnet.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k
    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.aspp.parameters())
        b.append(self.classifier.parameters())
        # b.append(self.layer7.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self):
        learning_rate=2.5e-4
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * learning_rate}]


class DeepLabV3PlusSimGlobalLinearKLBN(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimGlobalLinearKLBN, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.num_classes = num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), )

        self.classifier = nn.Conv2d(featDim, num_classes, 1)

        self.n_cluster = n_cluster
        self.prototypeN = 1
        # self.key = nn.Conv2d(featDim, featDim, kernel_size=3, padding=1)
        for i in range(num_classes):
            # setattr(self, f'query_{i}', nn.Conv2d(featDim * self.prototypeN, 128, kernel_size=1))
            setattr(self, f'a_{i}', nn.Parameter(torch.ones(1), requires_grad=True))
            setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1), requires_grad=True))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
        self.normP=nn.LayerNorm(128)
    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None,getPFlag=False):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)  # [10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        if getPFlag:
            return assp_features
        prototypes = []
        if ProtoInput == None:
            with torch.no_grad():
                pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
                #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
                mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
            # 计算每个类别的原型
                for i in range(self.num_classes):
                    class_mask = (mask == i).float()
                    if maskParam is not None:
                        class_mask = class_mask * maskParam  # Apply maskParam
                    if class_mask.sum() > 0:  # 确保类别在批次中存在
                        # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                        prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                                    class_mask.sum(dim=[2, 3]) + 1e-5)  # ([10, 256])
                        prototypes.append(prototype.unsqueeze(1))
                    else:
                        prototypes.append(zero_prototype.unsqueeze(1))
                prototypesC = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze( -1)  # prototypes torch.Size([10, 6, 128, 1, 1])
                prototypes = prototypesC
        elif ProtoInput != None:
            prototypes = ProtoInput
        query_outputList = []
        # prototypesOut = []
        similarityList = []
        # key_output = self.key(assp_features)
        key_output=assp_features
        for ii in range(prototypes.size(1)):
            prototype = prototypes[:, ii]
            if prototype is not None:

                deep_features_normalized = F.normalize(key_output, p=2, dim=1)  # [B, 128, H, W]
                proto_features_normalized = F.normalize(prototype, p=2, dim=1)  # [B, 128, 1, 1]
                similarity = (deep_features_normalized * proto_features_normalized) # [B, H, W]

                similarity = similarity.sum(dim=1).unsqueeze(1) # [B, H, W]

                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                             assp_features.size(3))  # ([10, 1, 16, 16])
                similarityList.append(similarity)
                query_output = prototype.view(prototype.size(0), -1,
                                                 prototype.size(1))  # Reshape to [10, 1, 128]
                query_outputList.append(query_output)

        query_outputcat = torch.cat(query_outputList, dim=1).unsqueeze(-1).unsqueeze(-1)
        # print('query_outputcat',query_outputcat.shape,prototypes.shape,GlobalProto_transOut.shape)
        similarityCat = torch.cat(similarityList, dim=1)#([10, 30, 32, 32])
        similarityWeiht = F.softmax(similarityCat, dim=1)#torch.Size([10, 30, 32, 32])
        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        assp_weighted = assp_features * similarityWeihtMax.unsqueeze(1)
        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': xup}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,
                'Weight': [similarityWeiht, similarityWeihtMax]}

# class SemanticSegmentationCosineLoss(nn.Module):
#     def __init__(self, num_classes):
#         super(SemanticSegmentationCosineLoss, self).__init__()
#         self.num_classes = num_classes
#     def forward(self, features, labels):
#         """
#         features: tensor of shape [B, 128, 32, 32], deep features
#         labels: tensor of shape [B, 32, 32], class labels
#         """
#         B, C, H, W = features.size()
#         lossSame = 0.0
#         lossDiff = 0.0
#
#         for cls in range(self.num_classes):
#             # Create a mask for the current class
#             mask = (labels == cls).unsqueeze(1).float()  # [B, 1, 32, 32]
#             # Compute the number of pixels in the current class
#             num_pixels = mask.sum(dim=(2, 3), keepdim=True)  # [B, 1, 1, 1]
#             # Avoid division by zero
#             num_pixels[num_pixels == 0] = 1
#             # Compute the mean feature for the current class
#             mean_feature = (features * mask).sum(dim=(2, 3), keepdim=True) / num_pixels  # [B, C, 1, 1]
#             # Normalize the features and the mean feature
#             features_norm = F.normalize(features, p=2, dim=1)  # [B, C, 32, 32]
#             mean_feature_norm = F.normalize(mean_feature.detach(), p=2, dim=1)  # [B, C, 1, 1]
#             # Compute the cosine similarity
#             cosine_similarity = (features_norm * mean_feature_norm).sum(dim=1, keepdim=True)  # [B, 1, 32, 32]
#             # Convert cosine similarity to cosine distance (1 - cosine similarity)
#             cosine_distance = 1 - cosine_similarity  # [B, 1, 32, 32]
#             # Compute the loss for the current class
#             lossSame += (cosine_distance * mask).sum() / num_pixels.sum()
#             lossDiff += (cosine_distance * (1-mask)).sum() / (H*W-num_pixels.sum())
#
#         return lossSame / B- lossDiff / B

class SemanticSegmentationCosineLoss(nn.Module):
    def __init__(self, num_classes):
        super(SemanticSegmentationCosineLoss, self).__init__()
        self.num_classes = num_classes
    def forward(self, features, labels):
        """
        features: tensor of shape [B, 128, 32, 32], deep features
        labels: tensor of shape [B, 32, 32], class labels
        """
        B, C, H, W = features.size()
        loss_same = 0.0
        loss_diff = 0.0
        mean_features = []
        for cls in range(self.num_classes):
            mask = (labels == cls).float()
            num_pixels = mask.sum(dim=(2, 3), keepdim=True)  # [B, 1, 1, 1]
            # ([10, 256])
            mean_feature = (features * mask).sum(dim=[2, 3]) / (mask.sum(dim=[2, 3]) + 1e-5)
            # mean_features.append(mean_feature.unsqueeze(1))
            mean_features.append(mean_feature)  # Detach to prevent gradient flow
            # Normalize the features and the mean feature
            features_norm = F.normalize(features, p=2, dim=1)  # [B, C, 32, 32]
            mean_feature_norm = F.normalize(mean_feature, p=2, dim=1).unsqueeze(-1).unsqueeze(-1)   # [B, C, 1, 1]
            # Compute the cosine similarity
            # print(features_norm.shape, mean_feature_norm.shape)
            cosine_similarity = (features_norm * mean_feature_norm).sum(dim=1, keepdim=True)  # [B, 1, 32, 32]
            # Convert cosine similarity to cosine distance (1 - cosine similarity)
            cosine_distance = 1 - cosine_similarity  # [B, 1, 32, 32]
            # Compute the loss for the current class
            loss_same += (cosine_distance * mask).sum() / (mask.sum()+1)
            # loss_diff += (cosine_distance * (1-mask)).sum() / ((1-mask).sum()+1)
        # Compute the inter-class cosine similarity (maximize the difference)
        mean_features = torch.stack(mean_features).unsqueeze(-1).unsqueeze(-1)   # [num_classes, B, C, 1, 1]
        # print('mean_features',mean_features.shape)
        mean_features = mean_features.mean(dim=1)  # [num_classes, C, 1, 1]
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                mean_feature_i = F.normalize(mean_features[i], p=2, dim=0)  # [C, 1, 1]
                mean_feature_j = F.normalize(mean_features[j], p=2, dim=0)  # [C, 1, 1]
                cosine_similarity = (mean_feature_i * mean_feature_j).sum()  # scalar
                loss_diff += cosine_similarity

            # Normalize loss_diff by the number of comparisons
        num_comparisons = self.num_classes * (self.num_classes - 1) / 2
        loss_diff /= num_comparisons
        return loss_same / B - 0.1*loss_diff

        # return loss_same / B - loss_diff / B


class SemanticSegmentationVarianceLoss(nn.Module):
    def __init__(self, num_classes):
        super(SemanticSegmentationVarianceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, mean_features):
        """
        features: tensor of shape [B, 128, 32, 32], deep features
        labels: tensor of shape [B, 32, 32], class labels
        """
        # B, C, H, W = features.size()
        # mean_features = []
        # for cls in range(self.num_classes):
        #     # Create a mask for the current class
        #     mask = (labels == cls).unsqueeze(1).float()  # [B, 1, 32, 32]
        #     # Compute the number of pixels in the current class
        #     num_pixels = mask.sum(dim=(2, 3), keepdim=True)  # [B, 1, 1, 1]
        #     # Avoid division by zero
        #     num_pixels[num_pixels == 0] = 1
        #     # Compute the mean feature for the current class
        #     mean_feature = (features * mask).sum(dim=(2, 3), keepdim=True) / num_pixels  # [B, C, 1, 1]
        #     mean_features.append(mean_feature)
        # # Stack mean features into a tensor of shape [num_classes, B, C, 1, 1]
        # mean_features = torch.stack(mean_features, dim=0)  # [num_classes, B, C, 1, 1]
        # Compute the variance of the mean features across the batch dimension
        mean_features = mean_features.squeeze(-1).squeeze(-1)  # [num_classes, B, C]
        # print('mean_features',mean_features.shape)
        mean_features = mean_features.permute(1, 0, 2)  # [B, num_classes, C]
        mean_feature_variance = mean_features.var(dim=1)  # [B, C]
        # Compute the mean variance across all channels
        loss = mean_feature_variance.mean()

        return loss
class DeepLabV3PlusSimGlobalLinearKLDP(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimGlobalLinearKLDP, self).__init__()
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
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode == 'nopre':
            self.resnet = models.resnet50(weights=None)
        self.num_classes = num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), )

        self.classifier = nn.Conv2d(featDim, num_classes, 1)

        self.n_cluster = n_cluster
        self.prototypeN = 1
        # self.key = nn.Conv2d(featDim, featDim, kernel_size=3, padding=1)
        for i in range(num_classes):
            # setattr(self, f'query_{i}', nn.Conv2d(featDim * self.prototypeN, 128, kernel_size=1))
            setattr(self, f'a_{i}', nn.Parameter(torch.ones(1), requires_grad=True))
            setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1), requires_grad=True))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
        self.normP=nn.LayerNorm(128)
    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None,getPFlag=False):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        assp_features = self.aspp(x4)  # [10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        if getPFlag:
            return assp_features
        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        prototypes = []
        if ProtoInput == None:
            with torch.no_grad():
                pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
                #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
                mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
            # 计算每个类别的原型
            for i in range(self.num_classes):
                class_mask = (mask == i).float()
                if maskParam is not None:
                    class_mask = class_mask * maskParam  # Apply maskParam
                if class_mask.sum() > 0:  # 确保类别在批次中存在
                    # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                    prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                                class_mask.sum(dim=[2, 3]) + 1e-5)  # ([10, 256])
                    prototypes.append(prototype.unsqueeze(1))
                else:
                    # for pp in range(self.n_cluster):
                    prototypes.append(zero_prototype.unsqueeze(1))
            prototypes = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze(
                -1)  # prototypes torch.Size([10, 6, 128, 1, 1])
        # elif DomainLabel==1 and ProtoInput!=None:
        elif ProtoInput != None:
            prototypes = ProtoInput
        query_outputList = []
        # prototypesOut = []
        similarityList = []
        # key_output = self.key(assp_features)
        key_output=assp_features
        # print('key_output',key_output.shape,assp_features.shape)
        # key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
        jj = 0
        # print('prototypes',prototypes.shape)
        for c in range(self.num_classes):
            if ProtoInput == None:
                prototype = prototypes[:, jj]
                jj = jj + 1
                if prototype is not None:
                    query_output = getattr(self, f'a_{c}') * prototype + getattr(self, f'b_{c}')
                    deep_features_normalized = F.normalize(key_output, p=2, dim=1)  # [B, 128, H, W]
                    proto_features_normalized = F.normalize(query_output, p=2, dim=1)  # [B, 128, 1, 1]
                    query_output = query_output.view(query_output.size(0), -1,
                                                     query_output.size(1))  # Reshape to [10, 1, 128]
                    similarity = (deep_features_normalized * proto_features_normalized)  # [B, H, W]
                    similarity = similarity.sum(dim=1).unsqueeze(1)  # [B, H, W]
                    similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                                 assp_features.size(3))  # ([10, 1, 16, 16])
                    similarityList.append(similarity)
                    query_output = query_output.view(query_output.size(0),1 ,128
                                                     )  # Reshape to [10, 1, 128]
                    # print('query_outputC', query_output.shape)
                    query_outputList.append(query_output)
            else:
                for ii in range(self.n_cluster[c]):
                    prototype = prototypes[:, jj]
                    jj = jj + 1
                    if prototype is not None:
                        query_output = getattr(self, f'a_{c}') * prototype + getattr(self, f'b_{c}')
                        deep_features_normalized = F.normalize(key_output, p=2, dim=1)  # [B, 128, H, W]
                        proto_features_normalized = F.normalize(query_output, p=2, dim=1)  # [B, 128, 1, 1]
                        query_output = query_output.view(query_output.size(0), -1,
                                                         query_output.size(1))  # Reshape to [10, 1, 128]
                        similarity = (deep_features_normalized * proto_features_normalized)  # [B, H, W]
                        similarity = similarity.sum(dim=1).unsqueeze(1)  # [B, H, W]
                        similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                                     assp_features.size(3))  # ([10, 1, 16, 16])
                        similarityList.append(similarity)
                        query_output = query_output.view(query_output.size(0), 1,
                                                         128)  # Reshape to [10, 1, 128]
                        query_outputList.append(query_output)

        query_outputcat = torch.cat(query_outputList, dim=1).unsqueeze(-1).unsqueeze(-1)
        similarityCat = torch.cat(similarityList, dim=1)#([10, 30, 32, 32])
        similarityWeiht = F.softmax(similarityCat, dim=1)#torch.Size([10, 30, 32, 32])

        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        assp_weighted = assp_features * similarityWeihtMax.unsqueeze(1)
        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': xup}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,
                'Weight': [similarityWeiht, similarityWeihtMax]}

class DeepLabV3PlusSimGlobalLinearDP(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimGlobalLinearDP, self).__init__()
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
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode == 'nopre':
            self.resnet = models.resnet50(weights=None)
        self.num_classes = num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), )

        self.classifier = nn.Conv2d(featDim, num_classes, 1)

        self.n_cluster = n_cluster
        self.prototypeN = 1
        # self.key = nn.Conv2d(featDim, featDim, kernel_size=3, padding=1)
        for i in range(num_classes):
            # setattr(self, f'query_{i}', nn.Conv2d(featDim * self.prototypeN, 128, kernel_size=1))
            setattr(self, f'a_{i}', nn.Parameter(torch.ones(1), requires_grad=True))
            setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1), requires_grad=True))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
        self.normP=nn.LayerNorm(1024)
    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None,getPFlag=False):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)  # [10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        if getPFlag:
            return assp_features
        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        prototypes = []
        if ProtoInput == None:
            with torch.no_grad():
                pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
                #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
                mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
            # 计算每个类别的原型
            for i in range(self.num_classes):
                class_mask = (mask == i).float()
                if maskParam is not None:
                    class_mask = class_mask * maskParam  # Apply maskParam
                if class_mask.sum() > 0:  # 确保类别在批次中存在
                    # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                    prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                                class_mask.sum(dim=[2, 3]) + 1e-5)  # ([10, 256])
                    prototypes.append(prototype.unsqueeze(1))
                else:
                    # for pp in range(self.n_cluster):
                    prototypes.append(zero_prototype.unsqueeze(1))
            prototypes = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze(
                -1)  # prototypes torch.Size([10, 6, 128, 1, 1])
        # elif DomainLabel==1 and ProtoInput!=None:
        elif ProtoInput != None:
            prototypes = ProtoInput
        query_outputList = []
        # prototypesOut = []
        similarityList = []
        # key_output = self.key(assp_features)
        key_output=assp_features
        # print('key_output',key_output.shape,assp_features.shape)
        key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
        jj=0
        for c in range(self.num_classes):
            if ProtoInput == None:
                prototype = prototypes[:, jj]
                jj = jj + 1
                if prototype is not None:
                    query_output = getattr(self, f'a_{c}') * prototype + getattr(self, f'b_{c}')
                    query_output = query_output.view(query_output.size(0), -1,
                                                     query_output.size(1))  # Reshape to [10, 1, 128]
                    similarity = self.normP(torch.matmul(query_output, key_output))
                    # similarity = torch.matmul(query_output, key_output)
                    similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                                 assp_features.size(3))  # ([10, 1, 16, 16])
                    similarityList.append(similarity)
                    query_outputList.append(query_output)

            else:
                for ii in range(self.n_cluster[c]):
                    prototype = prototypes[:, jj]
                    jj=jj+1
                    if prototype is not None:
                        query_output = getattr(self, f'a_{c}') * prototype + getattr(self, f'b_{c}')
                        query_output = query_output.view(query_output.size(0), -1,
                                                         query_output.size(1))  # Reshape to [10, 1, 128]
                        similarity = self.normP(torch.matmul(query_output, key_output))
                        # similarity = torch.matmul(query_output, key_output)
                        similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                                     assp_features.size(3))  # ([10, 1, 16, 16])
                        similarityList.append(similarity)
                        query_outputList.append(query_output)

        query_outputcat = torch.cat(query_outputList, dim=1).unsqueeze(-1).unsqueeze(-1)
        # print('query_outputcat',query_outputcat.shape,prototypes.shape,GlobalProto_transOut.shape)
        similarityCat = torch.cat(similarityList, dim=1)#([10, 30, 32, 32])
        # print('similarityCat',similarityCat.shape)
        # print('similarityCat',similarityCat.shape)
        similarityWeiht = F.softmax(similarityCat, dim=1)#torch.Size([10, 30, 32, 32])
        # print('similarityCat',similarityCat.shape)

        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        assp_weighted = assp_features * similarityWeihtMax.unsqueeze(1)
        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': xup}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,
                'Weight': [similarityWeiht, similarityWeihtMax]}
class DeepLabV3PlusSimGlobalLinearF(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimGlobalLinearF, self).__init__()
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
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode == 'nopre':
            self.resnet = models.resnet50(weights=None)
        self.num_classes = num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), )

        self.classifier = nn.Conv2d(featDim, num_classes, 1)
        self.classifierF = nn.Sequential(nn.Conv2d(featDim, featDim, 3),
                                         nn.BatchNorm2d(featDim),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(featDim, featDim, 3),
                                         nn.BatchNorm2d(featDim),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(featDim, num_classes, 1),  )


        self.n_cluster = n_cluster
        self.prototypeN = 1
        # self.key = nn.Conv2d(featDim, featDim, kernel_size=3, padding=1)
        for i in range(num_classes):
            # setattr(self, f'query_{i}', nn.Conv2d(featDim * self.prototypeN, 128, kernel_size=1))
            setattr(self, f'a_{i}', nn.Parameter(torch.ones(1), requires_grad=True))
            setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1), requires_grad=True))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
        self.normP=nn.LayerNorm(1024)
    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None,getPFlag=False):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)  # [10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        if getPFlag:
            return assp_features
        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        prototypes = []
        if ProtoInput == None:
            with torch.no_grad():
                pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
                #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
                mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
            # 计算每个类别的原型
            for i in range(self.num_classes):
                class_mask = (mask == i).float()
                if maskParam is not None:
                    class_mask = class_mask * maskParam  # Apply maskParam
                if class_mask.sum() > 0:  # 确保类别在批次中存在
                    # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                    prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                                class_mask.sum(dim=[2, 3]) + 1e-5)  # ([10, 256])
                    prototypes.append(prototype.unsqueeze(1))
                else:
                    # for pp in range(self.n_cluster):
                    prototypes.append(zero_prototype.unsqueeze(1))
            prototypes = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze(
                -1)  # prototypes torch.Size([10, 6, 128, 1, 1])
        # elif DomainLabel==1 and ProtoInput!=None:
        elif ProtoInput != None:
            prototypes = ProtoInput
        query_outputList = []
        # prototypesOut = []
        similarityList = []
        # key_output = self.key(assp_features)
        key_output=assp_features
        # print('key_output',key_output.shape,assp_features.shape)
        key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
        for ii in range(prototypes.size(1)):
            prototype = prototypes[:, ii]
            if prototype is not None:
                #######key
                # key_output=getattr(self, f'a_{ii//(prototypes.size(1)//6)}') * key_output + getattr(self, f'b_{ii//((prototypes.size(1)//6))}')
                # query_output = prototype.view(prototype.size(0), -1, prototype.size(1))  # Reshape to [10, 1, 128]
                ########query
                # query_output = getattr(self, f'query_{ii // (prototypes.size(1) // 6)}') (prototype)
                query_output = getattr(self, f'a_{ii // (prototypes.size(1) // 6)}') * prototype + \
                               getattr(self,f'b_{ii // ((prototypes.size(1) // 6))}')
                query_output = query_output.view(query_output.size(0), -1, query_output.size(1))  # Reshape to [10, 1, 128]
                # prototypesOut.append(query_output.squeeze(1))
                # similarity = torch.abs(
                #     torch.bmm(query_output, key_output) / 128)  # Perform batch matrix multiplication#([10, 1, 1024])
                # Normalize along the channel dimension
                # query_output = F.normalize(query_output, p=2,dim=2)
                # key_output = F.normalize(key_output, p=2, dim=2)
                # Perform batch matrix multiplication#([10, 1, 1024])
                similarity = self.normP(torch.matmul(query_output, key_output))
                # similarity = torch.matmul(query_output, key_output)
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                             assp_features.size(3))  # ([10, 1, 16, 16])
                similarityList.append(similarity)
                query_outputList.append(query_output)

        query_outputcat = torch.cat(query_outputList, dim=1).unsqueeze(-1).unsqueeze(-1)
        # print('query_outputcat',query_outputcat.shape,prototypes.shape,GlobalProto_transOut.shape)
        similarityCat = torch.cat(similarityList, dim=1)#([10, 30, 32, 32])
        # print('similarityCat',similarityCat.shape)
        # print('similarityCat',similarityCat.shape)
        similarityWeiht = F.softmax(similarityCat, dim=1)#torch.Size([10, 30, 32, 32])
        # print('similarityCat',similarityCat.shape)

        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        assp_weighted = assp_features * (1+similarityWeihtMax.unsqueeze(1))

        outF=self.classifierF(assp_weighted.detach())
        outF = nn.functional.interpolate(outF, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': xup,'outF':outF}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,
                'Weight': [similarityWeiht, similarityWeihtMax]}
class DeepLabV3PlusSimGlobalLinearDist2(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimGlobalLinearDist2, self).__init__()
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
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode == 'nopre':
            self.resnet = models.resnet50(weights=None)
        self.num_classes = num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), )

        self.classifier = nn.Conv2d(featDim, num_classes, 1)

        self.n_cluster = n_cluster
        self.prototypeN = 1
        # self.key = nn.Conv2d(featDim, featDim, kernel_size=1)
        for i in range(num_classes):
            setattr(self, f'query_{i}', nn.Conv2d(featDim * self.prototypeN, 128, kernel_size=1))
            # setattr(self, f'a_{i}', nn.Parameter(torch.ones(1), requires_grad=True))
            # setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1), requires_grad=True))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
        self.normP=nn.LayerNorm(1024)
    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None,getPFlag=False):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)  # [10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        if getPFlag:
            return assp_features
        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        prototypes = []
        if ProtoInput == None:
            with torch.no_grad():
                pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
                #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
                mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
            # 计算每个类别的原型
            for i in range(self.num_classes):
                class_mask = (mask == i).float()
                if maskParam is not None:
                    class_mask = class_mask * maskParam  # Apply maskParam
                if class_mask.sum() > 0:  # 确保类别在批次中存在
                    # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                    prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                                class_mask.sum(dim=[2, 3]) + 1e-5)  # ([10, 256])
                    prototypes.append(prototype.unsqueeze(1))
                else:
                    # for pp in range(self.n_cluster):
                    prototypes.append(zero_prototype.unsqueeze(1))
            prototypes = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze(
                -1)  # prototypes torch.Size([10, 6, 128, 1, 1])
        # elif DomainLabel==1 and ProtoInput!=None:
        elif ProtoInput != None:
            prototypes = ProtoInput
        query_outputList = []
        # prototypesOut = []
        similarityList = []
        key_output = assp_features
        # key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
        for ii in range(prototypes.size(1)):
            prototype = prototypes[:, ii]
            if prototype is not None:
                #######key
                # key_output=getattr(self, f'a_{ii//(prototypes.size(1)//6)}') * key_output + getattr(self, f'b_{ii//((prototypes.size(1)//6))}')
                # query_output = prototype.view(prototype.size(0), -1, prototype.size(1))  # Reshape to [10, 1, 128]
                ########query
                # query_output = getattr(self, f'a_{ii // (prototypes.size(1) // 6)}') * prototype + \
                #                getattr(self,f'b_{ii // ((prototypes.size(1) // 6))}')
                query_output=getattr(self, f'query_{ii // (prototypes.size(1) // 6)}') (prototype)
                query_output = query_output.view(query_output.size(0), -1, query_output.size(1))  # Reshape to [10, 1, 128]
                # prototypesOut.append(query_output.squeeze(1))
                # similarity = torch.abs(
                #     torch.bmm(query_output, key_output) / 128)  # Perform batch matrix multiplication#([10, 1, 1024])
                # Normalize along the channel dimension
                # query_output = F.normalize(query_output, p=2,dim=2)
                # key_output = F.normalize(key_output, p=2, dim=2)
                # Perform batch matrix multiplication#([10, 1, 1024])
                # print('key_output',key_output.shape,query_output.shape)
                # similarity = torch.cdist(key_output, query_output, p=2)  # [batch_size*height*width, num_classes]

                # distances = self.normP(torch.matmul(query_output, key_output))
                # similarity = torch.matmul(query_output, key_output)
                # similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                #                              assp_features.size(3))  # ([10, 1, 16, 16])
                # similarityList.append(similarity)
                query_outputList.append(query_output)

        query_outputcat = torch.cat(query_outputList, dim=1)
        key_output = key_output.permute(0, 2, 3, 1).contiguous().view(key_output.size(0), -1, 128)  # [batch_size, height*width, feature_dim]
        prototypes_expanded = query_outputcat.unsqueeze(1)  # [batch_size, 1, num_classes, feature_dim]
        feature_map_expanded = key_output.unsqueeze(2)  # [batch_size, height*width, 1, feature_dim]
        # print('prototypes_expanded',prototypes_expanded.shape,feature_map_expanded.shape)
        # distances = (-torch.norm(feature_map_expanded - prototypes_expanded, p='fro',
        #                                  dim=3) ) # [batch_size, height*width, num_classes]
        # print('distances',distances.shape)
        # frobenius_distances = frobenius_distances.view(batch_size, height, width, num_classes).permute(0, 3, 1,
        #                                                                                                2)  # [batch_size, num_classes, height, width]
        # distances = -F.cosine_similarity(feature_map_expanded, prototypes_expanded,
        #                                         dim=3)  # [batch_size, height*width, num_classes]

        distances = -torch.cdist(feature_map_expanded, prototypes_expanded, p=2)  # [batch_size, height*width, num_classes]
        # print('distances',distances.shape)
        similarityCat = distances.view(distances.size(0),distances.size(-1),32,32)
        distances = distances.view(distances.size(0), -1, 6, query_outputcat.size(1)//6)  # [batch_size, height*width, num_classes, num_prototypes_per_class]
        min_distances, _ = torch.max(distances, dim=3)  # [batch_size, height*width, num_classes]

        # print('min_distances',min_distances.shape)
        distancesReshape = min_distances.view(min_distances.size(0), 32, 32, 6).permute(0, 3, 1, 2)  # [batch_size, num_classes, height, width]
        # print('similarityCat',similarityCat.shape)
        # print('query_outputcat',query_outputcat.shape,prototypes.shape,GlobalProto_transOut.shape)
        # similarityCat = torch.cat(similarityList, dim=1)#([10, 30, 32, 32])
        # print('similarityCat',similarityCat.shape)
        # print('similarityCat',similarityCat.shape)
        similarityWeiht = F.softmax(distancesReshape, dim=1)#torch.Size([10, 30, 32, 32])
        # print('similarityCat',similarityCat.shape)

        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        # similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        assp_weighted = assp_features * similarityWeiht.max(1)[0].unsqueeze(1)
        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)
        query_outputcat=query_outputcat.unsqueeze(-1).unsqueeze(-1)
        return {'out': x, 'outUp': xup}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,
                'Weight':  distancesReshape}

class DeepLabV3PlusSimGlobalLinearNew(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimGlobalLinearNew, self).__init__()
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
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode == 'nopre':
            self.resnet = models.resnet50(weights=None)
        self.num_classes = num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), )

        self.classifier = nn.Conv2d(featDim, num_classes, 1)

        self.n_cluster = n_cluster
        self.prototypeN = 1
        # self.key = nn.Conv2d(featDim, featDim, kernel_size=1)
        for i in range(num_classes):
        #     # setattr(self, f'query_{i}', nn.Conv2d(featDim * self.prototypeN, 128, kernel_size=1))
            setattr(self, f'a_{i}', nn.Parameter(torch.ones(1), requires_grad=True))
            setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1), requires_grad=True))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
        self.normP=nn.LayerNorm(1024)
    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None,getPFlag=False):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)  # [10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        if getPFlag:
            return assp_features
        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        prototypes = []
        if ProtoInput == None:
            with torch.no_grad():
                pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
                #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
                mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
            # 计算每个类别的原型
            for i in range(self.num_classes):
                class_mask = (mask == i).float()
                if maskParam is not None:
                    class_mask = class_mask * maskParam  # Apply maskParam
                if class_mask.sum() > 0:  # 确保类别在批次中存在
                    # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                    prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                                class_mask.sum(dim=[2, 3]) + 1e-5)  # ([10, 256])
                    prototypes.append(prototype.unsqueeze(1))
                else:
                    # for pp in range(self.n_cluster):
                    prototypes.append(zero_prototype.unsqueeze(1))
            prototypes = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze(
                -1)  # prototypes torch.Size([10, 6, 128, 1, 1])
        # elif DomainLabel==1 and ProtoInput!=None:
        elif ProtoInput != None:
            prototypes = ProtoInput
        query_outputList = []
        # prototypesOut = []
        similarityList = []
        key_output = assp_features
        key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
        for ii in range(prototypes.size(1)):
            prototype = prototypes[:, ii]
            if prototype is not None:
                #######key
                # key_output=getattr(self, f'a_{ii//(prototypes.size(1)//6)}') * key_output + getattr(self, f'b_{ii//((prototypes.size(1)//6))}')
                # query_output = prototype.view(prototype.size(0), -1, prototype.size(1))  # Reshape to [10, 1, 128]
                ########query
                print(ii,ii // (prototypes.size(1) // 6))
                # query_output = getattr(self, f'a_{ii // (prototypes.size(1) // 6)}') * prototype + \
                #                getattr(self,f'b_{ii // ((prototypes.size(1) // 6))}')
                query_output = prototype.view(prototype.size(0), -1, prototype.size(1))  # Reshape to [10, 1, 128]
                # prototypesOut.append(query_output.squeeze(1))
                # similarity = torch.abs(
                #     torch.bmm(query_output, key_output) / 128)  # Perform batch matrix multiplication#([10, 1, 1024])
                # Normalize along the channel dimension
                # query_output = F.normalize(query_output, p=2,dim=2)
                # key_output = F.normalize(key_output, p=2, dim=2)
                # Perform batch matrix multiplication#([10, 1, 1024])
                similarity = self.normP(torch.matmul(query_output, key_output.detach()))
                # similarity = torch.matmul(query_output, key_output)
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                             assp_features.size(3))  # ([10, 1, 16, 16])
                similarityList.append(similarity)
                query_outputList.append(query_output)

        query_outputcat = torch.cat(query_outputList, dim=1).unsqueeze(-1).unsqueeze(-1)
        # print('query_outputcat',query_outputcat.shape,prototypes.shape,GlobalProto_transOut.shape)
        similarityCat = torch.cat(similarityList, dim=1)#([10, 30, 32, 32])
        # print('similarityCat',similarityCat.shape)
        # print('similarityCat',similarityCat.shape)
        similarityWeiht = F.softmax(similarityCat, dim=1)#torch.Size([10, 30, 32, 32])
        # print('similarityCat',similarityCat.shape)

        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        assp_weighted = assp_features * similarityWeihtMax.unsqueeze(1)
        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': xup}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,
                'Weight': [similarityWeiht, similarityWeihtMax]}


class DeepLabV3PlusSimGlobalLinearClassifer(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimGlobalLinearClassifer, self).__init__()
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
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode == 'nopre':
            self.resnet = models.resnet50(weights=None)
        self.num_classes = num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), )

        self.classifier = nn.Conv2d(featDim, num_classes, 1)
        self.classifierSim= nn.Sequential(
            nn.Conv2d(n_cluster*num_classes, num_classes, 1),
            nn.ReLU(),
            nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_classes, num_classes, 1)
        )


        self.n_cluster = n_cluster
        self.prototypeN = 1
        # self.key = nn.Conv2d(featDim, featDim, kernel_size=1)
        for i in range(num_classes):
        #     # setattr(self, f'query_{i}', nn.Conv2d(featDim * self.prototypeN, 128, kernel_size=1))
            setattr(self, f'a_{i}', nn.Parameter(torch.ones(1), requires_grad=True))
            setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1), requires_grad=True))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
        self.normP = nn.LayerNorm(1024)

    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None, getPFlag=False):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)  # [10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        if getPFlag:
            return assp_features
        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        prototypes = []
        if ProtoInput == None:
            with torch.no_grad():
                pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
                #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
                mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
            # 计算每个类别的原型
            for i in range(self.num_classes):
                class_mask = (mask == i).float()
                if maskParam is not None:
                    class_mask = class_mask * maskParam  # Apply maskParam
                if class_mask.sum() > 0:  # 确保类别在批次中存在
                    # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                    prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                            class_mask.sum(dim=[2, 3]) + 1e-5)  # ([10, 256])
                    prototypes.append(prototype.unsqueeze(1))
                else:
                    # for pp in range(self.n_cluster):
                    prototypes.append(zero_prototype.unsqueeze(1))
            prototypes = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze(
                -1)  # prototypes torch.Size([10, 6, 128, 1, 1])
        # elif DomainLabel==1 and ProtoInput!=None:
        elif ProtoInput != None:
            prototypes = ProtoInput
        query_outputList = []
        # prototypesOut = []
        similarityList = []
        key_output = assp_features
        key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
        for ii in range(prototypes.size(1)):
            prototype = prototypes[:, ii]
            if prototype is not None:
                #######key
                # key_output=getattr(self, f'a_{ii//(prototypes.size(1)//6)}') * key_output + getattr(self, f'b_{ii//((prototypes.size(1)//6))}')
                # query_output = prototype.view(prototype.size(0), -1, prototype.size(1))  # Reshape to [10, 1, 128]
                ########query
                query_output = getattr(self, f'a_{ii // (prototypes.size(1) // 6)}') * prototype + \
                               getattr(self,f'b_{ii // ((prototypes.size(1) // 6))}')
                query_output = query_output.view(query_output.size(0), -1, query_output.size(1))  # Reshape to [10, 1, 128]
                # prototypesOut.append(query_output.squeeze(1))
                # similarity = torch.abs(
                #     torch.bmm(query_output, key_output) / 128)  # Perform batch matrix multiplication#([10, 1, 1024])
                # Normalize along the channel dimension
                # query_output = F.normalize(query_output, p=2,dim=2)
                # key_output = F.normalize(key_output, p=2, dim=2)
                # Perform batch matrix multiplication#([10, 1, 1024])
                similarity = self.normP(torch.matmul(query_output, key_output))
                # similarity = torch.matmul(query_output, key_output)
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                             assp_features.size(3))  # ([10, 1, 16, 16])
                similarityList.append(similarity)
                query_outputList.append(query_output)

        query_outputcat = torch.cat(query_outputList, dim=1).unsqueeze(-1).unsqueeze(-1)
        # print('query_outputcat',query_outputcat.shape,prototypes.shape,GlobalProto_transOut.shape)
        similarityCat = torch.cat(similarityList, dim=1)  # ([10, 30, 32, 32])
        similarityCatC=self.classifierSim(similarityCat)
        # print('similarityCat',similarityCat.shape)
        # print('similarityCat',similarityCat.shape)
        similarityWeiht = F.softmax(similarityCatC, dim=1)  # torch.Size([10, 30, 32, 32])
        # print('similarityCat',similarityCat.shape)

        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        assp_weighted = assp_features * similarityWeihtMax.unsqueeze(1)
        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': xup}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,'cOut':similarityCatC,
                'Weight': [similarityWeiht, similarityWeihtMax]}
class DeepLabV3PlusSimGlobalFC(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimGlobalFC, self).__init__()
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
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode == 'nopre':
            self.resnet = models.resnet50(weights=None)
        self.num_classes = num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), )

        self.classifier = nn.Conv2d(featDim, num_classes, 1)
        self.n_cluster = n_cluster
        self.prototypeN = 1
        # self.key = nn.Conv2d(featDim, featDim, kernel_size=1)
        for i in range(num_classes):
            # setattr(self, f'query_{i}', nn.Conv2d(featDim * self.prototypeN, 128, kernel_size=1))
            # setattr(self, f'a_{i}', nn.Parameter(torch.ones(1), requires_grad=True))
            # setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1), requires_grad=True))
            setattr(self, f'query_{i}', nn.Conv2d(128, 128, kernel_size=1))
        self.normP=nn.LayerNorm(1024)
    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None,getPFlag=False):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)  # [10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        if getPFlag:
            return assp_features
        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        prototypes = []
        if ProtoInput == None:
            with torch.no_grad():
                pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
                #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
                mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
            # 计算每个类别的原型
            for i in range(self.num_classes):
                class_mask = (mask == i).float()
                if maskParam is not None:
                    class_mask = class_mask * maskParam  # Apply maskParam
                if class_mask.sum() > 0:  # 确保类别在批次中存在
                    # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                    prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                                class_mask.sum(dim=[2, 3]) + 1e-5)  # ([10, 256])
                    prototypes.append(prototype.unsqueeze(1))
                else:
                    # for pp in range(self.n_cluster):
                    prototypes.append(zero_prototype.unsqueeze(1))
            prototypes = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze(
                -1)  # prototypes torch.Size([10, 6, 128, 1, 1])
        # elif DomainLabel==1 and ProtoInput!=None:
        elif ProtoInput != None:
            prototypes = ProtoInput
        query_outputList = []
        # prototypesOut = []
        similarityList = []
        key_output = assp_features
        key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
        for ii in range(prototypes.size(1)):
            prototype = prototypes[:, ii]
            if prototype is not None:
                #######key
                query_output=getattr(self, f'query_{ii//(prototypes.size(1)//6)}')(prototype)
                query_output = query_output.view(query_output.size(0), -1, query_output.size(1))  # Reshape to [10, 1, 128]
                ########query
                # query_output = getattr(self, f'a_{ii // (prototypes.size(1) // 6)}') * prototype + \
                #                getattr(self,f'b_{ii // ((prototypes.size(1) // 6))}')
                # query_output = prototype.view(prototype.size(0), -1, prototype.size(1))  # Reshape to [10, 1, 128]
                # prototypesOut.append(query_output.squeeze(1))
                # similarity = torch.abs(
                #     torch.bmm(query_output, key_output) / 128)  # Perform batch matrix multiplication#([10, 1, 1024])
                # Normalize along the channel dimension
                # query_output = F.normalize(query_output, p=2,dim=2)
                # key_output = F.normalize(key_output, p=2, dim=2)
                # Perform batch matrix multiplication#([10, 1, 1024])
                similarity = self.normP(torch.matmul(query_output, key_output))
                # similarity = torch.matmul(query_output, key_output)
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                             assp_features.size(3))  # ([10, 1, 16, 16])
                similarityList.append(similarity)
                query_outputList.append(query_output)
        #generate global prototype
        # GlobalProto_transList=[]
        # if ProtoInput != None:
        #     with torch.no_grad():
        #         GlobalProto = ProtoInput.clone()
        #         for ii in range(GlobalProto.size(1)):
        #             prototype = GlobalProto[:, ii]
        #
        #             GlobalProto_trans = getattr(self, f'a_{ii // (prototype.size(1) // 6)}') * prototype + getattr(self,
        #                                                                                                        f'b_{ii // ((prototype.size(1) // 6))}')
        #             GlobalProto_trans = GlobalProto_trans.view(GlobalProto_trans.size(0), -1,
        #                                           GlobalProto_trans.size(1))  # Reshape to [10, 1, 128]
        #             GlobalProto_transList.append(GlobalProto_trans)
        #
        # GlobalProto_transOut = torch.cat(GlobalProto_transList, dim=1).unsqueeze(-1).unsqueeze(-1)


        query_outputcat = torch.cat(query_outputList, dim=1).unsqueeze(-1).unsqueeze(-1)
        # print('query_outputcat',query_outputcat.shape,prototypes.shape,GlobalProto_transOut.shape)
        similarityCat = torch.cat(similarityList, dim=1)#([10, 30, 32, 32])
        # print('similarityCat',similarityCat.shape)
        # print('similarityCat',similarityCat.shape)
        similarityWeiht = F.softmax(similarityCat, dim=1)#torch.Size([10, 30, 32, 32])
        # print('similarityCat',similarityCat.shape)

        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        assp_weighted = assp_features * similarityWeihtMax.unsqueeze(1)
        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': xup}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,
                'Weight': [similarityWeiht, similarityWeihtMax]}
class DeepLabV3PlusSimGlobalFCDist(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimGlobalFCDist, self).__init__()
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
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode == 'nopre':
            self.resnet = models.resnet50(weights=None)
        self.num_classes = num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), )

        self.classifier = nn.Conv2d(featDim, num_classes, 1)
        self.n_cluster = n_cluster
        self.prototypeN = 1
        # self.key = nn.Conv2d(featDim, 128, kernel_size=1)
        for i in range(num_classes):
            # setattr(self, f'query_{i}', nn.Conv2d(featDim * self.prototypeN, 128, kernel_size=1))
            # setattr(self, f'a_{i}', nn.Parameter(torch.ones(1), requires_grad=True))
            # setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1), requires_grad=True))
            setattr(self, f'query_{i}', nn.Conv2d(128, 128, kernel_size=1))
        self.normP = nn.LayerNorm(1024)

    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None, getPFlag=False):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)  # [10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        if getPFlag:
            return assp_features
        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        prototypes = []
        if ProtoInput == None:
            with torch.no_grad():
                pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
                #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
                mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
            # 计算每个类别的原型
            for i in range(self.num_classes):
                class_mask = (mask == i).float()
                if maskParam is not None:
                    class_mask = class_mask * maskParam  # Apply maskParam
                if class_mask.sum() > 0:  # 确保类别在批次中存在
                    # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                    prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                            class_mask.sum(dim=[2, 3]) + 1e-5)  # ([10, 256])
                    prototypes.append(prototype.unsqueeze(1))
                else:
                    # for pp in range(self.n_cluster):
                    prototypes.append(zero_prototype.unsqueeze(1))
            prototypes = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze(
                -1)  # prototypes torch.Size([10, 6, 128, 1, 1])
        # elif DomainLabel==1 and ProtoInput!=None:
        elif ProtoInput != None:
            prototypes = ProtoInput

        query_outputList = []
        # prototypesOut = []
        # similarityList = []
        # key_output = assp_features
        # key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
        for ii in range(prototypes.size(1)):
            prototype = prototypes[:, ii]
            # print('prototype',prototype.shape,prototypes.shape)

            if prototype is not None:
                #######key
                # key_output=getattr(self, f'a_{ii//(prototypes.size(1)//6)}') * key_output + getattr(self, f'b_{ii//((prototypes.size(1)//6))}')
                # query_output = prototype.view(prototype.size(0), -1, prototype.size(1))  # Reshape to [10, 1, 128]
                ########query
                # query_output = getattr(self, f'a_{ii // (prototypes.size(1) // 6)}') * prototype + getattr(self,
                #                                                                                            f'b_{ii // ((prototypes.size(1) // 6))}')
                # query_output = getattr(self, f'a_{ii // (prototypes.size(1) // 6)}') * prototype + \
                #                getattr(self, f'b_{ii // ((prototypes.size(1) // 6))}')
                query_output = getattr(self, f'query_{ii // (prototypes.size(1) // 6)}')(prototype)
                query_output = query_output.view(query_output.size(0), -1, query_output.size(1))  # Reshape to [10, 1, 128]
                query_outputList.append(query_output)

        query_outputcat = torch.cat(query_outputList, dim=1)#[10, 6, 128])
        query_outputcat=query_outputcat.unsqueeze(-1).unsqueeze(-1)
        prototypes_expanded = query_outputcat.expand(-1, -1, -1, assp_features.size(2), assp_features.size(3))
        similarityCat = 1/torch.norm(assp_features.unsqueeze(1) - prototypes_expanded, p=2, dim=2)  # [B, 6, H, W]
        # print('similarityCat',similarityCat.shape,prototypes_expanded.shape)
        similarityWeiht = F.softmax(similarityCat, dim=1)  # torch.Size([10, 30, 32, 32])
        # print('similarityCat',similarityCat.shape)

        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        assp_weighted = assp_features * similarityWeihtMax.unsqueeze(1)
        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': xup}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,
                'Weight': [similarityWeiht, similarityWeihtMax]}
class DeepLabV3PlusSimGlobalLinearDist(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimGlobalLinearDist, self).__init__()
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
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode == 'nopre':
            self.resnet = models.resnet50(weights=None)
        self.num_classes = num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), )

        self.classifier = nn.Conv2d(featDim, num_classes, 1)
        self.n_cluster = n_cluster
        self.prototypeN = 1
        # self.key = nn.Conv2d(featDim, 128, kernel_size=1)
        for i in range(num_classes):
            # setattr(self, f'query_{i}', nn.Conv2d(featDim * self.prototypeN, 128, kernel_size=1))
            setattr(self, f'a_{i}', nn.Parameter(torch.ones(1), requires_grad=True))
            setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1), requires_grad=True))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
        self.normP = nn.LayerNorm(1024)

        self.classifierSim = nn.Sequential(
            nn.Conv2d(n_cluster * num_classes, num_classes, 1),
            nn.ReLU(),
            nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_classes, num_classes, 1)
        )
    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None, getPFlag=False):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)  # [10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        if getPFlag:
            return assp_features
        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        prototypes = []
        if ProtoInput == None:
            with torch.no_grad():
                pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
                #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
                mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
            # 计算每个类别的原型
            for i in range(self.num_classes):
                class_mask = (mask == i).float()
                if maskParam is not None:
                    class_mask = class_mask * maskParam  # Apply maskParam
                if class_mask.sum() > 0:  # 确保类别在批次中存在
                    # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                    prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                            class_mask.sum(dim=[2, 3]) + 1e-5)  # ([10, 256])
                    prototypes.append(prototype.unsqueeze(1))
                else:
                    # for pp in range(self.n_cluster):
                    prototypes.append(zero_prototype.unsqueeze(1))
            prototypes = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze(
                -1)  # prototypes torch.Size([10, 6, 128, 1, 1])
        # elif DomainLabel==1 and ProtoInput!=None:
        elif ProtoInput != None:
            prototypes = ProtoInput

        query_outputList = []
        # prototypesOut = []
        # similarityList = []
        # key_output = assp_features
        # key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
        for ii in range(prototypes.size(1)):
            prototype = prototypes[:, ii]
            # print('prototype',prototype.shape,prototypes.shape)

            if prototype is not None:
                #######key
                # key_output=getattr(self, f'a_{ii//(prototypes.size(1)//6)}') * key_output + getattr(self, f'b_{ii//((prototypes.size(1)//6))}')
                # query_output = prototype.view(prototype.size(0), -1, prototype.size(1))  # Reshape to [10, 1, 128]
                ########query
                query_output = getattr(self, f'a_{ii // (prototypes.size(1) // 6)}') * prototype + getattr(self,
                                                                                                           f'b_{ii // ((prototypes.size(1) // 6))}')
                query_output = query_output.view(query_output.size(0), -1, query_output.size(1))  # Reshape to [10, 1, 128]
                query_outputList.append(query_output)

        query_outputcat = torch.cat(query_outputList, dim=1)#[10, 6, 128])
        query_outputcat=query_outputcat.unsqueeze(-1).unsqueeze(-1)
        prototypes_expanded = query_outputcat.expand(-1, -1, -1, assp_features.size(2), assp_features.size(3))
        similarityCat = 1/torch.norm(assp_features.unsqueeze(1) - prototypes_expanded, p=2, dim=2)  # [B, 6, H, W]
        # print('similarityCat',similarityCat.shape)
        similarityCatC=self.classifierSim(similarityCat)

        # print('similarityCat',similarityCat.shape,prototypes_expanded.shape)
        similarityWeiht = F.softmax(similarityCat, dim=1)  # torch.Size([10, 30, 32, 32])
        # print('similarityCat',similarityCat.shape)

        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        assp_weighted = assp_features * similarityWeihtMax.unsqueeze(1)
        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': xup}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,'cOut':similarityCatC,
                'Weight': [similarityWeiht, similarityWeihtMax]}
class DeepLabV3PlusSimCatLinearWeight(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimCatLinearWeight, self).__init__()
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
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode == 'nopre':
            self.resnet = models.resnet50(weights=None)
        self.num_classes = num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), )

        self.classifier = nn.Conv2d(featDim, num_classes, 1)
        self.n_cluster = n_cluster
        self.prototypeN = 1
        # self.key = nn.Conv2d(featDim, 128, kernel_size=1)
        for i in range(num_classes):
            # setattr(self, f'query_{i}', nn.Conv2d(featDim * self.prototypeN, 128, kernel_size=1))
            setattr(self, f'a_{i}', nn.Parameter(torch.ones(1), requires_grad=True))
            setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1), requires_grad=True))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
        self.normP=nn.LayerNorm(1024)
    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None,getPFlag=False):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)  # [10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        if getPFlag:
            return assp_features
        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        prototypes = []
        # if DomainLabel == 0:
        # if ProtoInput == None:
        with torch.no_grad():
            pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
            #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
            mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
        # 计算每个类别的原型
        for i in range(self.num_classes):
            class_mask = (mask == i).float()
            if maskParam is not None:
                class_mask = class_mask * maskParam  # Apply maskParam
            if class_mask.sum() > 0:  # 确保类别在批次中存在
                prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                            class_mask.sum(dim=[2, 3]) + 1e-5).detach()  # ([10, 256])
                prototypes.append(prototype.unsqueeze(1))
            else:
                prototypes.append(zero_prototype.unsqueeze(1))
            prototypes.append(ProtoInput[:,i*self.n_cluster:(i+1)*self.n_cluster,:,0,0])
        prototypes = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze(-1)  # prototypes t([10, 6, 128, 1, 1])
        query_outputList = []
        # prototypesOut = []
        similarityList = []
        key_output = assp_features
        key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
        for ii in range(prototypes.size(1)):
            prototype = prototypes[:, ii]
            if prototype is not None:
                key_output=getattr(self, f'a_{ii//(prototypes.size(1)//6)}') * key_output + getattr(self, f'b_{ii//((prototypes.size(1)//6))}')
                query_output = prototype.view(prototype.size(0), -1, prototype.size(1))  # Reshape to [10, 1, 128]

                # query_output = getattr(self, f'a_{ii // (prototypes.size(1) // 6)}') * prototype + getattr(self,
                #                                                                                            f'b_{ii // ((prototypes.size(1) // 6))}')
                # query_output = prototype.view(query_output.size(0), -1, query_output.size(1))  # Reshape to [10, 1, 128]
                # prototypesOut.append(query_output.squeeze(1))
                # similarity = torch.abs(
                #     torch.bmm(query_output, key_output) / 128)  # Perform batch matrix multiplication#([10, 1, 1024])
                # Normalize along the channel dimension
                # query_output = F.normalize(query_output, p=2,dim=2)
                # key_output = F.normalize(key_output, p=2, dim=2)
                # Perform batch matrix multiplication#([10, 1, 1024])
                similarity = self.normP(torch.matmul(query_output, key_output))
                # similarity = torch.matmul(query_output, key_output)
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                             assp_features.size(3))  # ([10, 1, 16, 16])
                similarityList.append(similarity)
                query_outputList.append(query_output)

        query_outputcat = torch.cat(query_outputList, dim=1).unsqueeze(-1).unsqueeze(-1)
        similarityCat = torch.cat(similarityList, dim=1)
        similarityWeiht = F.softmax(similarityCat, dim=1)
        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        assp_weighted = assp_features * similarityWeihtMax.unsqueeze(1)
        x = self.classifier(assp_weighted)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': xup}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,
                'Weight': [similarityWeiht, similarityWeihtMax]}
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
        self.key = nn.Conv2d(featDim, 128, kernel_size=1)
        self.bn_key = nn.BatchNorm2d(128)
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

        prototypes = []
        if DomainLabel == 0:
            with torch.no_grad():
                pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
                #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
                mask = torch.argmax(pseudo_out, dim=1).unsqueeze(1)
            # 计算每个类别的原型
            for i in range(self.num_classes):
                class_mask = (mask == i).float()
                if maskParam is not None:
                    # print('class_mask * maskParam',class_mask.shape , maskParam.shape)
                    class_mask = class_mask * maskParam  # Apply maskParam
                if class_mask.sum() > 0:  # 确保类别在批次中存在
                    # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                    prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]) + 1e-5)#([10, 256])
                    # for pp in range(self.n_cluster):
                        # print('prototype',prototype.shape)
                    prototypes.append(prototype.unsqueeze(1))
                else:
                    # for pp in range(self.n_cluster):
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
                query_output = getattr(self, f'query_{i//(len(prototypes)//6)}')(prototype[:,0,:].unsqueeze(-1).unsqueeze(-1))
                query_output = getattr(self, f'bn_query_{i//(len(prototypes)//6)}')(query_output)

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

        assp_weighted=assp_features*similarityWeihtMax.unsqueeze(1)

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


class DeepLabV3PlusMultiPrototypeSingleKeyBias(nn.Module):
    def __init__(self, num_classes=21,args=None,n_cluster=1):
        super(DeepLabV3PlusMultiPrototypeSingleKeyBias, self).__init__()
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
        # self.key=nn.Conv2d(featDim, 128, kernel_size=1)
        # self.value=nn.Conv2d(featDim, 128, kernel_size=1)
        # self.value = nn.Conv2d(featDim, 128, kernel_size=1, groups=128)
        self.value = nn.Linear(featDim, 128)
        # 初始化为单位矩阵和零偏置
        nn.init.eye_(self.value.weight)
        nn.init.zeros_(self.value.bias)
        # 初始化卷积权重和偏置
        # nn.init.constant_(self.value.weight, 1)  # 初始化权重为1
        # nn.init.constant_(self.value.bias, 0)  # 初始化偏置为0

        for i in range(num_classes):
            # setattr(self, f'query_{i}', nn.Conv2d(featDim * self.prototypeN, 128, kernel_size=1))
            setattr(self, f'a_{i}', nn.Parameter(torch.ones(1),requires_grad=True))
            setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1),requires_grad=True))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
            # self.a = nn.Parameter(torch.ones(1))  # 初始化为 1
            # self.b = nn.Parameter(torch.zeros(1))  # 初始化为 0
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
        # with torch.no_grad():
        #     self.value.weight.clamp_(0.9, 1.1)
        #     self.value.bias.clamp_(-0.1, 0.1)

        prototypes = []
        if DomainLabel == 0:
            with torch.no_grad():
                pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
                #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
                mask = torch.argmax(pseudo_out, dim=1).unsqueeze(1)
            # 计算每个类别的原型
            for i in range(self.num_classes):
                class_mask = (mask == i).float()
                if maskParam is not None:
                    # print('class_mask * maskParam',class_mask.shape , maskParam.shape)
                    class_mask = class_mask * maskParam  # Apply maskParam
                if class_mask.sum() > 0:  # 确保类别在批次中存在
                    # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                    prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]) + 1e-5)#([10, 256])
                    # for pp in range(self.n_cluster):
                        # print('prototype',prototype.shape)
                    prototypes.append(prototype.unsqueeze(1))
                else:
                    # for pp in range(self.n_cluster):
                    prototypes.append(zero_prototype.unsqueeze(1))
        elif DomainLabel==1 and ProtoInput!=None:
            prototypes=ProtoInput
        # print('lem',len(prototypes))
        # 将原型输入到对应的query层
        # total_weight=0
        # key_output = self.key(assp_features)
        key_output = assp_features.view(assp_features.size(0), assp_features.size(1), -1)  # Reshape to [10, 128, 16*16]
        # print('key_output',key_output.shape)
        # query_outputList=[]
        similarityList=[]
        query_outputList=[]
        aList=[]
        bList=[]
        prototypesOri=[]
        for i, prototype in enumerate(prototypes):
            if prototype is not None:#([10, 1, 128])
                inproto=prototype[:,0,:].unsqueeze(-1)
                # query_output = getattr(self, f'query_{i//(len(prototypes)//6)}')(prototype[:,0,:].unsqueeze(-1).unsqueeze(-1))
                # print('prototypes',prototype.shape)
                prototypesOri.append(prototype)
                query_output= getattr(self, f'a_{i//(len(prototypes)//6)}') * inproto + getattr(self, f'b_{i//(len(prototypes)//6)}')
                query_output = query_output.view(query_output.size(0), 1, query_output.size(1))  # Reshape to [10, 1, 128]
                # query_outputList.append(query_output)
                query_outputList.append(query_output)
                similarity = torch.abs(torch.bmm(query_output, key_output)/128)  # Perform batch matrix multiplication#torch.Size([10, 1, 256])
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                               assp_features.size(3))#([10, 1, 16, 16])
                similarityList.append(similarity)
                aList.append(getattr(self, f'a_{i // (len(prototypes) // 6)}').unsqueeze(0))
                bList.append(getattr(self, f'b_{i // (len(prototypes) // 6)}').unsqueeze(0))
        # prototypesOri=torch.cat(prototypesOri,dim=1)
        aOut=torch.cat(aList,dim=0)
        bOut=torch.cat(bList,dim=0)
        query_outputcat = torch.cat(query_outputList, dim=1)#[10,6,128]
        B, C, W, H = assp_features.shape
        query_outputcat=query_outputcat.unsqueeze(-1).unsqueeze(-1)
        prototypes_expanded = query_outputcat.expand(-1, -1, -1, W, H)  # [B, 6, 128, H, W]
        deep_features_expanded = assp_features.unsqueeze(1)  # [B, 1, 128, H, W]
        # deep_features_expanded = deep_features_expanded.expand(-1, 6, -1, -1, -1)  # [B, 6, 128, H, W]
        distances = torch.norm(prototypes_expanded - deep_features_expanded, p=2, dim=2)  # [B, 6, H, W]
        DistP = F.softmax(-distances.detach(), dim=1)  # [B, 6, H, W]
        DistPMax,_=torch.max(DistP,dim=1)

        # B, C, W, H = assp_features.shape
        # _, num_prototypes, proto_dim = query_outputcat.shape
        # # Reshape features to [B, W*H, C]
        # features_reshaped = assp_features.permute(0, 2, 3, 1).reshape(B, -1, C)
        # # Calculate distances between each feature and each prototype
        # # distances = torch.cdist(features_reshaped, query_outputcat, p=2)
        # distances = torch.norm(query_outputcat - features_reshaped, p=2, dim=2)
        #
        # DistP = F.softmax(-distances.detach(), dim=1)  # [B, 6, H, W]
        # DistPMax, _ = torch.max(DistP, dim=1)
        # Find the nearest prototype for each feature point
        # nearest_prototypes = torch.argmin(distances, dim=-1)
        # Reshape to [B, W, H] to represent class predictions
        # classification_result = nearest_prototypes.reshape(B, W, H)


        similarityCat = torch.cat(similarityList, dim=1)
        similarityWeiht = F.softmax(similarityCat / similarityCat.size(1) * self.n_cluster, dim=1)
        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)

        # similarityWeihtMaxMax,_=similarityWeihtMax.max(dim=(1, 2), keepdim=True)
        # print('similarityWeihtMax',similarityWeihtMax.shape)
        # 在 W 维度上计算最大值
        # max_along_w, _ = similarityWeihtMax.max(dim=1, keepdim=True)  # [B, 1, 1, H]
        # # 在 H 维度上计算最大值
        # similarityWeihtMaxMax, _ = max_along_w.max(dim=2, keepdim=True)  # [B, 1, 1, 1]
        # similarityWeihtMax = similarityWeihtMax / similarityWeihtMaxMax
        assp_featuresFC= assp_features.permute(0, 2, 3, 1).reshape(-1, 128)
        assp_featuresFC=self.value(assp_featuresFC).view(assp_features.size(0), assp_features.size(2), assp_features.size(3), assp_features.size(1)).permute(0, 3, 1, 2)
        assp_weighted=assp_featuresFC*similarityWeihtMax.unsqueeze(1)
        assp_weighted=F.relu(assp_weighted)
        # assp_weighted=(assp_features)*similarityWeihtMax.unsqueeze(1)

        x = self.classifier(assp_weighted)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h//4, w//4), mode='bilinear', align_corners=True)

        return {'out':x,'outUp':xup},{'CurrentPorotype':None,'GetProto':query_outputcat,'query':prototypes},\
               {'asspF':assp_features,'asspFW':assp_weighted,'cat':similarityCat,'param':[aOut,bOut], 'DistP':DistP,'Weight':[similarityWeiht,DistPMax,similarityWeihtMax]}


class DeepLabV3PlusMultiPrototypeSingleKeyBias2(nn.Module):
    def __init__(self, num_classes=21,args=None,n_cluster=1):
        super(DeepLabV3PlusMultiPrototypeSingleKeyBias2, self).__init__()
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
        # self.key=nn.Conv2d(featDim, 128, kernel_size=1)
        self.value=nn.Conv2d(featDim, 128, kernel_size=1)
        # nn.init.constant_(self.value.weight, 1)
        # nn.init.constant_(self.value.bias, 0)
        for i in range(num_classes):
            # setattr(self, f'query_{i}', nn.Conv2d(featDim * self.prototypeN, 128, kernel_size=1))
            setattr(self, f'a_{i}', nn.Parameter(torch.ones(1),requires_grad=True))
            setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1),requires_grad=True))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
            # self.a = nn.Parameter(torch.ones(1))  # 初始化为 1
            # self.b = nn.Parameter(torch.zeros(1))  # 初始化为 0
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
        # with torch.no_grad():
        #     self.value.weight.clamp_(0.9, 1.1)
        #     self.value.bias.clamp_(-0.1, 0.1)

        prototypes = []
        if DomainLabel == 0:
            with torch.no_grad():
                pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
                #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
                mask = torch.argmax(pseudo_out, dim=1).unsqueeze(1)
            # 计算每个类别的原型
            for i in range(self.num_classes):
                class_mask = (mask == i).float()
                if maskParam is not None:
                    # print('class_mask * maskParam',class_mask.shape , maskParam.shape)
                    class_mask = class_mask * maskParam  # Apply maskParam
                if class_mask.sum() > 0:  # 确保类别在批次中存在
                    # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                    prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]) + 1e-5)#([10, 256])
                    # for pp in range(self.n_cluster):
                        # print('prototype',prototype.shape)
                    prototypes.append(prototype.unsqueeze(1))
                else:
                    # for pp in range(self.n_cluster):
                    prototypes.append(zero_prototype.unsqueeze(1))
        elif DomainLabel==1 and ProtoInput!=None:
            prototypes=ProtoInput
        # print('lem',len(prototypes))
        # 将原型输入到对应的query层
        total_weight=0
        # key_output = self.key(assp_features)
        key_output = assp_features.view(assp_features.size(0), assp_features.size(1), -1)  # Reshape to [10, 128, 16*16]
        # print('key_output',key_output.shape)
        # query_outputList=[]
        similarityList=[]
        query_outputList=[]
        aList=[]
        bList=[]
        for i, prototype in enumerate(prototypes):
            if prototype is not None:
                inproto=prototype[:,0,:].unsqueeze(-1)
                # query_output = getattr(self, f'query_{i//(len(prototypes)//6)}')(prototype[:,0,:].unsqueeze(-1).unsqueeze(-1))
                query_output= getattr(self, f'a_{i//(len(prototypes)//6)}') * inproto + getattr(self, f'b_{i//(len(prototypes)//6)}')
                query_output = query_output.view(query_output.size(0), 1, query_output.size(1))  # Reshape to [10, 1, 128]
                # query_outputList.append(query_output)
                query_outputList.append(query_output)
                similarity = torch.abs(torch.bmm(query_output, key_output)/128)  # Perform batch matrix multiplication#torch.Size([10, 1, 256])
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                               assp_features.size(3))#([10, 1, 16, 16])
                similarityList.append(similarity)
                aList.append(getattr(self, f'a_{i // (len(prototypes) // 6)}').unsqueeze(0))
                bList.append(getattr(self, f'b_{i // (len(prototypes) // 6)}').unsqueeze(0))
        aOut=torch.cat(aList,dim=0)
        bOut=torch.cat(bList,dim=0)
        query_outputcat = torch.cat(query_outputList, dim=1).unsqueeze(-1).unsqueeze(-1)
        prototypes_expanded = query_outputcat.expand(-1, -1, -1, 32, 32)  # [B, 6, 128, H, W]
        deep_features_expanded = assp_features.unsqueeze(1)  # [B, 1, 128, H, W]
        # deep_features_expanded = deep_features_expanded.expand(-1, 1, -1, -1, -1)  # [B, 6, 128, H, W]
        distances = torch.norm(prototypes_expanded - deep_features_expanded, p=2, dim=2)  # [B, 6, H, W]
        DistP = F.softmax(-distances, dim=1)  # [B, 6, H, W]
        DistPMax,_=torch.max(DistP,dim=1)

        similarityCat=torch.cat(similarityList,dim=1)
        similarityWeiht=F.softmax(similarityCat/similarityCat.size(1)*self.n_cluster,dim=1)
        # similarityWeihtMean=similarityWeiht.mean(dim=1)

        similarityWeihtMax,_=torch.max(similarityWeiht,dim=1)

        # similarityWeihtMaxMax=similarityWeihtMax.max(dim=(1, 2), keepdim=True).values
        # similarityWeihtMax = similarityWeihtMax / similarityWeihtMaxMax

        assp_weighted=self.value(assp_features)*similarityWeihtMax.unsqueeze(1)

        x = self.classifier(assp_weighted)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h//4, w//4), mode='bilinear', align_corners=True)

        return {'out':x,'outUp':xup},{'CurrentPorotype':None,'GetProto':query_outputcat,'query':prototypes},\
               {'asspF':assp_features,'asspFW':assp_weighted,'cat':similarityCat,'param':[aOut,bOut], 'Weight':[similarityWeiht,DistPMax,similarityWeihtMax]}