import torch
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from torch.nn import functional as F
import math
# class OriCenterVisualization():
import matplotlib.pyplot as plt
from IPython import display
from sklearn.decomposition import PCA
import pickle
# import skfuzzy as fuzz
# from util.ProtoDistValidate import classify_out,multiFeatClaSum
# from util.ProtoDistValidate import classify_out
from matplotlib.colors import ListedColormap
import cv2
def calculate_variance(X):
    return torch.var(X, dim=0)

def normalize_variances(variances):
    min_var = torch.min(variances)
    max_var = torch.max(variances)
    return (variances - min_var) / (max_var - min_var)

def determine_cluster_centers(normalized_variances, threshold, K_min, K_max):
    cluster_centers = []
    for var in normalized_variances:
        if var < threshold:
            cluster_centers.append(K_min)
        else:
            cluster_centers.append(K_max)
    return cluster_centers
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
        # print('a',index, value)
        # 更新累加和
        if value == 0:
            return None
        if M == 'max':
            Feat[:,index]=Featin[:,cumulative_sum:cumulative_sum+value].max(dim=1)[0]
        elif M == 'sum':
            Feat[:,index]=Featin[:,cumulative_sum:cumulative_sum+value].sum(dim=1)
        elif M == 'mean':
            Feat[:,index]=Featin[:,cumulative_sum:cumulative_sum+value].mean(dim=1)
        cumulative_sum += value
        # 检查out是否在当前累加的区间内
    return Feat  # 返回当前区间的索引
class GetIntCentriodDP():
    def __init__(self,opt,dataloader,device=None,Osize=32,n_cluster=1):
        # self.model=model
        self.opt=opt
        self.DEVICE=device
        self.dataloader=dataloader
        self.Osize=Osize
        self.scaler = StandardScaler()
        self.n_cluster=n_cluster

    def getClusterKmean(self,model,iterFlag=False):
        self.model=model
        self.opt.phase = 'val'
        # Load Data
        val_dataload = self.dataloader.load_data()
        iter_val = iter(val_dataload)
        val_data_len = len(val_dataload)
        self.model.eval()
        # netTeacher.eval()
        tbar = tqdm(range(val_data_len - 1))
        # train_data_len = len_source_loader
        # Osize = 32
        class_features_list = [[] for _ in range(6)]
        centerCur=None
        with torch.no_grad():
            for i in tbar:
                current_means = []
                data_val = next(iter_val)
                ############## Forward Pass ######################
                data_img_val = Variable(data_val['img_full']).to(self.DEVICE)
                label = Variable(data_val['label_full']).to(self.DEVICE)
                assp_feature = self.model.forward(data_img_val,getPFlag=True)
                assp_features = assp_feature
                # print('assp_features', assp_features.shape)
                label = F.interpolate(label.unsqueeze(1).float(), size=(self.Osize, self.Osize),
                                      mode='nearest').long()  # torch.Size([10, 1, 16, 16])
                label=label.squeeze(1)
                # c=assp_features.size(1)
                centerCur = self.GetCenterS3_6c(assp_features, label.long(),centroidsLast=centerCur,device=self.DEVICE)
                for jj in range(self.opt.batch_size):
                    for ii in range(6):
                        if centerCur[jj][ii,: ].sum()!=0:
                            class_features_list[ii].append(centerCur[jj][ii,: ].unsqueeze(0))
                        # print(centerCur[jj][ii,: ].unsqueeze(0).shape)
                if i > 10 and iterFlag:
                    break
            # with open('class_features_list.pkl', 'wb') as file:
            #     pickle.dump(class_features_list, file)
            # with open('class_features_list.pkl', 'rb') as file:
            #     class_features_list = pickle.load(file)
            class_centroids = []
            n_components = 64  # PCA降维后的维数
            # pca = PCA(n_components=n_components)
            for class_idx in range(6):
                # 合并当前类别的所有特征
                if class_features_list[class_idx]:
                    features_matrix = torch.cat(class_features_list[class_idx], 0).view(-1, 128).detach().cpu().numpy()
                    # features_matrixMean=np.mean(features_matrix,axis=0)
                    # features_matrixMean=np.expand_dims(features_matrixMean,axis=0)
                    # print('features_matrix',features_matrix.shape,features_matrixMean.shape)
                    # 应用K-means聚类
                    features_matrix = self.scaler.fit_transform(features_matrix)
                    ###PCA
                    # features_matrix = pca.fit_transform(features_matrix)
                    if self.n_cluster[class_idx]<3:
                        kmeans = KMeans(n_clusters=self.n_cluster[class_idx], random_state=0).fit(features_matrix)
                        centroids = kmeans.cluster_centers_#2,128
                    else:
                        kmeans = KMeans(n_clusters=self.n_cluster[class_idx] - 1, random_state=0).fit(features_matrix)
                        centroids = kmeans.cluster_centers_  # 2,128

                        kmeans2 = KMeans(n_clusters=1, random_state=0).fit(features_matrix)
                        centroids2 = kmeans2.cluster_centers_
                        centroids=np.concatenate([centroids,centroids2],axis=0)


                    # print('centroids',class_idx,self.n_cluster[class_idx],max(self.n_cluster),centroids.shape)

                    ###inv PCA
                    # centroids = pca.inverse_transform(centroids)

                    centroids = self.scaler.inverse_transform(centroids)
                    expand_Center = np.expand_dims(centroids, axis=0)

                    expanded_cluster_centers_ = np.tile(expand_Center, (self.opt.batch_size, 1, 1))  # 在第一个维度复制10次，第二个维度保持不变
                    # print('expanded_cluster_centers_',expanded_cluster_centers_.shape)
                    expanded_cluster_centers_=torch.tensor(expanded_cluster_centers_, requires_grad=False).to(self.DEVICE)
                    class_centroids.append(expanded_cluster_centers_)
                    # print('expanded_cluster_centers_',expanded_cluster_centers_.shape)
            PrototypeScat=torch.cat(class_centroids,dim=1).unsqueeze(-1).unsqueeze(-1)
            return PrototypeScat
    def GetCenterS3_6c(self,features, pseudo_labels, centroidsLast=None, device=None):
        """
        Calculate centroids for multiple classes.
        :param features: Tensor of shape (batch_size, num_channels, height, width)
        :param pseudo_labels: Tensor of shape (batch_size, height, width) with values from 0 to num_classes-1
        :param centroidsLast: Optional; last centroids tensor of shape (num_classes, num_channels)
        :param device: Optional; torch.device on which tensors should be
        :return: Tensor of shape (num_channels, num_classes) representing the centroids
        """
        num_classes = 6
        batch_size, num_channels, height, width = features.shape
        centroids = torch.zeros(num_classes, num_channels).to(device)
        num= torch.zeros(6, 1).to(device)

        outcentroids=[]
        # if centroidsLast is None:
        #     # centroidsLast = torch.zeros(num_classes, num_channels).to(device)
        # else:
        #     # print('centroidsLast',centroidsLast)
        #     for ii in range(len(centroidsLast)):
        #         # print(ii,centroidsLast[-ii].shape)
        #         if centroidsLast[-ii].sum() !=0:
        #             # centroidsLast = centroidsLast[-ii]
        #             break
        #         elif ii == len(centroidsLast)-1:
        #             # centroidsLast = torch.zeros(num_classes, num_channels).to(device)
        #             break
        # Compute centroids for each class
        for batch in range(batch_size):
            batch_features = features[batch, :, :, :].unsqueeze(0)  # [1, num_channels, height, width]
            for class_id in range(num_classes):
                mask = (pseudo_labels[batch] == class_id).float().unsqueeze(0)  # [1, 1, height, width]
                if mask.sum() > 0:
                    centroid = (batch_features * mask).sum(dim=[2, 3]) / mask.sum()  # [1, num_channels]
                    centroids[class_id, :] = centroid.squeeze(0)
                    num[class_id,:]=1
                # else:
                #     centroids[class_id, :] += centroidsLast[class_id, :]
            # print('centroids',centroids.shape)
            outcentroids.append(centroids)
            centroids = torch.zeros(num_classes, num_channels).to(device)
        # batch_features = features[:, :, :, :] # [1, num_channels, height, width]([10, 128, 32, 32])
        # for class_id in range(num_classes):
        #     mask = (pseudo_labels == class_id).float().unsqueeze(1)  # [1, 1, height, width]
        #     if mask.sum() > 0:
        #         print(batch_features.shape , mask.shape)
        #         centroid = (batch_features * mask).sum(dim=[2, 3]) / (mask.sum(dim=[2, 3]) + 1e-5)   # [1, num_channels]([10, 1, 32, 32])
        #         print('centroid',centroid.shape)#10, 128]
        #         centroids[class_id, :] += centroid.squeeze(0)
        #     else:
        #         centroids[class_id, :] += centroidsLast[class_id, :]
        #     # print('centroids',centroids.shape)
        #     outcentroids.append(centroids)
        #     centroids = torch.zeros(num_classes, num_channels).to(device)
        # Normalize by the number of batches
        # centroids = (centroids / batch_size).permute(1, 0)
        return outcentroids
def getCurrentPro(feature,pseudo_labels,confidence,device):
    batch_size, num_channels, height, width = feature.shape
    num_classes = 6
    outcentroids=[]
    # num = torch.zeros(6, 1).to(device)
    centroids = torch.zeros(num_classes, num_channels).to(device)

    if pseudo_labels is None:
        softmax_output = F.softmax(feature.detach(), dim=1)
        pseudo_labels = torch.argmax(softmax_output, dim=1)
        pseudo_labels_prob = torch.max(softmax_output, dim=1)[0]  # 形状为 [B, H, W]
        pseudo_labels[pseudo_labels_prob < confidence] = 250

    # mean_features = torch.zeros(10, 6, 128)
    for batch in range(batch_size):
        batch_features = feature[batch, :, :, :].unsqueeze(0)  # [1, num_channels, height, width]
        for class_id in range(num_classes):
            mask = (pseudo_labels[batch] == class_id).float().unsqueeze(0)  # [1, 1, height, width]
            if mask.sum() > 0:
                centroid = (batch_features * mask).sum(dim=[2, 3]) / mask.sum()  # [1, num_channels]
                centroids[class_id, :] = centroid.squeeze(0)
            # else:
            #     centroids[class_id, :] += centroidsLast[class_id, :]
        # print('centroids',centroids.shape)
        outcentroids.append(centroids.unsqueeze(0))

        centroids = torch.zeros(num_classes, num_channels).to(device)
    outcentroids=torch.cat(outcentroids,dim=0)
    # print('outcentroids',outcentroids.shape)
    return outcentroids
class GetIntCentriodDPST():
    def __init__(self, opt, dataloaderS, dataloaderT, device=None, Osize=32, n_cluster=1):
        # self.model=model
        self.opt = opt
        self.DEVICE = device
        self.dataloaderS = dataloaderS
        self.dataloaderT = dataloaderT

        self.Osize = Osize
        self.scaler = StandardScaler()
        self.n_cluster = n_cluster

    def getClusterKmean(self,model,iterFlag=False,STFlag=False,confidence_threshold=0.9):
        self.model = model
        self.opt.phase = 'val'
        # Load Data
        S_dataload = self.dataloaderS.load_data()
        S_iter = iter(S_dataload)
        S_data_len = len(S_dataload)

        T_dataload = self.dataloaderT.load_data()
        T_iter = iter(T_dataload)
        T_data_len = len(T_dataload)
        self.model.eval()
        # netTeacher.eval()
        tbar = tqdm(range(S_data_len - 1))
        # train_data_len = len_source_loader
        # Osize = 32
        class_features_list = [[] for _ in range(6)]
        class_features_listS = [[] for _ in range(6)]
        class_features_listT = [[] for _ in range(6)]

        centerCur = None
        class_countsST = torch.zeros(6, dtype=torch.int64)
        class_countsTT = torch.zeros(6, dtype=torch.int64)
        class_countsTTori = torch.zeros(6, dtype=torch.int64)
        with torch.no_grad():
            for i in tbar:
                try:
                    Sdata = next(S_iter)
                except:
                    S_iter = iter(S_dataload)
                try:
                    Tdata = next(T_iter)
                except:
                    T_iter = iter(T_dataload)
                ############## Forward Pass ######################
                S_full_img = Variable(Sdata['img_full']).to(self.DEVICE)
                S_label = Variable(Sdata['label_full']).to(self.DEVICE)
                assp_feature = self.model.forward(S_full_img, DomainLabel=1, getPFlag=True)
                assp_features = assp_feature
                S_label = F.interpolate(S_label.unsqueeze(1).float(), size=(self.Osize, self.Osize),
                                        mode='nearest').long()  # torch.Size([10, 1, 16, 16])
                S_label = S_label.squeeze(1)
                class_countsST = class_countsST + count_elements_per_class(S_label)
                # print('assp_features',assp_features.shape,S_label.shape)
                centerCurS = self.GetCenterS3_6c(assp_features, S_label.long(), centroidsLast=centerCur,
                                                 device=self.DEVICE)
                if STFlag:
                    T_full_img = Variable(Tdata['img_full']).to(self.DEVICE)
                    #####only test
                    T_label = Variable(Tdata['label_full']).to(self.DEVICE)
                    class_countsTTori = class_countsTTori + count_elements_per_class(T_label)

                    T_pred, _, TFeat = model.forward(T_full_img, DomainLabel=1, ProtoInput=None,getPFlag=False)
                    Tsoft = F.softmax(T_pred['outUp'].detach(), dim=1)
                    pseudo_labels = torch.argmax(Tsoft, dim=1)
                    pseudo_labels_prob = torch.max(Tsoft, dim=1)[0]  # 形状为 [B, H, W]
                    # confidence_threshold = 0.9
                    pseudo_labels[pseudo_labels_prob < confidence_threshold] = 250
                    pseudo_labels = F.interpolate(pseudo_labels.unsqueeze(1).float(), size=(self.Osize, self.Osize),
                                                  mode='nearest').long()  # torch.Size([10, 1, 16, 16])
                    centerCurT = self.GetCenterS3_6c(TFeat['asspF'], pseudo_labels.long(), centroidsLast=centerCur,
                                                     device=self.DEVICE)
                    class_countsTT = class_countsTT + count_elements_per_class(pseudo_labels)
                for jj in range(self.opt.batch_size):
                    for ii in range(6):
                        # print('centerCurT',jj, centerCurT[jj].shape, centerCurS[jj].shape)

                        if centerCurS[jj][ii, :].sum() != 0:
                            class_features_list[ii].append(centerCurS[jj][ii, :].unsqueeze(0))
                            class_features_listS[ii].append(centerCurS[jj][ii, :].unsqueeze(0))
                        if STFlag:
                            if centerCurT[jj][ii, :].sum() != 0:
                                class_features_list[ii].append(centerCurT[jj][ii, :].unsqueeze(0))
                                class_features_listT[ii].append(centerCurT[jj][ii, :].unsqueeze(0))

                if i > 10 and iterFlag:
                    break
            # with open('class_features_list.pkl', 'wb') as file:
            #     pickle.dump(class_features_list, file)
            # with open('class_features_list.pkl', 'rb') as file:
            #     class_features_list = pickle.load(file)
            class_centroids = []
            rate = class_countsST / class_countsST.sum()
            # print('Srate:',rate)
            print('Srate:', rate / (rate).min())
            rate = class_countsTT / class_countsTT.sum()
            print('Trate:', rate / (rate).min())
            rate = class_countsTTori / class_countsTTori.sum()
            print('TOrirate:', rate / (rate).min())

            class_centroids = []

            # for varValue in [class_features_list,class_features_listS,class_features_listT]:
            #     variance_classT=[]
            #     for class_idx in range(6):
            #         # 合并当前类别的所有特征
            #         if varValue[class_idx]:
            #             features_matrix = torch.cat(varValue[class_idx], 0).view(-1,128).detach()
            #             variance_class1 = calculate_variance(features_matrix).mean()
            #             variance_classT.append(variance_class1)
            #     variance_classT=torch.tensor(variance_classT)
            #     # print(variance_classT)
            #     ratevar=variance_classT/variance_classT.min()
            #     print('ratevar',ratevar)

            # for varValue in [class_features_list,class_features_listT]:
            #     variance_classT=[]
            #     for class_idx in range(6):
            #         # 合并当前类别的所有特征
            #         if varValue[class_idx]:
            #             features_matrix = torch.cat(varValue[class_idx], 0).view(-1,128).detach()
            #             variance_class1 = calculate_variance(features_matrix).mean()
            #             variance_classT.append(variance_class1)
            #     variance_classT=torch.tensor(variance_classT)
            #     # print(variance_classT)
            #     ratevar=variance_classT/variance_classT.min()
            #     print('ratevar',ratevar)

            n_components = 64  # PCA降维后的维数
            # pca = PCA(n_components=n_components)
            for class_idx in range(6):
                # 合并当前类别的所有特征
                if class_features_list[class_idx]:
                    features_matrix = torch.cat(class_features_list[class_idx], 0).view(-1,
                                                                                        128).detach().cpu().numpy()
                    # features_matrixMean=np.mean(features_matrix,axis=0)
                    # features_matrixMean=np.expand_dims(features_matrixMean,axis=0)
                    # print('features_matrix',features_matrix.shape,features_matrixMean.shape)
                    # 应用K-means聚类
                    features_matrix = self.scaler.fit_transform(features_matrix)
                    ###PCA
                    # features_matrix = pca.fit_transform(features_matrix)
                    if self.n_cluster[class_idx] < 3:
                        kmeans = KMeans(n_clusters=self.n_cluster[class_idx], random_state=0).fit(features_matrix)
                        centroids = kmeans.cluster_centers_  # 2,128
                    else:
                        kmeans = KMeans(n_clusters=self.n_cluster[class_idx] - 1, random_state=0).fit(
                            features_matrix)
                        centroids = kmeans.cluster_centers_  # 2,128

                        kmeans2 = KMeans(n_clusters=1, random_state=0).fit(features_matrix)
                        centroids2 = kmeans2.cluster_centers_
                        centroids = np.concatenate([centroids, centroids2], axis=0)
                    ###inv PCA
                    # centroids = pca.inverse_transform(centroids)
                    if self.n_cluster[class_idx] == 1:
                        centroids = centroids.repeat(max(self.n_cluster), 0)

                    centroids = self.scaler.inverse_transform(centroids)
                    expand_Center = np.expand_dims(centroids, axis=0)
                    expanded_cluster_centers_ = np.tile(expand_Center,
                                                        (self.opt.batch_size, 1, 1))  # 在第一个维度复制10次，第二个维度保持不变
                    # print('expanded_cluster_centers_',expanded_cluster_centers_.shape)
                    expanded_cluster_centers_ = torch.tensor(expanded_cluster_centers_, requires_grad=False).to(
                        self.DEVICE)
                    class_centroids.append(expanded_cluster_centers_)
                    # print('expanded_cluster_centers_',expanded_cluster_centers_.shape)
            PrototypeScat = torch.cat(class_centroids, dim=1).unsqueeze(-1).unsqueeze(-1)
            return PrototypeScat
    def GetCenterS3_6c(self,features, pseudo_labels, centroidsLast=None, device=None):
        """
        Calculate centroids for multiple classes.
        :param features: Tensor of shape (batch_size, num_channels, height, width)
        :param pseudo_labels: Tensor of shape (batch_size, height, width) with values from 0 to num_classes-1
        :param centroidsLast: Optional; last centroids tensor of shape (num_classes, num_channels)
        :param device: Optional; torch.device on which tensors should be
        :return: Tensor of shape (num_channels, num_classes) representing the centroids
        """
        num_classes = 6
        batch_size, num_channels, height, width = features.shape
        centroids = torch.zeros(num_classes, num_channels).to(device)
        num= torch.zeros(6, 1).to(device)

        outcentroids=[]
        # if centroidsLast is None:
        #     # centroidsLast = torch.zeros(num_classes, num_channels).to(device)
        # else:
        #     # print('centroidsLast',centroidsLast)
        #     for ii in range(len(centroidsLast)):
        #         # print(ii,centroidsLast[-ii].shape)
        #         if centroidsLast[-ii].sum() !=0:
        #             # centroidsLast = centroidsLast[-ii]
        #             break
        #         elif ii == len(centroidsLast)-1:
        #             # centroidsLast = torch.zeros(num_classes, num_channels).to(device)
        #             break
        # Compute centroids for each class
        for batch in range(batch_size):
            batch_features = features[batch, :, :, :].unsqueeze(0)  # [1, num_channels, height, width]
            for class_id in range(num_classes):
                mask = (pseudo_labels[batch] == class_id).float().unsqueeze(0)  # [1, 1, height, width]
                if mask.sum() > 0:
                    centroid = (batch_features * mask).sum(dim=[2, 3]) / mask.sum()  # [1, num_channels]
                    centroids[class_id, :] = centroid.squeeze(0)
                    num[class_id,:]=1
                # else:
                #     centroids[class_id, :] += centroidsLast[class_id, :]
            # print('centroids',centroids.shape)
            outcentroids.append(centroids)
            centroids = torch.zeros(num_classes, num_channels).to(device)
        # batch_features = features[:, :, :, :] # [1, num_channels, height, width]([10, 128, 32, 32])
        # for class_id in range(num_classes):
        #     mask = (pseudo_labels == class_id).float().unsqueeze(1)  # [1, 1, height, width]
        #     if mask.sum() > 0:
        #         print(batch_features.shape , mask.shape)
        #         centroid = (batch_features * mask).sum(dim=[2, 3]) / (mask.sum(dim=[2, 3]) + 1e-5)   # [1, num_channels]([10, 1, 32, 32])
        #         print('centroid',centroid.shape)#10, 128]
        #         centroids[class_id, :] += centroid.squeeze(0)
        #     else:
        #         centroids[class_id, :] += centroidsLast[class_id, :]
        #     # print('centroids',centroids.shape)
        #     outcentroids.append(centroids)
        #     centroids = torch.zeros(num_classes, num_channels).to(device)
        # Normalize by the number of batches
        # centroids = (centroids / batch_size).permute(1, 0)
        return outcentroids

class GetIntCentriodDPT():
    def __init__(self, opt, dataloaderS, dataloaderT, device=None, Osize=32, n_cluster=1):
        # self.model=model
        self.opt = opt
        self.DEVICE = device
        self.dataloaderS = dataloaderS
        self.dataloaderT = dataloaderT

        self.Osize = Osize
        self.scaler = StandardScaler()
        self.n_cluster = n_cluster

    def getClusterKmean(self,model,iterFlag=False,STFlag=False,confidence_threshold=0.9):
        self.model = model
        self.opt.phase = 'val'
        # Load Data
        S_dataload = self.dataloaderS.load_data()
        S_iter = iter(S_dataload)
        S_data_len = len(S_dataload)

        T_dataload = self.dataloaderT.load_data()
        T_iter = iter(T_dataload)
        T_data_len = len(T_dataload)
        self.model.eval()
        # netTeacher.eval()
        tbar = tqdm(range(T_data_len - 1))
        # train_data_len = len_source_loader
        # Osize = 32
        class_features_list = [[] for _ in range(6)]
        class_features_listS = [[] for _ in range(6)]
        class_features_listT = [[] for _ in range(6)]

        centerCur = None
        class_countsST = torch.zeros(6, dtype=torch.int64)
        class_countsTT = torch.zeros(6, dtype=torch.int64)
        class_countsTTori = torch.zeros(6, dtype=torch.int64)
        with torch.no_grad():
            for i in tbar:
                try:
                    Sdata = next(S_iter)
                except:
                    S_iter = iter(S_dataload)
                try:
                    Tdata = next(T_iter)
                except:
                    T_iter = iter(T_dataload)
                ############## Forward Pass ######################
                # S_full_img = Variable(Sdata['img_full']).to(self.DEVICE)
                # S_label = Variable(Sdata['label_full']).to(self.DEVICE)
                # assp_feature = self.model.forward(S_full_img, DomainLabel=1, getPFlag=True)
                # assp_features = assp_feature
                # S_label = F.interpolate(S_label.unsqueeze(1).float(), size=(self.Osize, self.Osize),
                #                         mode='nearest').long()  # torch.Size([10, 1, 16, 16])
                # S_label = S_label.squeeze(1)
                # class_countsST = class_countsST + count_elements_per_class(S_label)
                # # print('assp_features',assp_features.shape,S_label.shape)
                # centerCurS = self.GetCenterS3_6c(assp_features, S_label.long(), centroidsLast=centerCur,
                #                                  device=self.DEVICE)
                if STFlag:
                    T_full_img = Variable(Tdata['img_full']).to(self.DEVICE)
                    #####only test
                    T_label = Variable(Tdata['label_full']).to(self.DEVICE)
                    class_countsTTori = class_countsTTori + count_elements_per_class(T_label)

                    T_pred, _, TFeat = model.forward(T_full_img, DomainLabel=1, ProtoInput=None,getPFlag=False)
                    Tsoft = F.softmax(T_pred['outUp'].detach(), dim=1)
                    pseudo_labels = torch.argmax(Tsoft, dim=1)
                    pseudo_labels_prob = torch.max(Tsoft, dim=1)[0]  # 形状为 [B, H, W]
                    # confidence_threshold = 0.9
                    pseudo_labels[pseudo_labels_prob < confidence_threshold] = 250
                    pseudo_labels = F.interpolate(pseudo_labels.unsqueeze(1).float(), size=(self.Osize, self.Osize),
                                                  mode='nearest').long()  # torch.Size([10, 1, 16, 16])
                    centerCurT = self.GetCenterS3_6c(TFeat['asspF'], pseudo_labels.long(), centroidsLast=centerCur,
                                                     device=self.DEVICE)
                    class_countsTT = class_countsTT + count_elements_per_class(pseudo_labels)
                for jj in range(self.opt.batch_size):
                    for ii in range(6):
                        # print('centerCurT',jj, centerCurT[jj].shape, centerCurS[jj].shape)

                        # if centerCurS[jj][ii, :].sum() != 0:
                        #     class_features_list[ii].append(centerCurS[jj][ii, :].unsqueeze(0))
                        #     class_features_listS[ii].append(centerCurS[jj][ii, :].unsqueeze(0))
                        if STFlag:
                            if centerCurT[jj][ii, :].sum() != 0:
                                class_features_list[ii].append(centerCurT[jj][ii, :].unsqueeze(0))
                                class_features_listT[ii].append(centerCurT[jj][ii, :].unsqueeze(0))

                if i > 10 and iterFlag:
                    break
            # with open('class_features_list.pkl', 'wb') as file:
            #     pickle.dump(class_features_list, file)
            # with open('class_features_list.pkl', 'rb') as file:
            #     class_features_list = pickle.load(file)
            class_centroids = []
            rate = class_countsST / class_countsST.sum()
            # print('Srate:',rate)
            print('Srate:', rate / (rate).min())
            rate = class_countsTT / class_countsTT.sum()
            print('Trate:', rate / (rate).min())
            rate = class_countsTTori / class_countsTTori.sum()
            print('TOrirate:', rate / (rate).min())

            class_centroids = []

            # for varValue in [class_features_list,class_features_listS,class_features_listT]:
            #     variance_classT=[]
            #     for class_idx in range(6):
            #         # 合并当前类别的所有特征
            #         if varValue[class_idx]:
            #             features_matrix = torch.cat(varValue[class_idx], 0).view(-1,128).detach()
            #             variance_class1 = calculate_variance(features_matrix).mean()
            #             variance_classT.append(variance_class1)
            #     variance_classT=torch.tensor(variance_classT)
            #     # print(variance_classT)
            #     ratevar=variance_classT/variance_classT.min()
            #     print('ratevar',ratevar)

            for varValue in [class_features_list,class_features_listT]:
                variance_classT=[]
                for class_idx in range(6):
                    # 合并当前类别的所有特征
                    if varValue[class_idx]:
                        features_matrix = torch.cat(varValue[class_idx], 0).view(-1,128).detach()
                        variance_class1 = calculate_variance(features_matrix).mean()
                        variance_classT.append(variance_class1)
                variance_classT=torch.tensor(variance_classT)
                # print(variance_classT)
                ratevar=variance_classT/variance_classT.min()
                print('ratevar',ratevar)

            n_components = 64  # PCA降维后的维数
            # pca = PCA(n_components=n_components)
            for class_idx in range(6):
                # 合并当前类别的所有特征
                if class_features_list[class_idx]:
                    features_matrix = torch.cat(class_features_list[class_idx], 0).view(-1,
                                                                                        128).detach().cpu().numpy()
                    # features_matrixMean=np.mean(features_matrix,axis=0)
                    # features_matrixMean=np.expand_dims(features_matrixMean,axis=0)
                    # print('features_matrix',features_matrix.shape,features_matrixMean.shape)
                    # 应用K-means聚类
                    features_matrix = self.scaler.fit_transform(features_matrix)
                    ###PCA
                    # features_matrix = pca.fit_transform(features_matrix)
                    if self.n_cluster[class_idx] < 3:
                        kmeans = KMeans(n_clusters=self.n_cluster[class_idx], random_state=0).fit(features_matrix)
                        centroids = kmeans.cluster_centers_  # 2,128
                    else:
                        kmeans = KMeans(n_clusters=self.n_cluster[class_idx] - 1, random_state=0).fit(
                            features_matrix)
                        centroids = kmeans.cluster_centers_  # 2,128

                        kmeans2 = KMeans(n_clusters=1, random_state=0).fit(features_matrix)
                        centroids2 = kmeans2.cluster_centers_
                        centroids = np.concatenate([centroids, centroids2], axis=0)
                    ###inv PCA
                    # centroids = pca.inverse_transform(centroids)
                    if self.n_cluster[class_idx] == 1:
                        centroids = centroids.repeat(max(self.n_cluster), 0)

                    centroids = self.scaler.inverse_transform(centroids)
                    expand_Center = np.expand_dims(centroids, axis=0)
                    expanded_cluster_centers_ = np.tile(expand_Center,
                                                        (self.opt.batch_size, 1, 1))  # 在第一个维度复制10次，第二个维度保持不变
                    # print('expanded_cluster_centers_',expanded_cluster_centers_.shape)
                    expanded_cluster_centers_ = torch.tensor(expanded_cluster_centers_, requires_grad=False).to(
                        self.DEVICE)
                    class_centroids.append(expanded_cluster_centers_)
                    # print('expanded_cluster_centers_',expanded_cluster_centers_.shape)
            PrototypeScat = torch.cat(class_centroids, dim=1).unsqueeze(-1).unsqueeze(-1)
            return PrototypeScat
    def GetCenterS3_6c(self,features, pseudo_labels, centroidsLast=None, device=None):
        """
        Calculate centroids for multiple classes.
        :param features: Tensor of shape (batch_size, num_channels, height, width)
        :param pseudo_labels: Tensor of shape (batch_size, height, width) with values from 0 to num_classes-1
        :param centroidsLast: Optional; last centroids tensor of shape (num_classes, num_channels)
        :param device: Optional; torch.device on which tensors should be
        :return: Tensor of shape (num_channels, num_classes) representing the centroids
        """
        num_classes = 6
        batch_size, num_channels, height, width = features.shape
        centroids = torch.zeros(num_classes, num_channels).to(device)
        num= torch.zeros(6, 1).to(device)

        outcentroids=[]
        # if centroidsLast is None:
        #     # centroidsLast = torch.zeros(num_classes, num_channels).to(device)
        # else:
        #     # print('centroidsLast',centroidsLast)
        #     for ii in range(len(centroidsLast)):
        #         # print(ii,centroidsLast[-ii].shape)
        #         if centroidsLast[-ii].sum() !=0:
        #             # centroidsLast = centroidsLast[-ii]
        #             break
        #         elif ii == len(centroidsLast)-1:
        #             # centroidsLast = torch.zeros(num_classes, num_channels).to(device)
        #             break
        # Compute centroids for each class
        for batch in range(batch_size):
            batch_features = features[batch, :, :, :].unsqueeze(0)  # [1, num_channels, height, width]
            for class_id in range(num_classes):
                mask = (pseudo_labels[batch] == class_id).float().unsqueeze(0)  # [1, 1, height, width]
                if mask.sum() > 0:
                    centroid = (batch_features * mask).sum(dim=[2, 3]) / mask.sum()  # [1, num_channels]
                    centroids[class_id, :] = centroid.squeeze(0)
                    num[class_id,:]=1
                # else:
                #     centroids[class_id, :] += centroidsLast[class_id, :]
            # print('centroids',centroids.shape)
            outcentroids.append(centroids)
            centroids = torch.zeros(num_classes, num_channels).to(device)
        # batch_features = features[:, :, :, :] # [1, num_channels, height, width]([10, 128, 32, 32])
        # for class_id in range(num_classes):
        #     mask = (pseudo_labels == class_id).float().unsqueeze(1)  # [1, 1, height, width]
        #     if mask.sum() > 0:
        #         print(batch_features.shape , mask.shape)
        #         centroid = (batch_features * mask).sum(dim=[2, 3]) / (mask.sum(dim=[2, 3]) + 1e-5)   # [1, num_channels]([10, 1, 32, 32])
        #         print('centroid',centroid.shape)#10, 128]
        #         centroids[class_id, :] += centroid.squeeze(0)
        #     else:
        #         centroids[class_id, :] += centroidsLast[class_id, :]
        #     # print('centroids',centroids.shape)
        #     outcentroids.append(centroids)
        #     centroids = torch.zeros(num_classes, num_channels).to(device)
        # Normalize by the number of batches
        # centroids = (centroids / batch_size).permute(1, 0)
        return outcentroids
class GetIntCentriodDPST2():
    def __init__(self, opt, dataloaderS, dataloaderT, device=None, Osize=32, n_cluster=1):
        # self.model=model
        self.opt = opt
        self.DEVICE = device
        self.dataloaderS = dataloaderS
        self.dataloaderT = dataloaderT

        self.Osize = Osize
        self.scaler = StandardScaler()
        self.n_cluster = n_cluster

    def getClusterKmean(self,model,iterFlag=False,STFlag=0,confidence_threshold=0.9):
        self.model = model
        self.opt.phase = 'val'
        # Load Data
        S_dataload = self.dataloaderS.load_data()
        S_iter = iter(S_dataload)
        S_data_len = len(S_dataload)

        T_dataload = self.dataloaderT.load_data()
        T_iter = iter(T_dataload)
        T_data_len = len(T_dataload)
        self.model.eval()
        # netTeacher.eval()
        tbar = tqdm(range(S_data_len - 1))
        # train_data_len = len_source_loader
        # Osize = 32
        class_features_list = [[] for _ in range(6)]
        centerCur = None
        class_countsST = torch.zeros(6, dtype=torch.int64)
        class_countsTT = torch.zeros(6, dtype=torch.int64)
        class_countsTTori = torch.zeros(6, dtype=torch.int64)
        with torch.no_grad():
            for i in tbar:
                try:
                    Sdata = next(S_iter)
                except:
                    S_iter = iter(S_dataload)
                try:
                    Tdata = next(T_iter)
                except:
                    T_iter = iter(T_dataload)
                ############## Forward Pass ######################
                S_full_img = Variable(Sdata['img_full']).to(self.DEVICE)
                S_label = Variable(Sdata['label_full']).to(self.DEVICE)
                assp_feature = self.model.forward(S_full_img, getPFlag=True)
                assp_features = assp_feature
                S_label = F.interpolate(S_label.unsqueeze(1).float(), size=(self.Osize, self.Osize),
                                        mode='nearest').long()  # torch.Size([10, 1, 16, 16])
                S_label = S_label.squeeze(1)
                class_countsST = class_countsST + count_elements_per_class(S_label)
                centerCurS = self.GetCenterS3_6c(assp_features, S_label.long(), centroidsLast=centerCur,
                                                 device=self.DEVICE)
                if STFlag:
                    T_full_img = Variable(Tdata['img_full']).to(self.DEVICE)
                    #####only test
                    T_label = Variable(Tdata['label_full']).to(self.DEVICE)
                    class_countsTTori = class_countsTTori + count_elements_per_class(T_label)

                    T_pred, _, TFeat = model.forward(T_full_img, DomainLabel=0, ProtoInput=None,getPFlag=False)
                    Tsoft = F.softmax(T_pred['outUp'].detach(), dim=1)
                    pseudo_labels = torch.argmax(Tsoft, dim=1)
                    pseudo_labels_prob = torch.max(Tsoft, dim=1)[0]  # 形状为 [B, H, W]
                    # confidence_threshold = 0.9
                    pseudo_labels[pseudo_labels_prob < confidence_threshold] = 250
                    pseudo_labels = F.interpolate(pseudo_labels.unsqueeze(1).float(), size=(self.Osize, self.Osize),
                                                  mode='nearest').long()  # torch.Size([10, 1, 16, 16])
                    centerCurT = self.GetCenterS3_6c(TFeat['asspF'], pseudo_labels.long(), centroidsLast=centerCur,
                                                     device=self.DEVICE)
                    class_countsTT = class_countsTT + count_elements_per_class(pseudo_labels)
                for jj in range(self.opt.batch_size):
                    for ii in range(6):

                        # if centerCurS[jj][ii, :].sum() != 0:
                        #     class_features_list[ii].append(centerCurS[jj][ii, :].unsqueeze(0))
                        if STFlag:
                            if centerCurT[jj][ii, :].sum() != 0:
                                class_features_list[ii].append(centerCurT[jj][ii, :].unsqueeze(0))
                if i > 10 and iterFlag:
                    break
            # with open('class_features_list.pkl', 'wb') as file:
            #     pickle.dump(class_features_list, file)
            # with open('class_features_list.pkl', 'rb') as file:
            #     class_features_list = pickle.load(file)
            class_centroids = []
            rate = class_countsST / class_countsST.sum()
            # print('Srate:',rate)
            print('Srate:', rate / (rate).min())
            rate = class_countsTT / class_countsTT.sum()
            print('Trate:', rate / (rate).min())
            rate = class_countsTTori / class_countsTTori.sum()
            print('TOrirate:', rate / (rate).min())

            class_centroids = []
            variance_classT=[]
            for class_idx in range(6):
                # 合并当前类别的所有特征
                if class_features_list[class_idx]:
                    features_matrix = torch.cat(class_features_list[class_idx], 0).view(-1,
                                                                                        128).detach()
                    variance_class1 = calculate_variance(features_matrix).mean()
                    variance_classT.append(variance_class1)
            variance_classT=torch.tensor(variance_classT)
            print(variance_classT)
            ratevar=variance_classT/variance_classT.min()
            print('ratevar',ratevar)


            n_components = 64  # PCA降维后的维数
            # pca = PCA(n_components=n_components)
            for class_idx in range(6):
                # 合并当前类别的所有特征
                if class_features_list[class_idx]:
                    features_matrix = torch.cat(class_features_list[class_idx], 0).view(-1,
                                                                                        128).detach().cpu().numpy()
                    # features_matrixMean=np.mean(features_matrix,axis=0)
                    # features_matrixMean=np.expand_dims(features_matrixMean,axis=0)
                    # print('features_matrix',features_matrix.shape,features_matrixMean.shape)
                    # 应用K-means聚类
                    features_matrix = self.scaler.fit_transform(features_matrix)
                    ###PCA
                    # features_matrix = pca.fit_transform(features_matrix)
                    if self.n_cluster[class_idx] < 3:
                        kmeans = KMeans(n_clusters=self.n_cluster[class_idx], random_state=0).fit(features_matrix)
                        centroids = kmeans.cluster_centers_  # 2,128
                    else:
                        kmeans = KMeans(n_clusters=self.n_cluster[class_idx] - 1, random_state=0).fit(
                            features_matrix)
                        centroids = kmeans.cluster_centers_  # 2,128

                        kmeans2 = KMeans(n_clusters=1, random_state=0).fit(features_matrix)
                        centroids2 = kmeans2.cluster_centers_
                        centroids = np.concatenate([centroids, centroids2], axis=0)
                    ###inv PCA
                    # centroids = pca.inverse_transform(centroids)
                    if self.n_cluster[class_idx] == 1:
                        centroids = centroids.repeat(max(self.n_cluster), 0)

                    centroids = self.scaler.inverse_transform(centroids)
                    expand_Center = np.expand_dims(centroids, axis=0)
                    expanded_cluster_centers_ = np.tile(expand_Center,
                                                        (self.opt.batch_size, 1, 1))  # 在第一个维度复制10次，第二个维度保持不变
                    # print('expanded_cluster_centers_',expanded_cluster_centers_.shape)
                    expanded_cluster_centers_ = torch.tensor(expanded_cluster_centers_, requires_grad=False).to(
                        self.DEVICE)
                    class_centroids.append(expanded_cluster_centers_)
                    # print('expanded_cluster_centers_',expanded_cluster_centers_.shape)
            PrototypeScat = torch.cat(class_centroids, dim=1).unsqueeze(-1).unsqueeze(-1)
            return PrototypeScat
    def GetCenterS3_6c(self,features, pseudo_labels, centroidsLast=None, device=None):
        """
        Calculate centroids for multiple classes.
        :param features: Tensor of shape (batch_size, num_channels, height, width)
        :param pseudo_labels: Tensor of shape (batch_size, height, width) with values from 0 to num_classes-1
        :param centroidsLast: Optional; last centroids tensor of shape (num_classes, num_channels)
        :param device: Optional; torch.device on which tensors should be
        :return: Tensor of shape (num_channels, num_classes) representing the centroids
        """
        num_classes = 6
        batch_size, num_channels, height, width = features.shape
        centroids = torch.zeros(num_classes, num_channels).to(device)
        num= torch.zeros(6, 1).to(device)

        outcentroids=[]
        # if centroidsLast is None:
        #     # centroidsLast = torch.zeros(num_classes, num_channels).to(device)
        # else:
        #     # print('centroidsLast',centroidsLast)
        #     for ii in range(len(centroidsLast)):
        #         # print(ii,centroidsLast[-ii].shape)
        #         if centroidsLast[-ii].sum() !=0:
        #             # centroidsLast = centroidsLast[-ii]
        #             break
        #         elif ii == len(centroidsLast)-1:
        #             # centroidsLast = torch.zeros(num_classes, num_channels).to(device)
        #             break
        # Compute centroids for each class
        for batch in range(batch_size):
            batch_features = features[batch, :, :, :].unsqueeze(0)  # [1, num_channels, height, width]
            for class_id in range(num_classes):
                mask = (pseudo_labels[batch] == class_id).float().unsqueeze(0)  # [1, 1, height, width]
                if mask.sum() > 0:
                    centroid = (batch_features * mask).sum(dim=[2, 3]) / mask.sum()  # [1, num_channels]
                    centroids[class_id, :] = centroid.squeeze(0)
                    num[class_id,:]=1
                # else:
                #     centroids[class_id, :] += centroidsLast[class_id, :]
            # print('centroids',centroids.shape)
            outcentroids.append(centroids)
            centroids = torch.zeros(num_classes, num_channels).to(device)
        # batch_features = features[:, :, :, :] # [1, num_channels, height, width]([10, 128, 32, 32])
        # for class_id in range(num_classes):
        #     mask = (pseudo_labels == class_id).float().unsqueeze(1)  # [1, 1, height, width]
        #     if mask.sum() > 0:
        #         print(batch_features.shape , mask.shape)
        #         centroid = (batch_features * mask).sum(dim=[2, 3]) / (mask.sum(dim=[2, 3]) + 1e-5)   # [1, num_channels]([10, 1, 32, 32])
        #         print('centroid',centroid.shape)#10, 128]
        #         centroids[class_id, :] += centroid.squeeze(0)
        #     else:
        #         centroids[class_id, :] += centroidsLast[class_id, :]
        #     # print('centroids',centroids.shape)
        #     outcentroids.append(centroids)
        #     centroids = torch.zeros(num_classes, num_channels).to(device)
        # Normalize by the number of batches
        # centroids = (centroids / batch_size).permute(1, 0)
        return outcentroids
def count_elements_per_class(labels, num_classes=6):
    class_counts = torch.zeros(num_classes, dtype=torch.int64)
    for i in range(num_classes):
        class_counts[i] = (labels == i).sum()
    return class_counts
class GetIntCentriodST():
    def __init__(self,opt,dataloaderS,dataloaderT,device=None,Osize=32,n_cluster=1):
        # self.model=model
        self.opt=opt
        self.DEVICE=device
        self.dataloaderS=dataloaderS
        self.dataloaderT=dataloaderT

        self.Osize=Osize
        self.scaler = StandardScaler()
        self.n_cluster=n_cluster
    def getClusterKmean(self,model,iterFlag=False,STFlag=False,confidence_threshold=0.9):
        self.model=model
        self.opt.phase = 'val'
        # Load Data
        S_dataload = self.dataloaderS.load_data()
        S_iter = iter(S_dataload)
        S_data_len = len(S_dataload)

        T_dataload = self.dataloaderT.load_data()
        T_iter = iter(T_dataload)
        T_data_len = len(T_dataload)
        self.model.eval()
        # netTeacher.eval()
        tbar = tqdm(range(S_data_len - 1))
        # train_data_len = len_source_loader
        # Osize = 32
        class_features_list = [[] for _ in range(6)]
        centerCur=None
        class_countsST = torch.zeros(6, dtype=torch.int64)
        class_countsTT = torch.zeros(6, dtype=torch.int64)
        class_countsTTori = torch.zeros(6, dtype=torch.int64)
        with torch.no_grad():
            for i in tbar:
                try:
                    Sdata = next(S_iter)
                except:
                    S_iter = iter(S_dataload)
                try:
                    Tdata = next(T_iter)
                except:
                    T_iter = iter(T_dataload)
                ############## Forward Pass ######################
                S_full_img = Variable(Sdata['img_full']).to(self.DEVICE)
                S_label = Variable(Sdata['label_full']).to(self.DEVICE)
                assp_feature = self.model.forward(S_full_img,getPFlag=True)
                assp_features = assp_feature
                S_label = F.interpolate(S_label.unsqueeze(1).float(), size=(self.Osize, self.Osize),
                                      mode='nearest').long()  # torch.Size([10, 1, 16, 16])
                S_label=S_label.squeeze(1)
                class_countsST = class_countsST+count_elements_per_class(S_label)
                centerCurS = self.GetCenterS3_6c(assp_features, S_label.long(),centroidsLast=centerCur,device=self.DEVICE)
                if STFlag:
                    T_full_img = Variable(Tdata['img_full']).to(self.DEVICE)
                    #####only test
                    T_label = Variable(Tdata['label_full']).to(self.DEVICE)
                    class_countsTTori = class_countsTTori+count_elements_per_class(T_label)

                    T_pred, _, TFeat = model.forward(T_full_img, DomainLabel=0,ProtoInput=None)
                    Tsoft = F.softmax(T_pred['outUp'].detach(),dim=1)
                    pseudo_labels = torch.argmax(Tsoft, dim=1)
                    pseudo_labels_prob = torch.max(Tsoft, dim=1)[0]  # 形状为 [B, H, W]
                    # confidence_threshold = 0.9
                    pseudo_labels[pseudo_labels_prob < confidence_threshold] = 250
                    pseudo_labels = F.interpolate(pseudo_labels.unsqueeze(1).float(), size=(self.Osize, self.Osize),
                                            mode='nearest').long()  # torch.Size([10, 1, 16, 16])
                    centerCurT = self.GetCenterS3_6c(TFeat['asspF'], pseudo_labels.long(),centroidsLast=centerCur,device=self.DEVICE)
                    class_countsTT = class_countsTT+count_elements_per_class(pseudo_labels)

                for jj in range(self.opt.batch_size):
                    for ii in range(6):
                        if centerCurS[jj][ii,: ].sum()!=0:
                            class_features_list[ii].append(centerCurS[jj][ii,: ].unsqueeze(0))
                        if STFlag:
                            if centerCurT[jj][ii, :].sum() != 0:
                                class_features_list[ii].append(centerCurT[jj][ii,: ].unsqueeze(0))
                if i > 10 and iterFlag:
                    break
            # with open('class_features_list.pkl', 'wb') as file:
            #     pickle.dump(class_features_list, file)
            # with open('class_features_list.pkl', 'rb') as file:
            #     class_features_list = pickle.load(file)
            class_centroids = []
            rate=class_countsST/class_countsST.sum()
            # print('Srate:',rate)
            print('Srate:',rate/(rate).min())
            rate = class_countsTT / class_countsTT.sum()
            print('Trate:',rate/(rate).min())
            rate = class_countsTTori / class_countsTTori.sum()
            print('TOrirate:', rate / (rate).min())
            n_components = 64  # PCA降维后的维数
            # pca = PCA(n_components=n_components)
            for class_idx in range(6):
                # 合并当前类别的所有特征
                if class_features_list[class_idx]:
                    features_matrix = torch.cat(class_features_list[class_idx], 0).view(-1, 128).detach().cpu().numpy()
                    # features_matrixMean=np.mean(features_matrix,axis=0)
                    # features_matrixMean=np.expand_dims(features_matrixMean,axis=0)
                    # print('features_matrix',features_matrix.shape,features_matrixMean.shape)
                    # 应用K-means聚类
                    features_matrix = self.scaler.fit_transform(features_matrix)
                    ###PCA
                    # features_matrix = pca.fit_transform(features_matrix)
                    if self.n_cluster==1:
                        kmeans = KMeans(n_clusters=self.n_cluster, random_state=0).fit(features_matrix)
                        centroids = kmeans.cluster_centers_#2,128
                    else:
                        kmeans = KMeans(n_clusters=self.n_cluster - 1, random_state=0).fit(features_matrix)
                        centroids = kmeans.cluster_centers_  # 2,128

                        kmeans2 = KMeans(n_clusters=1, random_state=0).fit(features_matrix)
                        centroids2 = kmeans2.cluster_centers_
                        centroids=np.concatenate([centroids,centroids2],axis=0)
                    ###inv PCA
                    # centroids = pca.inverse_transform(centroids)

                    centroids = self.scaler.inverse_transform(centroids)
                    expand_Center = np.expand_dims(centroids, axis=0)

                    expanded_cluster_centers_ = np.tile(expand_Center, (self.opt.batch_size, 1, 1))  # 在第一个维度复制10次，第二个维度保持不变
                    # print('expanded_cluster_centers_',expanded_cluster_centers_.shape)
                    expanded_cluster_centers_=torch.tensor(expanded_cluster_centers_, requires_grad=False).to(self.DEVICE)
                    class_centroids.append(expanded_cluster_centers_)
                    # print('expanded_cluster_centers_',expanded_cluster_centers_.shape)
            PrototypeScat=torch.cat(class_centroids,dim=1).unsqueeze(-1).unsqueeze(-1)
            return PrototypeScat
    def GetCenterS3_6c(self,features, pseudo_labels, centroidsLast=None, device=None):
        """
        Calculate centroids for multiple classes.
        :param features: Tensor of shape (batch_size, num_channels, height, width)
        :param pseudo_labels: Tensor of shape (batch_size, height, width) with values from 0 to num_classes-1
        :param centroidsLast: Optional; last centroids tensor of shape (num_classes, num_channels)
        :param device: Optional; torch.device on which tensors should be
        :return: Tensor of shape (num_channels, num_classes) representing the centroids
        """
        num_classes = 6
        batch_size, num_channels, height, width = features.shape
        centroids = torch.zeros(num_classes, num_channels).to(device)
        num= torch.zeros(6, 1).to(device)

        outcentroids=[]
        # if centroidsLast is None:
        #     # centroidsLast = torch.zeros(num_classes, num_channels).to(device)
        # else:
        #     # print('centroidsLast',centroidsLast)
        #     for ii in range(len(centroidsLast)):
        #         # print(ii,centroidsLast[-ii].shape)
        #         if centroidsLast[-ii].sum() !=0:
        #             # centroidsLast = centroidsLast[-ii]
        #             break
        #         elif ii == len(centroidsLast)-1:
        #             # centroidsLast = torch.zeros(num_classes, num_channels).to(device)
        #             break
        # Compute centroids for each class
        for batch in range(batch_size):
            batch_features = features[batch, :, :, :].unsqueeze(0)  # [1, num_channels, height, width]
            for class_id in range(num_classes):
                mask = (pseudo_labels[batch] == class_id).float().unsqueeze(0)  # [1, 1, height, width]
                if mask.sum() > 0:
                    centroid = (batch_features * mask).sum(dim=[2, 3]) / mask.sum()  # [1, num_channels]
                    centroids[class_id, :] = centroid.squeeze(0)
                    num[class_id,:]=1
                # else:
                #     centroids[class_id, :] += centroidsLast[class_id, :]
            # print('centroids',centroids.shape)
            outcentroids.append(centroids)
            centroids = torch.zeros(num_classes, num_channels).to(device)
        # batch_features = features[:, :, :, :] # [1, num_channels, height, width]([10, 128, 32, 32])
        # for class_id in range(num_classes):
        #     mask = (pseudo_labels == class_id).float().unsqueeze(1)  # [1, 1, height, width]
        #     if mask.sum() > 0:
        #         print(batch_features.shape , mask.shape)
        #         centroid = (batch_features * mask).sum(dim=[2, 3]) / (mask.sum(dim=[2, 3]) + 1e-5)   # [1, num_channels]([10, 1, 32, 32])
        #         print('centroid',centroid.shape)#10, 128]
        #         centroids[class_id, :] += centroid.squeeze(0)
        #     else:
        #         centroids[class_id, :] += centroidsLast[class_id, :]
        #     # print('centroids',centroids.shape)
        #     outcentroids.append(centroids)
        #     centroids = torch.zeros(num_classes, num_channels).to(device)
        # Normalize by the number of batches
        # centroids = (centroids / batch_size).permute(1, 0)
        return outcentroids

class GetIntCentriod():
    def __init__(self,opt,dataloader,device=None,Osize=32,n_cluster=1):
        # self.model=model
        self.opt=opt
        self.DEVICE=device
        self.dataloader=dataloader
        self.Osize=Osize
        self.scaler = StandardScaler()
        self.n_cluster=n_cluster

    def getClusterKmean(self,model,iterFlag=False):
        self.model=model
        self.opt.phase = 'val'
        # Load Data
        val_dataload = self.dataloader.load_data()
        iter_val = iter(val_dataload)
        val_data_len = len(val_dataload)
        self.model.eval()
        # netTeacher.eval()
        tbar = tqdm(range(val_data_len - 1))
        # train_data_len = len_source_loader
        # Osize = 32
        class_features_list = [[] for _ in range(6)]
        centerCur=None
        class_countsTT = torch.zeros(6, dtype=torch.int64)
        with torch.no_grad():
            for i in tbar:
                current_means = []
                data_val = next(iter_val)
                ############## Forward Pass ######################
                data_img_val = Variable(data_val['img_full']).to(self.DEVICE)
                label = Variable(data_val['label_full']).to(self.DEVICE)
                assp_feature = self.model.forward(data_img_val,getPFlag=True)
                assp_features = assp_feature
                # print('assp_features', assp_features.shape)
                label = F.interpolate(label.unsqueeze(1).float(), size=(self.Osize, self.Osize),
                                      mode='nearest').long()  # torch.Size([10, 1, 16, 16])
                label=label.squeeze(1)
                # c=assp_features.size(1)
                class_countsTT=class_countsTT+count_elements_per_class(label)
                centerCur = self.GetCenterS3_6c(assp_features, label.long(),centroidsLast=centerCur,device=self.DEVICE)
                for jj in range(self.opt.batch_size):
                    for ii in range(6):
                        if centerCur[jj][ii,: ].sum()!=0:
                            class_features_list[ii].append(centerCur[jj][ii,: ].unsqueeze(0))
                        # print(centerCur[jj][ii,: ].unsqueeze(0).shape)
                if i > 10 and iterFlag:
                    break
            # with open('class_features_list.pkl', 'wb') as file:
            #     pickle.dump(class_features_list, file)
            # with open('class_features_list.pkl', 'rb') as file:
            #     class_features_list = pickle.load(file)
            class_centroids = []
            rate=class_countsTT/class_countsTT.sum()
            print('rate:',rate)
            print('rate:',rate/(rate).min())

            n_components = 64  # PCA降维后的维数
            # pca = PCA(n_components=n_components)
            for class_idx in range(6):
                # 合并当前类别的所有特征
                if class_features_list[class_idx]:
                    features_matrix = torch.cat(class_features_list[class_idx], 0).view(-1, 128).detach().cpu().numpy()
                    # features_matrixMean=np.mean(features_matrix,axis=0)
                    # features_matrixMean=np.expand_dims(features_matrixMean,axis=0)
                    # print('features_matrix',features_matrix.shape,features_matrixMean.shape)
                    # 应用K-means聚类
                    features_matrix = self.scaler.fit_transform(features_matrix)
                    ###PCA
                    # features_matrix = pca.fit_transform(features_matrix)
                    if self.n_cluster==1:
                        kmeans = KMeans(n_clusters=self.n_cluster, random_state=0).fit(features_matrix)
                        centroids = kmeans.cluster_centers_#2,128
                    else:
                        kmeans = KMeans(n_clusters=self.n_cluster - 1, random_state=0).fit(features_matrix)
                        centroids = kmeans.cluster_centers_  # 2,128

                        kmeans2 = KMeans(n_clusters=1, random_state=0).fit(features_matrix)
                        centroids2 = kmeans2.cluster_centers_
                        centroids=np.concatenate([centroids,centroids2],axis=0)
                    ###inv PCA
                    # centroids = pca.inverse_transform(centroids)

                    centroids = self.scaler.inverse_transform(centroids)
                    expand_Center = np.expand_dims(centroids, axis=0)

                    expanded_cluster_centers_ = np.tile(expand_Center, (self.opt.batch_size, 1, 1))  # 在第一个维度复制10次，第二个维度保持不变
                    # print('expanded_cluster_centers_',expanded_cluster_centers_.shape)
                    expanded_cluster_centers_=torch.tensor(expanded_cluster_centers_, requires_grad=False).to(self.DEVICE)
                    class_centroids.append(expanded_cluster_centers_)
                    # print('expanded_cluster_centers_',expanded_cluster_centers_.shape)
            PrototypeScat=torch.cat(class_centroids,dim=1).unsqueeze(-1).unsqueeze(-1)
            return PrototypeScat
    def GetCenterS3_6c(self,features, pseudo_labels, centroidsLast=None, device=None):
        """
        Calculate centroids for multiple classes.
        :param features: Tensor of shape (batch_size, num_channels, height, width)
        :param pseudo_labels: Tensor of shape (batch_size, height, width) with values from 0 to num_classes-1
        :param centroidsLast: Optional; last centroids tensor of shape (num_classes, num_channels)
        :param device: Optional; torch.device on which tensors should be
        :return: Tensor of shape (num_channels, num_classes) representing the centroids
        """
        num_classes = 6
        batch_size, num_channels, height, width = features.shape
        centroids = torch.zeros(num_classes, num_channels).to(device)
        num= torch.zeros(6, 1).to(device)

        outcentroids=[]
        # if centroidsLast is None:
        #     # centroidsLast = torch.zeros(num_classes, num_channels).to(device)
        # else:
        #     # print('centroidsLast',centroidsLast)
        #     for ii in range(len(centroidsLast)):
        #         # print(ii,centroidsLast[-ii].shape)
        #         if centroidsLast[-ii].sum() !=0:
        #             # centroidsLast = centroidsLast[-ii]
        #             break
        #         elif ii == len(centroidsLast)-1:
        #             # centroidsLast = torch.zeros(num_classes, num_channels).to(device)
        #             break
        # Compute centroids for each class
        for batch in range(batch_size):
            batch_features = features[batch, :, :, :].unsqueeze(0)  # [1, num_channels, height, width]
            for class_id in range(num_classes):
                mask = (pseudo_labels[batch] == class_id).float().unsqueeze(0)  # [1, 1, height, width]
                if mask.sum() > 0:
                    centroid = (batch_features * mask).sum(dim=[2, 3]) / mask.sum()  # [1, num_channels]
                    centroids[class_id, :] = centroid.squeeze(0)
                    num[class_id,:]=1
                # else:
                #     centroids[class_id, :] += centroidsLast[class_id, :]
            # print('centroids',centroids.shape)
            outcentroids.append(centroids)
            centroids = torch.zeros(num_classes, num_channels).to(device)
        # batch_features = features[:, :, :, :] # [1, num_channels, height, width]([10, 128, 32, 32])
        # for class_id in range(num_classes):
        #     mask = (pseudo_labels == class_id).float().unsqueeze(1)  # [1, 1, height, width]
        #     if mask.sum() > 0:
        #         print(batch_features.shape , mask.shape)
        #         centroid = (batch_features * mask).sum(dim=[2, 3]) / (mask.sum(dim=[2, 3]) + 1e-5)   # [1, num_channels]([10, 1, 32, 32])
        #         print('centroid',centroid.shape)#10, 128]
        #         centroids[class_id, :] += centroid.squeeze(0)
        #     else:
        #         centroids[class_id, :] += centroidsLast[class_id, :]
        #     # print('centroids',centroids.shape)
        #     outcentroids.append(centroids)
        #     centroids = torch.zeros(num_classes, num_channels).to(device)
        # Normalize by the number of batches
        # centroids = (centroids / batch_size).permute(1, 0)
        return outcentroids

class GetIntCentriodPCA():
    def __init__(self,opt,dataloader,device=None,Osize=32,n_cluster=1):
        # self.model=model
        self.opt=opt
        self.DEVICE=device
        self.dataloader=dataloader
        self.Osize=Osize
        self.scaler = StandardScaler()
        self.n_cluster=n_cluster

    def getClusterKmean(self,model,iterFlag=False):
        self.model=model
        self.opt.phase = 'val'
        val_dataload = self.dataloader.load_data()
        iter_val = iter(val_dataload)
        val_data_len = len(val_dataload)
        self.model.eval()
        # netTeacher.eval()
        tbar = tqdm(range(val_data_len - 1))
        # train_data_len = len_source_loader
        # Osize = 32
        class_features_list = [[] for _ in range(6)]
        centerCur=None
        with torch.no_grad():
            for i in tbar:
                current_means = []
                data_val = next(iter_val)
                ############## Forward Pass ######################
                data_img_val = Variable(data_val['img_full']).to(self.DEVICE)
                label = Variable(data_val['label_full']).to(self.DEVICE)
                assp_feature = self.model.forward(data_img_val,getPFlag=True)
                assp_features = assp_feature
                # print('assp_features', assp_features.shape)
                label = F.interpolate(label.unsqueeze(1).float(), size=(self.Osize, self.Osize),
                                      mode='nearest').long()  # torch.Size([10, 1, 16, 16])
                label=label.squeeze(1)
                # c=assp_features.size(1)
                centerCur = self.GetCenterS3_6c(assp_features, label.long(),centroidsLast=centerCur,device=self.DEVICE)
                for jj in range(self.opt.batch_size):
                    for ii in range(6):
                        if centerCur[jj][ii,: ].sum()!=0:
                            class_features_list[ii].append(centerCur[jj][ii,: ].unsqueeze(0))
                        # print(centerCur[jj][ii,: ].unsqueeze(0).shape)
                if i > 10 and iterFlag:
                    break
            # with open('class_features_list.pkl', 'wb') as file:
            #     pickle.dump(class_features_list, file)
            # with open('class_features_list.pkl', 'rb') as file:
            #     class_features_list = pickle.load(file)
            class_centroids = []
            n_components = self.n_cluster
            # PCA降维后的维数
            # pca = PCA(n_components=n_components)
            for class_idx in range(6):
                # 合并当前类别的所有特征
                if class_features_list[class_idx]:
                    features_matrix = torch.cat(class_features_list[class_idx], 0).view(-1, 128)
                    # Center the data (subtract the mean)
                    mean = torch.mean(features_matrix, dim=0)
                    data_centered = features_matrix - mean
                    # Compute the covariance matrix
                    covariance_matrix = torch.mm(data_centered.t(), data_centered) / (data_centered.size(0) - 1)
                    # Compute eigenvalues and eigenvectors
                    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix, UPLO='U')
                    # Sort eigenvectors by eigenvalues in descending order
                    sorted_indices = torch.argsort(eigenvalues, descending=True)

                    principal_components = eigenvectors[:, sorted_indices][:, :n_components].transpose(1,0)#principal_components torch.Size([128, 1])
                    # print('principal_components',principal_components.shape)
                    principal_components=principal_components.repeat(self.opt.batch_size, 1, 1)
                    # expanded_cluster_centers_ = torch.(principal_components, (self.opt.batch_size, 1, 1))  # 在第一个维度复制10次，第二个维度保持不变
                    class_centroids.append(principal_components)
            PrototypeScat=torch.cat(class_centroids,dim=1).unsqueeze(-1).unsqueeze(-1)
            return PrototypeScat
    def GetCenterS3_6c(self,features, pseudo_labels, centroidsLast=None, device=None):
        """
        Calculate centroids for multiple classes.
        :param features: Tensor of shape (batch_size, num_channels, height, width)
        :param pseudo_labels: Tensor of shape (batch_size, height, width) with values from 0 to num_classes-1
        :param centroidsLast: Optional; last centroids tensor of shape (num_classes, num_channels)
        :param device: Optional; torch.device on which tensors should be
        :return: Tensor of shape (num_channels, num_classes) representing the centroids
        """
        num_classes = 6
        batch_size, num_channels, height, width = features.shape
        centroids = torch.zeros(num_classes, num_channels).to(device)
        num= torch.zeros(6, 1).to(device)

        outcentroids=[]
        # if centroidsLast is None:
        #     # centroidsLast = torch.zeros(num_classes, num_channels).to(device)
        # else:
        #     # print('centroidsLast',centroidsLast)
        #     for ii in range(len(centroidsLast)):
        #         # print(ii,centroidsLast[-ii].shape)
        #         if centroidsLast[-ii].sum() !=0:
        #             # centroidsLast = centroidsLast[-ii]
        #             break
        #         elif ii == len(centroidsLast)-1:
        #             # centroidsLast = torch.zeros(num_classes, num_channels).to(device)
        #             break
        # Compute centroids for each class
        for batch in range(batch_size):
            batch_features = features[batch, :, :, :].unsqueeze(0)  # [1, num_channels, height, width]
            for class_id in range(num_classes):
                mask = (pseudo_labels[batch] == class_id).float().unsqueeze(0)  # [1, 1, height, width]
                if mask.sum() > 0:
                    centroid = (batch_features * mask).sum(dim=[2, 3]) / mask.sum()  # [1, num_channels]
                    centroids[class_id, :] = centroid.squeeze(0)
                    num[class_id,:]=1
                # else:
                #     centroids[class_id, :] += centroidsLast[class_id, :]
            # print('centroids',centroids.shape)
            outcentroids.append(centroids)
            centroids = torch.zeros(num_classes, num_channels).to(device)
        # batch_features = features[:, :, :, :] # [1, num_channels, height, width]([10, 128, 32, 32])
        # for class_id in range(num_classes):
        #     mask = (pseudo_labels == class_id).float().unsqueeze(1)  # [1, 1, height, width]
        #     if mask.sum() > 0:
        #         print(batch_features.shape , mask.shape)
        #         centroid = (batch_features * mask).sum(dim=[2, 3]) / (mask.sum(dim=[2, 3]) + 1e-5)   # [1, num_channels]([10, 1, 32, 32])
        #         print('centroid',centroid.shape)#10, 128]
        #         centroids[class_id, :] += centroid.squeeze(0)
        #     else:
        #         centroids[class_id, :] += centroidsLast[class_id, :]
        #     # print('centroids',centroids.shape)
        #     outcentroids.append(centroids)
        #     centroids = torch.zeros(num_classes, num_channels).to(device)
        # Normalize by the number of batches
        # centroids = (centroids / batch_size).permute(1, 0)
        return outcentroids

class PredictVisualizaion():
    def __init__(self, device=None,Osize=32,savepath=None,showIndex=None,n_cluster=1,CatFlag=False):
        # self.model = model
        self.DEVICE = device
        # self.dataloader = dataloader
        self.Osize=Osize
        self.cm = plt.cm.get_cmap('jet')
        self.savepath=savepath
        self.showIndex=showIndex
        self.n_cluster=n_cluster
        self.CatFlag=CatFlag
    def main(self,model,inputProto=None,epoch=1,dataloader=None,currentOpt=None):
        t_data = dataloader.load_data()
        t_data_len = len(t_data)
        tbar = tqdm(range(t_data_len))
        iter_t = iter(t_data)
        # running_metric.clear()
        model.eval()
        # netTeacher.eval()
        with torch.no_grad():
            for i in tbar:
                data_test = next(iter_t)
                if i not in self.showIndex:
                    continue
                data_img_T = Variable(data_test['img_full']).to(self.DEVICE)
                labelT_ = Variable(data_test['label_full']).to(self.DEVICE)
                # labelT = F.interpolate(labelT_.unsqueeze(1).float(), size=(128, 128), mode='nearest')
                # labelT = labelT.squeeze(1)  # 尺寸变为 [14, 512, 512]
                seg_target_pred, PrototypeT, FeatTT = model.forward(data_img_T, DomainLabel=1,
                                                                         ProtoInput=None)
                #################predict
                target_pred = torch.argmax(seg_target_pred['outUp'].detach(), dim=1)
                #######asspFW+OUtProto
                # print(PrototypeT['GetProto'].shape)
                prototypes_expanded = PrototypeT['query'].expand(-1, -1, -1, self.Osize,
                                                                    self.Osize)  # [B, 6, 128, H, W]
                deep_features_expanded = FeatTT['asspF'].unsqueeze(1)  # [B, 1, 128, H, W]
                # deep_features_expanded = deep_features_expanded.expand(-1, 6, -1, -1, -1)  # [B, 6, 128, H, W]
                distances = torch.norm(prototypes_expanded - deep_features_expanded, dim=2)  # [B, 6, H, W]
                if self.self.CatFlag:
                    target_pred_Fw_Out = torch.argmin(distances.detach(), dim=1)//(self.n_cluster+1)
                else:
                    target_pred_Fw_Out = torch.argmin(distances.detach(), dim=1)
                #######asspFW+OriProto
                prototypes_expanded = inputProto.expand(-1, -1, -1, self.Osize,
                                                           self.Osize)  # [B, 6, 128, H, W]inputProto
                deep_features_expanded = FeatTT['asspF'].unsqueeze(1)  # [B, 1, 128, H, W]
                # deep_features_expanded = deep_features_expanded.expand(-1, 6, -1, -1, -1)  # [B, 6, 128, H, W]
                distances = torch.norm(prototypes_expanded - deep_features_expanded, dim=2)  # [B, 6, H, W]
                # DistP = F.softmax(-distances.detach(), dim=1)  # [B, 6, H, W]
                target_pred_Fw_Ori = torch.argmin(distances.detach(), dim=1)//self.n_cluster
                #######asspF+OUtProto
                prototypes_expanded = PrototypeT['query'].expand(-1, -1, -1, self.Osize,
                                                                    self.Osize)  # [B, 6, 128, H, W]
                deep_features_expanded = FeatTT['asspF'].unsqueeze(1)  # [B, 1, 128, H, W]
                # deep_features_expanded = deep_features_expanded.expand(-1, 6, -1, -1, -1)  # [B, 6, 128, H, W]
                distances = torch.norm(prototypes_expanded - deep_features_expanded, dim=2)  # [B, 6, H, W]
                if self.self.CatFlag:
                    target_pred_F_Out = torch.argmin(distances.detach(), dim=1)//(self.n_cluster+1)
                else:
                    target_pred_F_Out = torch.argmin(distances.detach(), dim=1)
                #######asspFW+OriProto
                prototypes_expanded = inputProto.expand(-1, -1, -1, self.Osize,
                                                           self.Osize)  # [B, 6, 128, H, W]
                deep_features_expanded = FeatTT['asspF'].unsqueeze(1)  # [B, 1, 128, H, W]
                # deep_features_expanded = deep_features_expanded.expand(-1, 6, -1, -1, -1)  # [B, 6, 128, H, W]
                distances = torch.norm(prototypes_expanded - deep_features_expanded, dim=2)  # [B, 6, H, W]
                # DistP = F.softmax(-distances.detach(), dim=1)  # [B, 6, H, W]
                target_pred_F_Ori = torch.argmin(distances.detach(), dim=1)//self.n_cluster

                #########PLT
                labelshow = F.interpolate(labelT_.unsqueeze(1).float(), size=(32, 32), mode='nearest')
                fig1, axs1 = plt.subplots(currentOpt.batch_size * 1, 6, figsize=(10, 2 * currentOpt.batch_size))  # b行2列

                for ii in range(currentOpt.batch_size):
                    path = data_test['label_path'][ii].split('/')[-1]
                    img1 = axs1[ii, 0].imshow(labelshow[ii][0].detach().cpu().numpy(), cmap='gray',
                                              vmin=0, vmax=5)
                    axs1[ii, 0].set_title(path)
                    axs1[ii, 0].axis('off')

                    img2 = axs1[ii, 1].imshow(target_pred_Fw_Out[ii].detach().cpu().numpy(), cmap='gray',
                                              vmin=0, vmax=5)
                    # axs1[ii, 1].set_title('Fw_Out')
                    axs1[ii, 1].set_title('Fw_Out-%d_%d' % (target_pred_Fw_Out[ii].min(), target_pred_Fw_Out[ii].max()))

                    axs1[ii, 1].axis('off')
                    img3 = axs1[ii, 2].imshow(target_pred_Fw_Ori[ii].detach().cpu().numpy(),
                                              cmap='gray',
                                              vmin=0, vmax=5)
                    # axs1[ii, 2].set_title('Fw_Ori')
                    axs1[ii, 2].set_title('Fw_Ori-%d_%d' % (target_pred_Fw_Ori[ii].min(), target_pred_Fw_Ori[ii].max()))
                    axs1[ii, 2].axis('off')

                    img4 = axs1[ii, 3].imshow(target_pred_F_Out[ii].detach().cpu().numpy(),
                                              cmap='gray',
                                              vmin=0, vmax=5)
                    axs1[ii, 3].set_title('F_Out-%d_%d' % (target_pred_F_Out[ii].min(), target_pred_F_Out[ii].max()))
                    axs1[ii, 3].axis('off')

                    img5 = axs1[ii, 4].imshow(target_pred_F_Ori[ii].detach().cpu().numpy(),
                                              cmap='gray',
                                              vmin=0, vmax=5)
                    # axs1[ii, 4].set_title('F_Ori')
                    axs1[ii, 4].set_title('F_Ori-%d_%d' % (target_pred_F_Ori[ii].min(), target_pred_F_Ori[ii].max()))
                    axs1[ii, 4].axis('off')

                    img6 = axs1[ii, 5].imshow(target_pred[ii].detach().cpu().numpy(),
                                              cmap='gray',
                                              vmin=0, vmax=5)
                    # axs1[ii, 4].set_title('F_Ori')
                    axs1[ii, 5].set_title(
                        'target_pred-%d_%d' % (target_pred[ii].min(), target_pred[ii].max()))
                    axs1[ii, 5].axis('off')

                plt.tight_layout()
                plt.savefig(self.savepath + '/' + 'WightPic/savepic/Pesudo%d_%d.png' % (
                    epoch, i))
                plt.clf()
                display.clear_output(wait=True)
                display.display(plt.gcf())

class SimilarityVis():
    def __init__(self, device=None,Osize=32,savepath=None,showIndex=None,n_cluster=1,outFeatSize=32,CatFlag=False,classifierFlag=False):
        # self.model = model
        self.DEVICE = device
        # self.dataloader = dataloader
        self.Osize=Osize
        self.cm = plt.cm.get_cmap('jet')
        self.savepath=savepath
        self.showIndex=showIndex
        self.n_cluster=n_cluster
        self.outFeatSize=outFeatSize
        self.metric=Metric_tool()
        self.CatFlag=CatFlag
        PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
        # 将颜色值从 0-255 范围转换到 0-1 范围
        PALETTE = np.array(PALETTE) / 255.0
        self.cmap = ListedColormap(PALETTE)

    def main(self,model,inputProto=None,epoch=1,dataloader=None,currentOpt=None,classifierFlag=False):
        t_data = dataloader.load_data()
        t_data_len = len(t_data)
        tbar = tqdm(range(t_data_len))
        iter_t = iter(t_data)
        model.eval()
        # netTeacher.eval()

        with torch.no_grad():
            for i in tbar:
                data_test = next(iter_t)
                if i not in self.showIndex:
                    continue
                data_img_T = Variable(data_test['img_full']).to(self.DEVICE)
                labelT_ = Variable(data_test['label_full']).to(self.DEVICE)
                labelT = F.interpolate(labelT_.unsqueeze(1).float(), size=(128, 128), mode='nearest')
                labelT2 = F.interpolate(labelT_.unsqueeze(1).float(), size=(128, 128), mode='bilinear')

                labelshow = F.interpolate(labelT_.unsqueeze(1).float(), size=(32, 32), mode='bilinear')
                # labelshow=labelT_.unsqueeze(1)
                # labelT = labelT.squeeze(1)  # 尺寸变为 [14, 512, 512]
                seg_target_predG, ProtoG, FeatTTG = model.forward(data_img_T, DomainLabel=0, ProtoInput=inputProto)
                if self.CatFlag:
                    seg_target_predC, ProtoC, FeatTTC = model.forward(data_img_T, DomainLabel=0, ProtoInput=inputProto)
                else:
                    seg_target_predC, ProtoC, FeatTTC = model.forward(data_img_T, DomainLabel=0, ProtoInput=inputProto)
                # TCELoss = cross_entropy(seg_target_pred['outUp'], labelT.long())
                # update metric
                # TScore = {'LossT': TCELoss.item(), 'TCET': TCELoss.item()}
                # target_pred = torch.argmax(seg_target_pred['outUp'].detach(), dim=1)
                Probability = F.softmax(seg_target_predC['outUp'], dim=1)

                Probabilitymax, _ = torch.max(Probability, dim=1)
                fig1, axs1 = plt.subplots(currentOpt.batch_size * 3, 2 + 6, figsize=(11, 3 * currentOpt.batch_size))  # b行2列
                title_font = {
                    # 'family': 'serif',  # 字体家族，如 'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
                    # 'color': 'darkred',  # 字体颜色
                    # 'weight': 'normal',  # 字体粗细，如 'normal', 'bold', 'heavy', 'light', 'ultrabold', 'ultralight'
                    'size': 6,  # 字体大小
                }
                title_font2 = {
                    'size': 6,  # 字体大小
                }
                '''
                ###########################Glabal Original Similarity
                prototypes_expanded = inputProto.expand(-1, -1, -1, self.outFeatSize, self.outFeatSize)  # [B, 6, 128, H, W]
                deep_features_expanded = FeatTTG['asspF'].unsqueeze(1)  # [B, 1, 128, H, W]
                # Normalize along the channel dimension
                prototypes_expanded = F.normalize(prototypes_expanded, p=2,dim=2)
                deep_features_expanded = F.normalize(deep_features_expanded, p=2, dim=2)
                # # # distances = torch.norm(prototypes_expanded - deep_features_expanded, p=2, dim=2)  # [B, 6, H, W]
                distances = torch.norm(deep_features_expanded - prototypes_expanded, p=2, dim=2)  # [B, 6, H, W]
                distancesSoftmax = F.softmax(1/distances, dim=1)
                distancesSoftmax = distancesSoftmax.view(distances.size(0), Probability.size(1), self.n_cluster,
                                                         distances.size(2), distances.size(3)).sum(dim=2)
                G_pred = torch.argmax(distancesSoftmax.detach(), dim=1)#//self.n_cluster
                distancesSoftmaxMax, _ = torch.max(distancesSoftmax, dim=1)
                for ii in range(currentOpt.batch_size):
                    img1 = axs1[ii*3 , 0].imshow(G_pred[ii].detach().cpu().numpy(), cmap='gray',
                                                  vmin=0, vmax=5)
                    # axs1[ii*3 , 0].set_title('Max W:%.2f-%.2f' % (
                    #     FeatTT['Weight'][-1][ii].min(), FeatTT['Weight'][-1][ii].max()),fontdict=title_font)
                    acc=self.metric.accuracy(G_pred[ii].unsqueeze(0),labelshow[ii]).cpu().numpy()
                    miou=self.metric.mean_iou(G_pred[ii].unsqueeze(0),labelshow[ii],num_classes=6).cpu().numpy()
                    mf1=self.metric.mean_f1(G_pred[ii].unsqueeze(0),labelshow[ii],num_classes=6).cpu().numpy()
                    axs1[ii * 3, 0].set_title('acc:%.2f-miou:%.2f-mf1:%.2f' % (acc,miou,mf1), fontdict=title_font)
                    axs1[ii*3 , 0].axis('off')
                    # cbar1 = fig1.colorbar(img1, ax=axs1[ii*3, 0], extend='both', fraction=0.046, pad=0.04)
                    # cbar1.set_ticks([0, 0.25, 0.5, 0.75, 1])
                    # cbar1.set_label('Probability')

                    img2 = axs1[ii*3 , 1].imshow(distancesSoftmaxMax[ii].detach().cpu().numpy(), cmap='jet',
                                                  vmin=0, vmax=1)
                    # axs[ii, 1].set_title('Mean Weight')
                    axs1[ii*3, 1].set_title('G-MaxDist: %.2f-%.2f' % (
                        distancesSoftmaxMax[ii].min(), distancesSoftmaxMax[ii].max()),fontdict=title_font)
                    axs1[ii*3, 1].axis('off')
                    cbar2 = fig1.colorbar(img2, ax=axs1[ii *3, 1], extend='both', fraction=0.046, pad=0.04)
                    cbar2.set_ticks([0, 0.25, 0.5, 0.75, 1])
                    cbar2.ax.tick_params(labelsize=6)  # 设置刻度标签的字体大小为10
                    # cbar2.set_label('Probability')
                    for jj in range(2, 8):
                        # print(FeatTT['Weight'][-3].shape,'aaaaaaa')
                        img3 = axs1[ii*3, jj].imshow(distancesSoftmax[ii, jj - 2].detach().cpu().numpy(),
                                                       cmap='jet', vmin=0, vmax=1)
                        axs1[ii*3, jj].set_title('G-Dist:%.2f-%.2f' % (
                            distancesSoftmax[ii, jj - 2].min(), distancesSoftmax[ii, jj - 2].max()),fontdict=title_font)
                        axs1[ii*3, jj].axis('off')
                        cbar3 = fig1.colorbar(img3, ax=axs1[ii*3, jj], extend='both', fraction=0.046, pad=0.04)
                        cbar3.set_ticks([0, 0.25, 0.5, 0.75, 1])
                        cbar3.ax.tick_params(labelsize=6)  # 设置刻度标签的字体大小为10
                '''
                ###########################Glabal Original Similarity
                prototype_features_transposed = ProtoG['query'].squeeze(-1).squeeze(-1)  # [B, 128, 6]
                deep_features_reshaped = FeatTTG['asspF'].reshape(FeatTTG['asspF'].size(0), FeatTTG['asspF'].size(1), -1)  # [B, 128, H*W]
                # print('prototype_features_transposed',prototype_features_transposed.shape,deep_features_reshaped.shape)
                prototype_features_transposed = F.normalize(prototype_features_transposed, p=2,
                                                  dim=2)  # Normalize along the channel dimension
                deep_features_reshaped = F.normalize(deep_features_reshaped, p=2,
                                                            dim=2)  # Normalize along the channel dimension
                similarity = torch.matmul(prototype_features_transposed, deep_features_reshaped)  # [B, 6, H*W]#([10, 6, 1024])

                similarity_reshaped = similarity.reshape(FeatTTG['asspF'].size(0), ProtoG['query'].size(1),
                                                         FeatTTG['asspF'].size(2), FeatTTG['asspF'].size(3))  # [B, 6, H, W]

                G_pred = torch.argmax(similarity_reshaped.detach(), dim=1) // (self.n_cluster)  #############!!!!!!!!!!!!!!!!!!!!!!
                distancesSoftmax = F.softmax(similarity_reshaped, dim=1)
                distancesSoftmax = distancesSoftmax.view(distancesSoftmax.size(0),
                                                         distancesSoftmax.size(1) // (self.n_cluster),
                                                         (self.n_cluster),
                                                         distancesSoftmax.size(2), distancesSoftmax.size(3))
                distancesSoftmax = distancesSoftmax.sum(dim=2)
                distancesSoftmaxMax, _ = torch.max(distancesSoftmax, dim=1)

                for ii in range(currentOpt.batch_size):
                    img1 = axs1[ii * 3, 0].imshow(G_pred[ii].detach().cpu().numpy(), cmap=self.cmap,
                                                  vmin=0, vmax=5)


                    # axs1[ii*3 , 0].set_title('Max W:%.2f-%.2f' % (
                    #     FeatTT['Weight'][-1][ii].min(), FeatTT['Weight'][-1][ii].max()),fontdict=title_font)
                    acc = self.metric.accuracy(G_pred[ii].unsqueeze(0), labelshow[ii]).cpu().numpy()
                    miou = self.metric.mean_iou(G_pred[ii].unsqueeze(0), labelshow[ii], num_classes=6).cpu().numpy()
                    mf1 = self.metric.mean_f1(G_pred[ii].unsqueeze(0), labelshow[ii], num_classes=6).cpu().numpy()
                    axs1[ii * 3, 0].set_title('acc:%.2f-miou:%.2f-mf1:%.2f' % (acc, miou, mf1), fontdict=title_font)
                    axs1[ii * 3, 0].axis('off')
                    # cbar1 = fig1.colorbar(img1, ax=axs1[ii*3, 0], extend='both', fraction=0.046, pad=0.04)
                    # cbar1.set_ticks([0, 0.25, 0.5, 0.75, 1])
                    # cbar1.set_label('Probability')
                    img2 = axs1[ii * 3, 1].imshow(distancesSoftmaxMax[ii].detach().cpu().numpy(), cmap='jet')
                    # axs[ii, 1].set_title('Mean Weight')
                    axs1[ii * 3, 1].set_title('G-MaxDist: %.2f-%.2f' % (
                        distancesSoftmaxMax[ii].min(), distancesSoftmaxMax[ii].max()), fontdict=title_font)
                    axs1[ii * 3, 1].axis('off')
                    cbar2 = fig1.colorbar(img2, ax=axs1[ii * 3, 1], extend='both', fraction=0.046, pad=0.04)
                    cbar2.set_ticks([0, 0.25, 0.5, 0.75, 1])
                    cbar2.ax.tick_params(labelsize=6)  # 设置刻度标签的字体大小为10
                    # cbar2.set_label('Probability')
                    for jj in range(2, 8):
                        # print(FeatTT['Weight'][-3].shape,'aaaaaaa')
                        img3 = axs1[ii * 3, jj].imshow(distancesSoftmax[ii, jj - 2].detach().cpu().numpy(),
                                                       cmap='jet')
                        axs1[ii * 3, jj].set_title('G-Dist:%.2f-%.2f' % (
                            distancesSoftmax[ii, jj - 2].min(), distancesSoftmax[ii, jj - 2].max()),
                                                   fontdict=title_font)
                        axs1[ii * 3, jj].axis('off')
                        cbar3 = fig1.colorbar(img3, ax=axs1[ii * 3, jj], extend='both', fraction=0.046, pad=0.04)
                        # cbar3.set_ticks([0, 0.25, 0.5, 0.75, 1])
                        cbar3.ax.tick_params(labelsize=6)  # 设置刻度标签的字体大小为10
                #########Similarity/Current
                # prototypes_expanded = Proto['GetProto'].expand(-1, -1, -1, self.outFeatSize,
                #                                         self.outFeatSize)  # [B, 6, 128, H, W]
                # deep_features_expanded = FeatTT['asspF'].unsqueeze(1)  # [B, 1, 128, H, W]
                # # Normalize along the channel dimension
                # prototypes_expanded = F.normalize(prototypes_expanded, p=2, dim=2)
                # deep_features_expanded = F.normalize(deep_features_expanded, p=2, dim=2)
                # # # # distances = torch.norm(prototypes_expanded - deep_features_expanded, p=2, dim=2)  # [B, 6, H, W]
                # distances = torch.norm(deep_features_expanded - prototypes_expanded, p=2, dim=2)  # [B, 6, H, W]
                # distancesSoftmax = F.softmax(distances, dim=1)
                # C_pred = torch.argmin(distancesSoftmax.detach(), dim=1)

                if self.CatFlag:
                    C_pred = torch.argmax(FeatTTC['cat'].detach(),
                                          dim=1) // (self.n_cluster + 1)  #############!!!!!!!!!!!!!!!!!!!!!!
                    distancesSoftmax = F.softmax(FeatTTC['cat'], dim=1)
                    distancesSoftmax = distancesSoftmax.view(distancesSoftmax.size(0),
                                                             distancesSoftmax.size(1) // (self.n_cluster + 1),
                                                             (self.n_cluster + 1),
                                                             distancesSoftmax.size(2), distancesSoftmax.size(3))
                    distancesSoftmax = distancesSoftmax.sum(dim=2)

                else:
                    if classifierFlag:
                        C_pred = torch.argmax(FeatTTC['cOut'].detach(), dim=1)
                        distancesSoftmax = F.softmax(FeatTTC['cOut'], dim=1)

                        #############!!!!!!!!!!!!!!!!!!!!!!
                    else:
                        C_pred = torch.argmax(FeatTTC['cat'].detach(),  dim=1) // self.n_cluster
                        distancesSoftmax = F.softmax(FeatTTC['cat'], dim=1)
                        distancesSoftmax = distancesSoftmax.view(distancesSoftmax.size(0),
                                                                 distancesSoftmax.size(1) // (self.n_cluster),
                                                                 (self.n_cluster),
                                                                 distancesSoftmax.size(2), distancesSoftmax.size(3))
                        distancesSoftmax = distancesSoftmax.sum(dim=2)
                distancesSoftmaxMax, _ = torch.max(distancesSoftmax, dim=1)
                # if self.CatFlag:
                #     C_pred = torch.argmax(FeatTTC['cat'].detach(),
                #                           dim=1) // (self.n_cluster +1) #############!!!!!!!!!!!!!!!!!!!!!!
                # else:
                #     C_pred= torch.argmax(FeatTTC['cat'].detach(), dim=1)//self.n_cluster#############!!!!!!!!!!!!!!!!!!!!!!
                for ii in range(currentOpt.batch_size):
                    # print('FeatTT[-1]',FeatTT[-1].shape)
                    acc = self.metric.accuracy(C_pred[ii].unsqueeze(0), labelshow[ii]).cpu().numpy()
                    miou = self.metric.mean_iou(C_pred[ii].unsqueeze(0), labelshow[ii], num_classes=6).cpu().numpy()
                    mf1 = self.metric.mean_f1(C_pred[ii].unsqueeze(0), labelshow[ii], num_classes=6).cpu().numpy()
                    # img1 = axs1[ii * 3+1, 0].imshow(C_pred[ii].detach().cpu().numpy(), cmap='jet',
                    #                               vmin=0, vmax=1)

                    # img1 = axs1[ii * 3 + 1, 0].imshow(labelT_[ii].detach().cpu().numpy(), cmap=self.cmap,
                    #                                   vmin=0, vmax=5)

                    img1 = axs1[ii * 3 + 1, 0].imshow(C_pred[ii].detach().cpu().numpy(), cmap=self.cmap,
                                                      vmin=0, vmax=5)
                    axs1[ii * 3+1, 0].set_title('acc:%.2f-miou:%.2f-mf1:%.2f' % (acc, miou, mf1), fontdict=title_font)
                    # axs1[ii * 3+1, 0].set_title('Max Weight:%.2f-%.2f' % (
                    #     FeatTT['Weight'][-1][ii].min(), FeatTT['Weight'][-1][ii].max()),fontdict=title_font)
                    axs1[ii * 3+1, 0].axis('off')
                    # cbar1 = fig1.colorbar(img1, ax=axs1[ii * 3+1, 0], extend='both', fraction=0.046, pad=0.04)
                    # cbar1.set_ticks([0, 0.25, 0.5, 0.75, 1])
                    # cbar1.set_label('Probability')

                    img2 = axs1[ii * 3+1, 1].imshow(distancesSoftmaxMax[ii].detach().cpu().numpy(), cmap='jet')
                    # axs[ii, 1].set_title('Mean Weight')
                    axs1[ii * 3+1, 1].set_title('Max Sim: %.2f-%.2f' % (
                        distancesSoftmaxMax[ii].min(), distancesSoftmaxMax[ii].max()),fontdict=title_font)
                    axs1[ii * 3+1, 1].axis('off')
                    cbar2 = fig1.colorbar(img2, ax=axs1[ii * 3+1, 1], extend='both', fraction=0.046, pad=0.04)
                    # cbar2.set_ticks([0, 0.25, 0.5, 0.75, 1])
                    cbar2.ax.tick_params(labelsize=6)  # 设置刻度标签的字体大小为10

                    # cbar2.set_label('Probability')
                    for jj in range(2, 8):
                        img3 = axs1[ii * 3+1, jj].imshow(distancesSoftmax[ii, jj - 2].detach().cpu().numpy(),
                                                       cmap='jet')
                        # vmin=0, vmax=1)
                        axs1[ii * 3+1, jj].set_title('Sim:%.2f-%.2f' % (
                            distancesSoftmax[ii, jj - 2].min(), distancesSoftmax[ii, jj - 2].max()),fontdict=title_font)
                        axs1[ii * 3+1, jj].axis('off')
                        cbar3 = fig1.colorbar(img3, ax=axs1[ii * 3+1, jj], extend='both', fraction=0.046, pad=0.04)
                        # cbar3.set_ticks([0, 0.25, 0.5, 0.75, 1])
                        # cbar3.set_label('Probability')
                        cbar3.ax.tick_params(labelsize=6)  # 设置刻度标签的字体大小为10
                ################Predict
                Prob_pred = torch.argmax(Probability.detach(), dim=1)

                for ii in range(currentOpt.batch_size):
                    acc = self.metric.accuracy(Prob_pred[ii].unsqueeze(0), labelT[ii]).cpu().numpy()
                    miou = self.metric.mean_iou(Prob_pred[ii].unsqueeze(0), labelT[ii], num_classes=6).cpu().numpy()
                    mf1 = self.metric.mean_f1(Prob_pred[ii].unsqueeze(0), labelT[ii], num_classes=6).cpu().numpy()

                    path = data_test['label_path'][ii].split('/')[-1]
                    outimg=labelT_[ii].detach().cpu().numpy()
                    cv2.imwrite('./outimg/label/'+path,outimg.astype(np.uint8))
                    outimg2=labelT[ii][0].detach().cpu().numpy()
                    cv2.imwrite('./outimg/label2/'+path,outimg2.astype(np.uint8))

                    img1 = axs1[ii * 3 + 2, 0].imshow(labelT[ii][0].detach().cpu().numpy().astype(np.uint8), cmap=self.cmap, vmin=0,
                                                      vmax=5)
                    axs1[ii * 3 + 2, 0].set_title(path,fontdict=title_font2)
                    axs1[ii * 3 + 2, 0].axis('off')

                    img2 = axs1[ii * 3 + 2, 1].imshow(Prob_pred[ii].detach().cpu().numpy(), cmap=self.cmap,
                                                      vmin=0, vmax=5)
                    # img2 = axs1[ii * 3 + 2, 1].imshow(labelT2[ii][0].detach().cpu().numpy().astype(np.uint8), cmap=self.cmap,
                    #                                   vmin=0, vmax=5)
                    # axs[ii, 1].set_title('Mean Weight')
                    axs1[ii * 3 + 2, 1].set_title('acc:%.2f-miou:%.2f-mf1:%.2f' % (acc, miou, mf1), fontdict=title_font)
                    # axs1[ii * 3 + 2, 1].set_title(
                    #     'Max Proba:%.2f-%.2f' % (Probabilitymax[ii].min(), Probabilitymax[ii].max()),fontdict=title_font)
                    axs1[ii * 3 + 2, 1].axis('off')
                    # cbar2 = fig1.colorbar(img2, ax=axs1[ii * 3 + 2, 1], extend='both', fraction=0.046, pad=0.04)
                    # cbar2.set_ticks([0, 0.25, 0.5, 0.75, 1])
                    # cbar2.set_label('Probability')
                    cbar2.ax.tick_params(labelsize=6)  # 设置刻度标签的字体大小为10

                    for jj in range(2, 8):
                        img3 = axs1[ii * 3 + 2, jj].imshow(Probability[ii, jj - 2].detach().cpu().numpy(),
                                                           cmap='jet',
                                                           vmin=0, vmax=1)
                        axs1[ii * 3 + 2, jj].set_title('Proba:%.2f-%.2f' % (
                            Probability[ii, jj - 2].min(), Probability[ii, jj - 2].max()),fontdict=title_font)
                        axs1[ii * 3 + 2, jj].axis('off')
                        cbar3 = fig1.colorbar(img3, ax=axs1[ii * 3 + 2, jj], extend='both', fraction=0.046, pad=0.04)
                        cbar3.set_ticks([0, 0.25, 0.5, 0.75, 1])
                        # cbar3.set_label('Probability')
                        cbar3.ax.tick_params(labelsize=6)  # 设置刻度标签的字体大小为10

                plt.tight_layout()
                plt.savefig(self.savepath + 'WightPic/savepic/image%d_%d.png' % (epoch, i))
                plt.clf()
                display.clear_output(wait=True)
                display.display(plt.gcf())
class SimilarityVisDP():
    def __init__(self, device=None,Osize=32,savepath=None,showIndex=None,n_cluster=1,outFeatSize=32,CatFlag=False,classifierFlag=False):
        # self.model = model
        self.DEVICE = device
        # self.dataloader = dataloader
        self.Osize=Osize
        self.cm = plt.cm.get_cmap('jet')
        self.savepath=savepath
        self.showIndex=showIndex
        self.n_cluster=n_cluster
        self.outFeatSize=outFeatSize
        self.metric=Metric_tool()
        self.CatFlag=CatFlag
    def main(self,model,inputProto=None,epoch=1,dataloader=None,currentOpt=None,classifierFlag=False):
        t_data = dataloader.load_data()
        t_data_len = len(t_data)
        tbar = tqdm(range(t_data_len))
        iter_t = iter(t_data)
        model.eval()
        # netTeacher.eval()

        with torch.no_grad():
            for i in tbar:
                data_test = next(iter_t)
                if i not in self.showIndex:
                    continue
                data_img_T = Variable(data_test['img_full']).to(self.DEVICE)
                labelT_ = Variable(data_test['label_full']).to(self.DEVICE)
                labelT = F.interpolate(labelT_.unsqueeze(1).float(), size=(128, 128), mode='nearest')
                labelshow = F.interpolate(labelT_.unsqueeze(1).float(), size=(32, 32), mode='nearest')

                # labelT = labelT.squeeze(1)  # 尺寸变为 [14, 512, 512]
                seg_target_predG, ProtoG, FeatTTG = model.forward(data_img_T, DomainLabel=0, ProtoInput=inputProto)
                if self.CatFlag:
                    seg_target_predC, ProtoC, FeatTTC = model.forward(data_img_T, DomainLabel=0, ProtoInput=inputProto)
                else:
                    seg_target_predC, ProtoC, FeatTTC = model.forward(data_img_T, DomainLabel=0, ProtoInput=inputProto)
                # TCELoss = cross_entropy(seg_target_pred['outUp'], labelT.long())
                # update metric
                # TScore = {'LossT': TCELoss.item(), 'TCET': TCELoss.item()}
                # target_pred = torch.argmax(seg_target_pred['outUp'].detach(), dim=1)
                Probability = F.softmax(seg_target_predC['outUp'], dim=1)

                Probabilitymax, _ = torch.max(Probability, dim=1)
                fig1, axs1 = plt.subplots(currentOpt.batch_size * 3, 2 + 6, figsize=(11, 3 * currentOpt.batch_size))  # b行2列
                title_font = {
                    # 'family': 'serif',  # 字体家族，如 'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
                    # 'color': 'darkred',  # 字体颜色
                    # 'weight': 'normal',  # 字体粗细，如 'normal', 'bold', 'heavy', 'light', 'ultrabold', 'ultralight'
                    'size': 6,  # 字体大小
                }
                title_font2 = {
                    'size': 6,  # 字体大小
                }
                '''
                ###########################Glabal Original Similarity
                prototypes_expanded = inputProto.expand(-1, -1, -1, self.outFeatSize, self.outFeatSize)  # [B, 6, 128, H, W]
                deep_features_expanded = FeatTTG['asspF'].unsqueeze(1)  # [B, 1, 128, H, W]
                # Normalize along the channel dimension
                prototypes_expanded = F.normalize(prototypes_expanded, p=2,dim=2)
                deep_features_expanded = F.normalize(deep_features_expanded, p=2, dim=2)
                # # # distances = torch.norm(prototypes_expanded - deep_features_expanded, p=2, dim=2)  # [B, 6, H, W]
                distances = torch.norm(deep_features_expanded - prototypes_expanded, p=2, dim=2)  # [B, 6, H, W]
                distancesSoftmax = F.softmax(1/distances, dim=1)
                distancesSoftmax = distancesSoftmax.view(distances.size(0), Probability.size(1), self.n_cluster,
                                                         distances.size(2), distances.size(3)).sum(dim=2)
                G_pred = torch.argmax(distancesSoftmax.detach(), dim=1)#//self.n_cluster
                distancesSoftmaxMax, _ = torch.max(distancesSoftmax, dim=1)
                for ii in range(currentOpt.batch_size):
                    img1 = axs1[ii*3 , 0].imshow(G_pred[ii].detach().cpu().numpy(), cmap='gray',
                                                  vmin=0, vmax=5)
                    # axs1[ii*3 , 0].set_title('Max W:%.2f-%.2f' % (
                    #     FeatTT['Weight'][-1][ii].min(), FeatTT['Weight'][-1][ii].max()),fontdict=title_font)
                    acc=self.metric.accuracy(G_pred[ii].unsqueeze(0),labelshow[ii]).cpu().numpy()
                    miou=self.metric.mean_iou(G_pred[ii].unsqueeze(0),labelshow[ii],num_classes=6).cpu().numpy()
                    mf1=self.metric.mean_f1(G_pred[ii].unsqueeze(0),labelshow[ii],num_classes=6).cpu().numpy()
                    axs1[ii * 3, 0].set_title('acc:%.2f-miou:%.2f-mf1:%.2f' % (acc,miou,mf1), fontdict=title_font)
                    axs1[ii*3 , 0].axis('off')
                    # cbar1 = fig1.colorbar(img1, ax=axs1[ii*3, 0], extend='both', fraction=0.046, pad=0.04)
                    # cbar1.set_ticks([0, 0.25, 0.5, 0.75, 1])
                    # cbar1.set_label('Probability')

                    img2 = axs1[ii*3 , 1].imshow(distancesSoftmaxMax[ii].detach().cpu().numpy(), cmap='jet',
                                                  vmin=0, vmax=1)
                    # axs[ii, 1].set_title('Mean Weight')
                    axs1[ii*3, 1].set_title('G-MaxDist: %.2f-%.2f' % (
                        distancesSoftmaxMax[ii].min(), distancesSoftmaxMax[ii].max()),fontdict=title_font)
                    axs1[ii*3, 1].axis('off')
                    cbar2 = fig1.colorbar(img2, ax=axs1[ii *3, 1], extend='both', fraction=0.046, pad=0.04)
                    cbar2.set_ticks([0, 0.25, 0.5, 0.75, 1])
                    cbar2.ax.tick_params(labelsize=6)  # 设置刻度标签的字体大小为10
                    # cbar2.set_label('Probability')
                    for jj in range(2, 8):
                        # print(FeatTT['Weight'][-3].shape,'aaaaaaa')
                        img3 = axs1[ii*3, jj].imshow(distancesSoftmax[ii, jj - 2].detach().cpu().numpy(),
                                                       cmap='jet', vmin=0, vmax=1)
                        axs1[ii*3, jj].set_title('G-Dist:%.2f-%.2f' % (
                            distancesSoftmax[ii, jj - 2].min(), distancesSoftmax[ii, jj - 2].max()),fontdict=title_font)
                        axs1[ii*3, jj].axis('off')
                        cbar3 = fig1.colorbar(img3, ax=axs1[ii*3, jj], extend='both', fraction=0.046, pad=0.04)
                        cbar3.set_ticks([0, 0.25, 0.5, 0.75, 1])
                        cbar3.ax.tick_params(labelsize=6)  # 设置刻度标签的字体大小为10
                '''
                ###########################Glabal Original Similarity
                prototype_features_transposed = ProtoG['query'].squeeze(-1).squeeze(-1)  # [B, 128, 6]
                deep_features_reshaped = FeatTTG['asspF'].reshape(FeatTTG['asspF'].size(0), FeatTTG['asspF'].size(1), -1)  # [B, 128, H*W]
                # print('prototype_features_transposed',prototype_features_transposed.shape,deep_features_reshaped.shape)
                prototype_features_transposed = F.normalize(prototype_features_transposed, p=2,
                                                  dim=2)  # Normalize along the channel dimension
                deep_features_reshaped = F.normalize(deep_features_reshaped, p=2,
                                                            dim=2)  # Normalize along the channel dimension
                similarity = torch.matmul(prototype_features_transposed, deep_features_reshaped)  # [B, 6, H*W]#([10, 6, 1024])

                similarity_reshaped = similarity.reshape(FeatTTG['asspF'].size(0), ProtoG['query'].size(1),
                                                         FeatTTG['asspF'].size(2), FeatTTG['asspF'].size(3))  # [B, 6, H, W]

                G_pred = torch.argmax(similarity_reshaped.detach(), dim=1)   #############!!!!!!!!!!!!!!!!!!!!!!
                G_pred=classify_out(G_pred,self.n_cluster)
                distancesSoftmax = F.softmax(similarity_reshaped, dim=1)
                distancesSoftmax=multiFeatClaSum(distancesSoftmax,self.n_cluster,M='max')

                # distancesSoftmax=distancesSoftmax(distancesSoftmax,self.n_cluster)
                # distancesSoftmax = distancesSoftmax.view(distancesSoftmax.size(0),
                #                                          distancesSoftmax.size(1) // (self.n_cluster),
                #                                          (self.n_cluster),
                #                                          distancesSoftmax.size(2), distancesSoftmax.size(3))
                # distancesSoftmax = distancesSoftmax.sum(dim=2)
                distancesSoftmaxMax, _ = torch.max(distancesSoftmax, dim=1)
                # print('distancesSoftmaxMax',distancesSoftmaxMax.shape)
                for ii in range(currentOpt.batch_size):
                    img1 = axs1[ii * 3, 0].imshow(G_pred[ii].detach().cpu().numpy(), cmap='gray',
                                                  vmin=0, vmax=5)
                    # axs1[ii*3 , 0].set_title('Max W:%.2f-%.2f' % (
                    #     FeatTT['Weight'][-1][ii].min(), FeatTT['Weight'][-1][ii].max()),fontdict=title_font)
                    acc = self.metric.accuracy(G_pred[ii].unsqueeze(0), labelshow[ii]).cpu().numpy()
                    miou = self.metric.mean_iou(G_pred[ii].unsqueeze(0), labelshow[ii], num_classes=6).cpu().numpy()
                    mf1 = self.metric.mean_f1(G_pred[ii].unsqueeze(0), labelshow[ii], num_classes=6).cpu().numpy()
                    axs1[ii * 3, 0].set_title('acc:%.2f-miou:%.2f-mf1:%.2f' % (acc, miou, mf1), fontdict=title_font)
                    axs1[ii * 3, 0].axis('off')
                    # cbar1 = fig1.colorbar(img1, ax=axs1[ii*3, 0], extend='both', fraction=0.046, pad=0.04)
                    # cbar1.set_ticks([0, 0.25, 0.5, 0.75, 1])
                    # cbar1.set_label('Probability')
                    img2 = axs1[ii * 3, 1].imshow(distancesSoftmaxMax[ii].detach().cpu().numpy(), cmap='jet')
                    # axs[ii, 1].set_title('Mean Weight')
                    axs1[ii * 3, 1].set_title('G-MaxDist: %.2f-%.2f' % (
                        distancesSoftmaxMax[ii].min(), distancesSoftmaxMax[ii].max()), fontdict=title_font)
                    axs1[ii * 3, 1].axis('off')
                    cbar2 = fig1.colorbar(img2, ax=axs1[ii * 3, 1], extend='both', fraction=0.046, pad=0.04)
                    cbar2.set_ticks([0, 0.25, 0.5, 0.75, 1])
                    cbar2.ax.tick_params(labelsize=6)  # 设置刻度标签的字体大小为10
                    # cbar2.set_label('Probability')
                    for jj in range(2, 8):
                        # print(FeatTT['Weight'][-3].shape,'aaaaaaa')
                        img3 = axs1[ii * 3, jj].imshow(distancesSoftmax[ii, jj - 2].detach().cpu().numpy(),
                                                       cmap='jet')
                        axs1[ii * 3, jj].set_title('G-Dist:%.2f-%.2f' % (
                            distancesSoftmax[ii, jj - 2].min(), distancesSoftmax[ii, jj - 2].max()),
                                                   fontdict=title_font)
                        axs1[ii * 3, jj].axis('off')
                        cbar3 = fig1.colorbar(img3, ax=axs1[ii * 3, jj], extend='both', fraction=0.046, pad=0.04)
                        # cbar3.set_ticks([0, 0.25, 0.5, 0.75, 1])
                        cbar3.ax.tick_params(labelsize=6)  # 设置刻度标签的字体大小为10
                #########Similarity/Current
                # prototypes_expanded = Proto['GetProto'].expand(-1, -1, -1, self.outFeatSize,
                #                                         self.outFeatSize)  # [B, 6, 128, H, W]
                # deep_features_expanded = FeatTT['asspF'].unsqueeze(1)  # [B, 1, 128, H, W]
                # # Normalize along the channel dimension
                # prototypes_expanded = F.normalize(prototypes_expanded, p=2, dim=2)
                # deep_features_expanded = F.normalize(deep_features_expanded, p=2, dim=2)
                # # # # distances = torch.norm(prototypes_expanded - deep_features_expanded, p=2, dim=2)  # [B, 6, H, W]
                # distances = torch.norm(deep_features_expanded - prototypes_expanded, p=2, dim=2)  # [B, 6, H, W]
                # distancesSoftmax = F.softmax(distances, dim=1)
                # C_pred = torch.argmin(distancesSoftmax.detach(), dim=1)

                if self.CatFlag:
                    C_pred = torch.argmax(FeatTTC['cat'].detach(),
                                          dim=1) // (self.n_cluster + 1)  #############!!!!!!!!!!!!!!!!!!!!!!
                    distancesSoftmax = F.softmax(FeatTTC['cat'], dim=1)
                    distancesSoftmax = distancesSoftmax.view(distancesSoftmax.size(0),
                                                             distancesSoftmax.size(1) // (self.n_cluster + 1),
                                                             (self.n_cluster + 1),
                                                             distancesSoftmax.size(2), distancesSoftmax.size(3))
                    distancesSoftmax = distancesSoftmax.sum(dim=2)
                else:
                    if classifierFlag:
                        C_pred = torch.argmax(FeatTTC['cOut'].detach(), dim=1)

                        distancesSoftmax = F.softmax(FeatTTC['cOut'], dim=1)
                        #############!!!!!!!!!!!!!!!!!!!!!!
                    else:

                        C_pred = torch.argmax(FeatTTC['cat'].detach(),  dim=1)
                        C_pred=classify_out(C_pred,self.n_cluster)
                        # print('C_pred',C_pred.shape)
                        distancesSoftmax = F.softmax(FeatTTC['cat'], dim=1)
                        distancesSoftmax=multiFeatClaSum(distancesSoftmax,self.n_cluster,M='max')

                        # distancesSoftmax = distancesSoftmax.view(distancesSoftmax.size(0),
                        #                                          distancesSoftmax.size(1) // (self.n_cluster),
                        #                                          (self.n_cluster),
                        #                                          distancesSoftmax.size(2), distancesSoftmax.size(3))
                        # distancesSoftmax = distancesSoftmax.sum(dim=2)
                distancesSoftmaxMax, _ = torch.max(distancesSoftmax, dim=1)
                # if self.CatFlag:
                #     C_pred = torch.argmax(FeatTTC['cat'].detach(),
                #                           dim=1) // (self.n_cluster +1) #############!!!!!!!!!!!!!!!!!!!!!!
                # else:
                #     C_pred= torch.argmax(FeatTTC['cat'].detach(), dim=1)//self.n_cluster#############!!!!!!!!!!!!!!!!!!!!!!
                for ii in range(currentOpt.batch_size):
                    # print('FeatTT[-1]',FeatTT[-1].shape)
                    acc = self.metric.accuracy(C_pred[ii].unsqueeze(0), labelshow[ii]).cpu().numpy()
                    miou = self.metric.mean_iou(C_pred[ii].unsqueeze(0), labelshow[ii], num_classes=6).cpu().numpy()
                    mf1 = self.metric.mean_f1(C_pred[ii].unsqueeze(0), labelshow[ii], num_classes=6).cpu().numpy()
                    # img1 = axs1[ii * 3+1, 0].imshow(C_pred[ii].detach().cpu().numpy(), cmap='jet',
                    #                               vmin=0, vmax=1)
                    img1 = axs1[ii * 3 + 1, 0].imshow(C_pred[ii].detach().cpu().numpy(), cmap='gray',
                                                      vmin=0, vmax=5)
                    axs1[ii * 3+1, 0].set_title('acc:%.2f-miou:%.2f-mf1:%.2f' % (acc, miou, mf1), fontdict=title_font)
                    # axs1[ii * 3+1, 0].set_title('Max Weight:%.2f-%.2f' % (
                    #     FeatTT['Weight'][-1][ii].min(), FeatTT['Weight'][-1][ii].max()),fontdict=title_font)
                    axs1[ii * 3+1, 0].axis('off')
                    # cbar1 = fig1.colorbar(img1, ax=axs1[ii * 3+1, 0], extend='both', fraction=0.046, pad=0.04)
                    # cbar1.set_ticks([0, 0.25, 0.5, 0.75, 1])
                    # cbar1.set_label('Probability')

                    img2 = axs1[ii * 3+1, 1].imshow(distancesSoftmaxMax[ii].detach().cpu().numpy(), cmap='jet')
                    # axs[ii, 1].set_title('Mean Weight')
                    axs1[ii * 3+1, 1].set_title('Max Sim: %.2f-%.2f' % (
                        distancesSoftmaxMax[ii].min(), distancesSoftmaxMax[ii].max()),fontdict=title_font)
                    axs1[ii * 3+1, 1].axis('off')
                    cbar2 = fig1.colorbar(img2, ax=axs1[ii * 3+1, 1], extend='both', fraction=0.046, pad=0.04)
                    # cbar2.set_ticks([0, 0.25, 0.5, 0.75, 1])
                    cbar2.ax.tick_params(labelsize=6)  # 设置刻度标签的字体大小为10

                    # cbar2.set_label('Probability')
                    for jj in range(2, 8):
                        img3 = axs1[ii * 3+1, jj].imshow(distancesSoftmax[ii, jj - 2].detach().cpu().numpy(),
                                                       cmap='jet')
                        # vmin=0, vmax=1)
                        axs1[ii * 3+1, jj].set_title('Sim:%.2f-%.2f' % (
                            distancesSoftmax[ii, jj - 2].min(), distancesSoftmax[ii, jj - 2].max()),fontdict=title_font)
                        axs1[ii * 3+1, jj].axis('off')
                        cbar3 = fig1.colorbar(img3, ax=axs1[ii * 3+1, jj], extend='both', fraction=0.046, pad=0.04)
                        # cbar3.set_ticks([0, 0.25, 0.5, 0.75, 1])
                        # cbar3.set_label('Probability')
                        cbar3.ax.tick_params(labelsize=6)  # 设置刻度标签的字体大小为10
                ################Predict
                Prob_pred = torch.argmax(Probability.detach(), dim=1)

                for ii in range(currentOpt.batch_size):
                    acc = self.metric.accuracy(Prob_pred[ii].unsqueeze(0), labelT[ii]).cpu().numpy()
                    miou = self.metric.mean_iou(Prob_pred[ii].unsqueeze(0), labelT[ii], num_classes=6).cpu().numpy()
                    mf1 = self.metric.mean_f1(Prob_pred[ii].unsqueeze(0), labelT[ii], num_classes=6).cpu().numpy()

                    path = data_test['label_path'][ii].split('/')[-1]
                    img1 = axs1[ii * 3 + 2, 0].imshow(labelshow[ii][0].detach().cpu().numpy(), cmap='gray', vmin=0,
                                                      vmax=5)
                    axs1[ii * 3 + 2, 0].set_title(path,fontdict=title_font2)
                    axs1[ii * 3 + 2, 0].axis('off')

                    img2 = axs1[ii * 3 + 2, 1].imshow(Prob_pred[ii].detach().cpu().numpy(), cmap='gray',
                                                      vmin=0, vmax=5)
                    # axs[ii, 1].set_title('Mean Weight')
                    axs1[ii * 3 + 2, 1].set_title('acc:%.2f-miou:%.2f-mf1:%.2f' % (acc, miou, mf1), fontdict=title_font)
                    # axs1[ii * 3 + 2, 1].set_title(
                    #     'Max Proba:%.2f-%.2f' % (Probabilitymax[ii].min(), Probabilitymax[ii].max()),fontdict=title_font)
                    axs1[ii * 3 + 2, 1].axis('off')
                    # cbar2 = fig1.colorbar(img2, ax=axs1[ii * 3 + 2, 1], extend='both', fraction=0.046, pad=0.04)
                    # cbar2.set_ticks([0, 0.25, 0.5, 0.75, 1])
                    # cbar2.set_label('Probability')
                    cbar2.ax.tick_params(labelsize=6)  # 设置刻度标签的字体大小为10

                    for jj in range(2, 8):
                        img3 = axs1[ii * 3 + 2, jj].imshow(Probability[ii, jj - 2].detach().cpu().numpy(),
                                                           cmap='jet',
                                                           vmin=0, vmax=1)
                        axs1[ii * 3 + 2, jj].set_title('Proba:%.2f-%.2f' % (
                            Probability[ii, jj - 2].min(), Probability[ii, jj - 2].max()),fontdict=title_font)
                        axs1[ii * 3 + 2, jj].axis('off')
                        cbar3 = fig1.colorbar(img3, ax=axs1[ii * 3 + 2, jj], extend='both', fraction=0.046, pad=0.04)
                        cbar3.set_ticks([0, 0.25, 0.5, 0.75, 1])
                        # cbar3.set_label('Probability')
                        cbar3.ax.tick_params(labelsize=6)  # 设置刻度标签的字体大小为10

                plt.tight_layout()
                plt.savefig(self.savepath + 'WightPic/savepic/image%d_%d.png' % (epoch, i))
                plt.clf()
                display.clear_output(wait=True)
                display.display(plt.gcf())
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
class Metric_tool():
    # 函数：计算准确度
    def accuracy(self,preds, labs):
        correct = (preds == labs).sum().float()
        total = torch.numel(labs)
        return correct / total

    # 函数：计算每个类别的IoU
    def mean_iou(self,preds, labs, num_classes):
        iou_list = []
        for cls in range(num_classes):
            true_positive = ((preds == cls) & (labs == cls)).sum().float()
            false_positive = ((preds == cls) & (labs != cls)).sum().float()
            false_negative = ((preds != cls) & (labs == cls)).sum().float()
            union = true_positive + false_positive + false_negative
            if union == 0:
                iou = 0.0
            else:
                iou = true_positive / union
            iou_list.append(iou)
        mean_iou = torch.tensor(iou_list).mean()
        return mean_iou

    # 函数：计算每个类别的F1分数
    def mean_f1(self,preds, labs, num_classes):
        f1_list = []
        for cls in range(num_classes):
            true_positive = ((preds == cls) & (labs == cls)).sum().float()
            false_positive = ((preds == cls) & (labs != cls)).sum().float()
            false_negative = ((preds != cls) & (labs == cls)).sum().float()
            precision = true_positive / (true_positive + false_positive + 1e-6)
            recall = true_positive / (true_positive + false_negative + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            f1_list.append(f1)
        mean_f1 = torch.tensor(f1_list).mean()
        return mean_f1