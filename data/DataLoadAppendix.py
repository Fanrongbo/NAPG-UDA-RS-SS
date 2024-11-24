import torch
import numpy as np
from torch.nn import functional as F
import torch.nn as nn

class feat_prototype_distance_module(nn.Module):
    def __init__(self):
        super(feat_prototype_distance_module, self).__init__()

    def forward(self, feat, objective_vectors, class_numbers):
        N, C, H, W = feat.shape
        feat_proto_distance = -torch.ones((N, class_numbers, H, W)).to(feat.device)
        for i in range(class_numbers):
            #feat_proto_distance[:, i, :, :] = torch.norm(torch.Tensor(self.objective_vectors[i]).reshape(-1,1,1).expand(-1, H, W).to(feat.device) - feat, 2, dim=1,)
            feat_proto_distance[:, i, :, :] = torch.norm(objective_vectors[0, i].reshape(-1,1,1).expand(-1, H, W) - feat, 2, dim=1,)
        return feat_proto_distance
def regular_loss(activation,opt):
    logp = F.log_softmax(activation, dim=1)
    if opt.regular_type == 'MRENT':
        p = F.softmax(activation, dim=1)
        loss = (p * logp).sum() / (p.shape[0]*p.shape[2]*p.shape[3])
    elif opt.regular_type == 'MRKLD':
        loss = - logp.sum() / (logp.shape[0]*logp.shape[1]*logp.shape[2]*logp.shape[3])
    return loss
def feat_prototype_distance(objective_vectors,feat):
    N, C, H, W = feat.shape
    feat_proto_distance = -torch.ones((N, 6, H, W)).to(feat.device)

    for i in range(6):
        # print(feat_proto_distance.shape, feat.shape,objective_vectors.shape, objective_vectors[i].reshape(-1, 1, 1).expand(-1, H, W).shape)

        #feat_proto_distance[:, i, :, :] = torch.norm(torch.Tensor(self.objective_vectors[i]).reshape(-1,1,1).expand(-1, H, W).to(feat.device) - feat, 2, dim=1,)
        feat_proto_distance[:, i, :, :] = torch.norm(objective_vectors[i].reshape(-1,1,1).expand(-1, H, W) - feat, 2, dim=1,)
    return feat_proto_distance
def process_label( label,Device):
    batch, channel, w, h = label.size()
    pred1 = torch.zeros(batch, 6 + 1, w, h).to(Device)
    id = torch.where(label < 6, label, torch.Tensor([6]).to(Device))
    pred1 = pred1.scatter_(1, id.long(), 1)
    return pred1
def calculate_mean_vector( feat_cls, outputs, labels=None, thresh=None,Device=None):
    outputs_softmax = F.softmax(outputs, dim=1)
    if thresh is None:
        thresh = -1
    conf = outputs_softmax.max(dim=1, keepdim=True)[0]
    mask = conf.ge(thresh)
    outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
    outputs_argmax = process_label(outputs_argmax.float(),Device=Device)
    if labels is None:
        outputs_pred = outputs_argmax
    else:
        labels_expanded = process_label(labels,Device=Device)
        outputs_pred = labels_expanded * outputs_argmax
    scale_factor = F.adaptive_avg_pool2d(outputs_pred * mask, 1)
    vectors = []
    ids = []
    for n in range(feat_cls.size()[0]):
        for t in range(6):
            if scale_factor[n][t].item()==0:
                continue
            if (outputs_pred[n][t] > 0).sum() < 10:
                continue
            s = feat_cls[n] * outputs_pred[n][t] * mask[n]
            # scale = torch.sum(outputs_pred[n][t]) / labels.shape[2] / labels.shape[3] * 2
            # s = normalisation_pooling()(s, scale)
            s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
            vectors.append(s)
            ids.append(t)
    return vectors, ids
def update_objective_SingleVector( id, vector, name='moving_average', start_mean=True,opt=None,objective_vectors_num=None,objective_vectors=None):
    if vector.sum().item() == 0:
        return
    if start_mean and objective_vectors_num[id].item() < 100:
        name = 'mean'
    if name == 'moving_average':
        objective_vectors[id] = objective_vectors[id] * (1 - opt.proto_momentum) + opt.proto_momentum * vector.squeeze()
        objective_vectors_num[id] += 1
        objective_vectors_num[id] = min(objective_vectors_num[id], 3000)
    elif name == 'mean':
        objective_vectors[id] = objective_vectors[id] * objective_vectors_num[id] + vector.squeeze()
        objective_vectors_num[id] += 1
        objective_vectors[id] = objective_vectors[id] / objective_vectors_num[id]
        objective_vectors_num[id] = min(objective_vectors_num[id], 3000)
        pass
    else:
        raise NotImplementedError('no such updating way of objective vectors {}'.format(name))
def full2weak( feat, target_weak_params):
    tmp = []
    for i in range(feat.shape[0]):
        # print('RandomSized',len(target_weak_params['RandomSized'][0]))
        h, w = target_weak_params['RandomSized'][0][i], target_weak_params['RandomSized'][1][i]
        feat_ = F.interpolate(feat[i:i+1], size=[int(h/4), int(w/4)], mode='bilinear', align_corners=True)
        y1, y2, x1, x2 = target_weak_params['RandomCrop'][0][i], target_weak_params['RandomCrop'][1][i], target_weak_params['RandomCrop'][2][i], target_weak_params['RandomCrop'][3][i]
        y1, th, x1, tw = int(y1/4), int((y2-y1)/4), int(x1/4), int((x2-x1)/4)
        feat_ = feat_[:, :, y1:y1+th, x1:x1+tw]
        if target_weak_params['RandomHorizontallyFlip'][i]:
            inv_idx = torch.arange(feat_.size(3)-1,-1,-1).long().to(feat_.device)
            feat_ = feat_.index_select(3,inv_idx)
        tmp.append(feat_)
    feat = torch.cat(tmp, 0)
    return feat
def get_prototype_weight(feat, label=None, target_weak_params=None,opt=None,objective_vectors=None):
    feat = full2weak(feat, target_weak_params)
    feat_proto_distance = feat_prototype_distance(objective_vectors,feat)
    feat_nearest_proto_distance, feat_nearest_proto = feat_proto_distance.min(dim=1, keepdim=True)

    feat_proto_distance = feat_proto_distance - feat_nearest_proto_distance
    weight = F.softmax(-feat_proto_distance * opt.proto_temperature, dim=1)
    return weight

def rcef( pred, labels,Device):
    pred = F.softmax(pred, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0)
    mask = (labels != 250).float()
    labels[labels==250] = 6
    label_one_hot = torch.nn.functional.one_hot(labels, 6 + 1).float().to(Device)
    label_one_hot = torch.clamp(label_one_hot.permute(0,3,1,2)[:,:-1,:,:], min=1e-4, max=1.0)
    rce = -(torch.sum(pred * torch.log(label_one_hot), dim=1) * mask).sum() / (mask.sum() + 1e-6)
    return rce
def affine_sample(tensor, v, type):
    # tensor: B*C*H*W
    # v: scalar, translation param
    if type == 'Rotate':
        theta = np.array([[np.cos(v/180*np.pi), -np.sin(v/180*np.pi), 0], [np.sin(v/180*np.pi), np.cos(v/180*np.pi), 0]]).astype(float)
    elif type == 'ShearX':
        theta = np.array([[1, v, 0], [0, 1, 0]]).astype(float)
    elif type == 'ShearY':
        theta = np.array([[1, 0, 0], [v, 1, 0]]).astype(float)
    elif type == 'TranslateX':
        theta = np.array([[1, 0, v], [0, 1, 0]]).astype(float)
    elif type == 'TranslateY':
        theta = np.array([[1, 0, 0], [0, 1, v]]).astype(float)

    H = tensor.shape[2]
    W = tensor.shape[3]
    theta[0,1] = theta[0,1]*H/W
    theta[1,0] = theta[1,0]*W/H
    if type != 'Rotate':
        theta[0,2] = theta[0,2]*2/H + theta[0,0] + theta[0,1] - 1
        theta[1,2] = theta[1,2]*2/H + theta[1,0] + theta[1,1] - 1

    theta = torch.Tensor(theta).unsqueeze(0)
    grid = F.affine_grid(theta, tensor.size(), align_corners=True).to(tensor.device)
    tensor_t = F.grid_sample(tensor, grid, mode='nearest', align_corners=True)
    return tensor_t
def label_strong_T( label, params, padding, scale=1):
    label = label + 1
    for i in range(label.shape[0]):
        for (Tform, param) in params.items():
            if Tform == 'Hflip' and param[i].item() == 1:
                label[i] = label[i].clone().flip(-1)
            elif (Tform == 'ShearX' or Tform == 'ShearY' or Tform == 'TranslateX' or Tform == 'TranslateY' or Tform == 'Rotate') and param[i].item() != 1e4:
                v = int(param[i].item() // scale) if Tform == 'TranslateX' or Tform == 'TranslateY' else param[i].item()
                label[i:i+1] = affine_sample(label[i:i+1].clone(), v, Tform)
            elif Tform == 'CutoutAbs' and isinstance(param, list):
                x0 = int(param[0][i].item() // scale)
                y0 = int(param[1][i].item() // scale)
                x1 = int(param[2][i].item() // scale)
                y1 = int(param[3][i].item() // scale)
                label[i, :, y0:y1, x0:x1] = 0
    label[label == 0] = padding + 1  # for strong augmentation, constant padding
    label = label - 1
    return label
def rce( pred, labels,device):
    pred = F.softmax(pred, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0)
    mask = (labels != 250).float()
    labels[labels==250] = 6
    label_one_hot = torch.nn.functional.one_hot(labels, 6 + 1).float().to(device)
    label_one_hot = torch.clamp(label_one_hot.permute(0,3,1,2)[:,:-1,:,:], min=1e-4, max=1.0)
    rce = -(torch.sum(pred * torch.log(label_one_hot), dim=1) * mask).sum() / (mask.sum() + 1e-6)
    return rce