import os.path
import torch
from data.image_folder import make_dataset
from data.preprocessing import Preprocessing
from PIL import Image
import numpy as np
from option.config import cfg
from proDAdata.randaugment import RandAugmentMC
from proDAdata.augmentations import *


class SegmentationMMLabDataset(torch.utils.data.Dataset):

    def initialize(self, opt):
        self.randaug = RandAugmentMC(2, 3)
        self.augmentations=Compose([RandomSized(opt.resize),
                    RandomCrop(opt.rcrop),
                    RandomHorizontallyFlip(opt.hflip)])
        self.opt = opt
        self.root = opt.dataroot
        self.img_size=(512,512)
        ### input T1_img
        if opt.phase in ['train','val']:
            self.img=os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.s],'img_dir',opt.phase)
            self.imgpath=sorted(make_dataset([self.img]))

            self.dir_label=os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.s],'ann_dir',opt.phase)
            self.label_paths = sorted(make_dataset([self.dir_label]))
        elif opt.phase in ['targetSelect']:
            self.img = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.t], 'img_dir', 'train')
            print('opt.hardspilt',opt.hardspilt)
            list_path = opt.hardspilt.format('all')
            print('list_path',list_path)
            ### input T2_img
            # self.t2_paths = sorted(make_dataset([self.dir_t2]))
            # dir_label = 'label'
            self.dir_label = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.t], 'ann_dir', 'train')

            self.imgpath=[]
            self.label_paths=[]
            with open(list_path) as f:
                for i_id in f:
                    i_id=i_id.strip()
                    self.imgpath.append(i_id)
                    self.label_paths.append(i_id.replace('img_dir', 'ann_dir'))
            print('label_paths', self.label_paths)
        elif opt.phase in ['target','targetTest']:
            self.img = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.t], 'img_dir', 'train')
            self.imgpath = sorted(make_dataset([self.img]))

            self.dir_label = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.t], 'ann_dir', 'train')
            self.label_paths = sorted(make_dataset([self.dir_label]))


        self.dataset_size = len(self.label_paths)
        # if self.opt.phase == 'train':
        #     print('with_Lchannel=opt.LChannel', opt.LChannel)
        # self.preprocess = Preprocessing(
        #     img_size=self.opt.img_size,
        #     with_random_hflip=False,
        #     with_random_vflip=False,
        #     with_scale_random_crop=False,
        #     with_random_blur=opt.aug,
        #     with_Lchannel=opt.LChannel
        # )
        # else:
        #     self.preprocess= Preprocessing(
        #         img_size=self.opt.img_size,
        #         with_Lchannel=opt.LChannel
        #         )
        # self.preprocess = Preprocessing(
        #     img_size=self.opt.img_size,
        #     with_Lchannel=opt.LChannel
        # )
    def __getitem__(self, index):


        ### input T1_img 
        img_path = self.imgpath[index]
        label_path = self.label_paths[index]

        img = Image.open(img_path).convert('RGB')
        lbl = Image.open(label_path)
        # img = img.resize(self.img_size, Image.BILINEAR)
        # lbl = lbl.resize(self.img_size, Image.NEAREST)

        img = np.array(img, dtype=np.uint8)
        lbl = np.array(lbl, dtype=np.uint8)
        # print('lbl',lbl.shape,lbl.max(),lbl.min())
        # lbl[lbl == 0] = 6  # 首先将0变为-1，以避免与其他值冲突
        # lbl[lbl == 5] = 0  # 将5变为0
        # lbl[lbl == 6] = 5  # 最后将-1变为5
        # lbl = np.array(lbl, dtype=np.uint8)

        lblori=lbl.copy()
        # lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        img_full = img.copy().astype(np.float64)
        # img_full -= self.mean
        img_full = img_full.astype(float) / 255.0
        img_full = img_full.transpose(2, 0, 1)
        lp, lpsoft, weak_params = None, None, None
        # if cfg.TRAINLOG.DATA_NAMES[self.opt.t] in img_path:
        #     lpRootpath = os.path.join(self.opt.dataroot, cfg.TRAINLOG.DATA_NAMES[self.opt.t], 'lp')
        # else:
        #     lpRootpath = os.path.join(self.opt.dataroot, cfg.TRAINLOG.DATA_NAMES[self.opt.s], 'lp')
        # # print('img_path',img_path)
        # lpsoft = np.load(os.path.join(lpRootpath, os.path.basename(img_path).replace('.png', '_lpSoft.npy')))
        # lpsoft_max=np.max(lpsoft,axis=0)
        # lp = np.load(os.path.join(lpRootpath, os.path.basename(img_path).replace('.png', '_lp.npy')))
        # # print('lp', lp.shape, lpsoft.shape)#(1, 128, 128) (1, 6, 128, 128)
        # lp[lpsoft_max<= 0.1] = 250

        # if self.split == 'train' and self.opt.used_save_pseudo:
        #     if self.opt.proto_rectify:
        #         lpRootpath = os.path.join(self.opt.dataroot, cfg.TRAINLOG.DATA_NAMES[self.opt.t], 'lp')
        #         print('lpRootpath',lpRootpath)
        #         print(os.path.join(self.opt.path_soft, os.path.basename(img_path).replace('.png', '_lpSoft.npy')))
        #         # print(os.path.join(self.opt.path_soft, os.path.basename(img_path).replace('.png', '.npy')))
        #         lpsoft = np.load(os.path.join(lpRootpath, os.path.basename(img_path).replace('.png', '_lpSoft.npy')))
        #     else:
        #         lpRootpath = os.path.join(self.opt.dataroot, cfg.TRAINLOG.DATA_NAMES[self.opt.t], 'lp')
        #         lp_path = np.load(os.path.join(lpRootpath, os.path.basename(img_path).replace('.png', '_lpSoft.npy')))
        #
        #         # lp_path = os.path.join(self.opt.path_LP, os.path.basename(img_path))
        #         lp = Image.open(lp_path)
        #         lp = lp.resize(self.img_size, Image.NEAREST)
        #         lp = np.array(lp, dtype=np.uint8)
        #         if self.opt.threshold:
        #             conf = np.load(
        #                 os.path.join(self.opt.path_LP, os.path.basename(img_path).replace('.png', '_conf.npy')))
        #             lp[conf <= self.opt.threshold] = 250
        # [img_weak], [label_weak] = self.preprocess.transform([img_full], [lblori], to_tensor=False)

        input_dict = {}
        if self.opt.augmentations:
        # if self.augmentations != None:
            img, lbl, lp, lpsoft, weak_params = self.augmentations(img, lbl, lp, lpsoft)
            img_strong, params = self.randaug(Image.fromarray(img))
            img_strong, _, _ = self.transform(img_strong, lbl)
            input_dict['img_strong'] = img_strong
            input_dict['params'] = params

        img, lbl_, lp = self.transform(img, lbl, lp)

        input_dict['img'] = img

        input_dict['img_full'] = torch.from_numpy(img_full).float()
        input_dict['label_full'] = self.transformlabel(lblori)
        input_dict['label'] = lbl_

        input_dict['label_path'] = label_path
        input_dict['lp'] = lp
        input_dict['lpsoft'] = lpsoft
        input_dict['weak_params'] = weak_params  # full2weak
        input_dict['img_path'] = img_path
        input_dict = {k: v for k, v in input_dict.items() if v is not None}
        return input_dict


    def __len__(self):
        return len(self.label_paths) // self.opt.batch_size * self.opt.batch_size
    def transformlabel(self,lbl):
        classes = np.unique(lbl)
        lbl = np.array(lbl)
        lbl = lbl.astype(float)
        # lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")  # TODO: compare the original and processed ones
        lbl = torch.from_numpy(lbl).long()
        return lbl
    def transform(self, img, lbl, lp=None, check=True):
        """transform

        :param img:
        :param lbl:
        """
        # img = m.imresize(
        #     img, (self.img_size[0], self.img_size[1])
        # )  # uint8 with RGB mode
        img = np.array(img)
        # img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        # img -= self.mean
        img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = np.array(lbl)
        lbl = lbl.astype(float)
        # lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")  # TODO: compare the original and processed ones

        # if check and not np.all(
        #         np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):  # todo: understanding the meaning
        #     print("after det", classes, np.unique(lbl))
        #     raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        if lp is not None:
            classes = np.unique(lp)
            lp = np.array(lp)
            # if not np.all(np.unique(lp[lp != self.ignore_index]) < self.n_classes):
            #     raise ValueError("lp Segmentation map contained invalid class values")

            lp = torch.from_numpy(lp).long()

        return img, lbl, lp

    def encode_segmap(self, mask):
        # Put all void classes to zero
        label_copy = 250 * np.ones(mask.shape, dtype=np.uint8)
        for k, v in list(self.class_map.items()):
            label_copy[mask == k] = v
        return label_copy