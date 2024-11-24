import argparse
import os
import torch
from util import util


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='LEVIR-CD', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--result_dir', type=str, default='./results', help='predictions are saved here')

        self.parser.add_argument('--resnet', type=str, default='resnet18', help='resnet18|resnet34|resnet50|resnext50_32x4d|wide_resnet50_2')
        self.parser.add_argument('--batch_size', default=48, type=int)
        self.parser.add_argument('--num_class', default=2, type=int, help='two-class classification: changed or unchanged')
        self.parser.add_argument('--model_type', default='FullMM', type=str,
                                 help='FullMM|PartialMM')
        self.parser.add_argument('--aug', type=bool, default=True, help='')
        self.parser.add_argument('--init_type', type=str, default='normal', help='init type')
        self.parser.add_argument('--optimizer', default='adam', type=str)
        self.parser.add_argument('--lr', default=0.0001, type=float)
        self.parser.add_argument('--load_pretrain', type=bool, default=False,
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--use_ce_loss', default=False, type=bool)
        self.parser.add_argument('--use_hybrid_loss', default=False, type=bool)
        self.parser.add_argument('--use_UnchgInCenterLoss', default=True, type=bool)
        self.parser.add_argument('--use_UnchgNoCenterLoss', default=True, type=bool)
        self.parser.add_argument('--use_UnchgInCenterLossNew', default=True, type=bool)

        self.parser.add_argument('--gamma', type=int, default=0, help='gamma for Focal loss')
        self.parser.add_argument('--dataroot', type=str, default='/home/yjh/DatasetF/')
        # self.parser.add_argument('--dataset', type=str, default='LEVIR_CD',
        #                     help='choose which dataset to use.')
        self.parser.add_argument('--dset', type=str, default='OfficeHomeDataset_10072016',
                            choices=['office', 'office-home', 'office-caltech'])
        self.parser.add_argument('--seed', type=int, default=2021, help="random seed")

        self.parser.add_argument('--img_size', type=int, default=512)
        self.parser.add_argument('--label_norm', type=bool, default=True,  help='normalize label or not')
        self.parser.add_argument('--num_threads', type=int, default=16, help='# threads for loading data')
        self.parser.add_argument('--LChannel', type=bool, default=False, help='#generate L channel')
        self.parser.add_argument('--s', type=int, default=1, help="source")
        self.parser.add_argument('--t', type=int, default=1, help="source")
        self.parser.add_argument('--hardspilt', type=str, default='', help="hardspilt")
        self.parser.add_argument('--easyspilt', type=str, default='', help="easyspilt")
        self.parser.add_argument('--resize', type=int, default=2200, help='resize long size')
        self.parser.add_argument('--rcrop', type=str, default='512', help='rondom crop size')
        self.parser.add_argument('--hflip', type=float, default=0.5, help='random flip probility')
        self.parser.add_argument('--path_LP', type=float, default=0.5, help='random flip probility')
        self.parser.add_argument('--path_soft', type=float, default=0.5, help='random flip probility')
        self.parser.add_argument('--augmentations', type=bool, default=True, help='random flip probility')


        self.initialized = True

    def parse(self, save=True,gpu='0'):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain
        # self.opt.isTrain=True
        self.opt.gpu_ids=gpu
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        # if len(self.opt.gpu_ids) > 0:
        #     torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        # if save:
        #     expr_dir = os.path.join(self.opt.checkpoint_dir, self.opt.name)
        #     util.mkdirs(expr_dir)
        #     visualization_dir = os.path.join(self.opt.checkpoint_dir, self.opt.name, 'visualization')
        #     util.mkdirs(visualization_dir)
        #
        #     file_name = os.path.join(expr_dir, 'opt.txt')
        #     with open(file_name, 'wt') as opt_file:
        #         opt_file.write('------------ Options -------------\n')
        #         for k, v in sorted(args.items()):
        #             opt_file.write('%s: %s\n' % (str(k), str(v)))
        #         opt_file.write('-------------- End ----------------\n')
        return self.opt
