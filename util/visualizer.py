"""
Copied and modified from
https://github.com/NVIDIA/pix2pixHD/tree/master/util
"""
import os
import torch
import numpy as np
import time
from . import util
import matplotlib.pyplot as plt
from skimage import io
from option.config import cfg


class Visualizer():
    def __init__(self, opt):
        self.name = opt.name
        self.log_name = os.path.join(opt.checkpoint_dir, opt.name, 'loss_log.txt')
        self.visualization_dir = os.path.join(opt.checkpoint_dir, opt.name, 'visualization')
        self.result_dir = os.path.join(opt.result_dir)

        # with open(self.log_name, "a") as log_file:
        # now = time.strftime("%c")
        # cfg.TRAINLOG.LOGTXT.write('================ Training Loss (%s) ================\n' % now)


    def print_current_errors_acc(self, phase, epoch, i, errors, acc, t):
        message = '(phase: %s, epoch: %d, iters: %d, running_mf1: %.5f, time: %.5f) ' % (phase, epoch, i, acc, t)
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)


    def print_scores(self, phase, epoch, scores):

        message = '%s \n (phase: %s, epoch: %s) ' % (time.strftime("%c"),phase, epoch)

        for k, v in scores.items():
            if k in ['tn', 'tp', 'fn', 'fp']:
                message += '%s: %d ' % (k, int(v) )
                if k=='tn':
                    message+='\n      '
                continue
            message += '%s: %.3f ' % (k, v*100)

        print(message)
        # with open(self.log_name, "a") as log_file:
        #     log_file.write('%s\n' % message)
        return message
    def print_sorces_Seg(self,phase,epoch,scores):
        message = '%s \n (phase: %s, epoch: %s) ' % (time.strftime("%c"),phase, epoch)
        # print('scores',scores)
        for k, v in scores.items():
            # if k in ['preacc', 'preIou']:
            #     continue
            if k in ['preacc', 'preIou','pref1']:
                # print(k,(scores[k]))

                for ii in range(len(scores[k])):
                    message += '%s-%d: %.2f ' % (k,ii, (scores[k])[ii])

                # message += '%s: %.2f ' % (k, v)
                continue
            if isinstance(v,int):
                message += '%s: %d ' % (k, v)
            else:
                # print('k',k,'v', v)
                message += '%s: %.1f ' % (k, v*100)
        print(message)

        return message
    def print_current_scores(self, phase, epoch,c,a ,scores):
        message = '(phase: %s, epoch: %s | %d/%d) ' % (phase, epoch,c,a)

        for k, v in scores.items():
            if k in ['tn', 'tp', 'fn', 'fp']:
                continue
            if k in ['CEL', 'DALoss', 'FeatLoss','Cdist']:
                message += '%s: %.2f ' % (k, v)
                continue
            if isinstance(v,int):
                message += '%s: %d ' % (k, v)
            else:
                # print('k',k,'v', v)
                message += '%s: %.1f ' % (k, v*100)

        # print(message)
        return message
    def print_current_sorces_Seg(self,phase,epoch,c,a ,scores):
        message= '(phase: %s, epoch: %s | %d/%d) ' % (phase, epoch,c,a)
        for k, v in scores.items():
            if k in ['preacc', 'preIou','pref1']:
                continue
            if k in ['CEL', 'DALoss', 'FeatLoss','Cdist']:
                message += '%s: %.2f ' % (k, v)
                continue
            if isinstance(v,int):
                message += '%s: %d ' % (k, v)
            else:
                # print('k',k,'v', v)
                message += '%s: %.1f ' % (k, v*100)

        return message
    def visualize_current_results(self, phase, epoch, t1_img, t2_img, gt, pred):
        file_name = 'phase_' + str(phase) + '_epoch_' + str(epoch) + '.jpg'
        t1_img_ = util.make_numpy_grid(util.de_norm(t1_img))
        t2_img_ = util.make_numpy_grid(util.de_norm(t2_img))
        gt_ = util.make_numpy_grid(gt)
        pred_ = torch.argmax(pred, dim=1, keepdim=True)
        pred_ = util.make_numpy_grid(pred_ * 255)

        vis = np.concatenate([t1_img_, t2_img_, gt_, pred_], axis=0)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        save_path = os.path.join(self.visualization_dir, file_name)
        plt.imsave(save_path, vis)


    def save_pred(self, t1_img, t2_img, name, model):
        model.eval()
        try:
            preds = model.module.inference(t1_img.cuda(), t2_img.cuda())
        except:
            preds = model.inference(t1_img, t2_img)
        pred = torch.argmax(preds, dim=1, keepdim=True)
        pred = pred * 255
        save_path = os.path.join(self.result_dir, str(name) + '.png')
        pred = pred[0].cpu().numpy()
        io.imsave(save_path, np.array(np.squeeze(pred), dtype=np.uint8))

