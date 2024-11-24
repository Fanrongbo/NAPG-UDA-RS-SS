"""
Copied and modified from
https://github.com/justchenhao/BIT_CD
"""
import numpy as np


###################      cm metrics      ###################
class ConfuseMatrixMeter():
    """Computes and stores the average and current value"""
    def __init__(self, n_class):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class
        self.initializedS = False
        self.initializedT = False
        self.confuse_matrixT=0
        self.confuse_matrixS=0
    def clear(self):
        self.initializedS = False
        self.initializedT = False

    def confuseMS(self,pr, gt):
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        current_score = cm2F1(val)
        if not self.initializedS:
            self.confuse_matrixS=val
            self.initializedS = True
        else:
            self.confuse_matrixS = self.confuse_matrixS + val
        return current_score
    def confuseMT(self,pr, gt):
        # pr=np.array(pr,dtype=np.int)
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        current_score = cm2F1(val)
        if not self.initializedT:
            self.confuse_matrixT=val
            self.initializedT = True
        else:
            self.confuse_matrixT = self.confuse_matrixT + val
        core_dict = {'accT': current_score['acc'], 'chgT': current_score['chgAcc'], 'unchgT': current_score['unchgAcc'],
                     'mF1T': current_score['fm1'], 'recallT':current_score['recall']}
        return core_dict

    def get_scoresT(self):
        scores_dict = cm2F1(self.confuse_matrixT)
        core_dict = {'accT': scores_dict['acc'], 'chgT': scores_dict['chgAcc'],
                     'unchgT': scores_dict['unchgAcc'], 'mF1T': scores_dict['fm1']}
        message='T:'
        for k, v in core_dict.items():
            message += '%s: %.3f ' % (k, v * 100)
        print(message)
        return message,core_dict
    def get_scores(self):
        scores_dict = cm2score(self.confuse_matrixS)
        return scores_dict
    def get_scoresTT(self):
        scores_dict = cm2score(self.confuse_matrixT)
        return scores_dict



def harmonic_mean(xs):
    harmonic_mean = len(xs) / sum((x+1e-6)**-1 for x in xs)
    return harmonic_mean


def cm2F1(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    fn = hist.sum(axis=1) - np.diag(hist)
    fp = hist.sum(axis=0) - np.diag(hist)
    if n_class==2:
        tn = hist.sum() - (fp + fn + tp)
    else:
        tn = hist.sum() - (fp + fn + tp) - hist.sum(axis=1) + tp

    sum_a1 = hist.sum(axis=1)#TP+FN
    sum_a0 = hist.sum(axis=0)#TP+FP
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    chgAcc = tp[1] / (tp[1] + fn[1]+1)
    unchgAcc = tn[1] / (tn[1] + fp[1]+1)

    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)
    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    recallM=np.nanmean(recall)
    precisionM=np.nanmean(precision)
    score_dict = {'acc': acc, 'chgAcc': chgAcc, 'unchgAcc': unchgAcc,
                  'tp': int(tp[1]), 'fn': int(fn[1]), 'fp': int(fp[1]), 'tn': int(tn[1]),'fm1':mean_F1,'recall':recallM,'precision':precisionM}
    return score_dict





def get_confuse_matrix(num_classes, label_gts, label_preds):
    def __fast_hist(label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < num_classes)
        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=num_classes**2).reshape(num_classes, num_classes)
        return hist
    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix


def get_mIoU(num_classes, label_gts, label_preds):
    confusion_matrix = get_confuse_matrix(num_classes, label_gts, label_preds)
    score_dict = cm2score(confusion_matrix)
    return score_dict['miou']
def cm2score(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    fn = hist.sum(axis=1) - np.diag(hist)
    fp = hist.sum(axis=0) - np.diag(hist)
    tn = hist.sum() - (fp + fn + tp)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    chgAcc=tp[1]/(tp[1]+fn[1]+1)
    unchgAcc = tn[1] / (tn[1] + fp[1]+1)
    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    recallm = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)
    precisionm = np.nanmean(precision)

    # F1 score
    F1 = 2*recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    # ---------------------------------------------------------------------- #
    # 2. Frequency weighted Accuracy & Mean IoU
    # ---------------------------------------------------------------------- #
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    #
    cls_iou = dict(zip(['iou_'+str(i) for i in range(n_class)], iu))

    cls_precision = dict(zip(['precision_'+str(i) for i in range(n_class)], precision))
    cls_recall = dict(zip(['recall_'+str(i) for i in range(n_class)], recall))
    cls_F1 = dict(zip(['F1_'+str(i) for i in range(n_class)], F1))

    score_dict = {'acc': acc, 'miou': mean_iu, 'mf1':mean_F1, 'chgAcc': chgAcc, 'unchgAcc': unchgAcc,
                  'tp': int(tp[1]), 'fn': int(fn[1]), 'fp': int(fp[1]), 'tn': int(tn[1]),'recall':recallm,'precision':precisionm}
    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)
    return score_dict