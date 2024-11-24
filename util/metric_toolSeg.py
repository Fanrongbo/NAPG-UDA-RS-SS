import torch
import torch.nn.functional as F
import numpy as np
def calculate_miou_and_f1(preds, labels, num_classes):
    # 将预测结果转换为与标签相同的形状
    # preds = torch.argmax(preds, dim=1)

    # 初始化混淆矩阵
    confusion_matrix = torch.zeros((num_classes, num_classes))

    for i in range(num_classes):
        for j in range(num_classes):
            confusion_matrix[i, j] = torch.sum((preds == i) & (labels == j))

    # 计算每个类别的 IoU
    IoU = torch.diag(confusion_matrix) / (confusion_matrix.sum(1) + confusion_matrix.sum(0) - torch.diag(confusion_matrix))
    mIoU = IoU.mean()

    # 计算 F1 分数
    precision = torch.diag(confusion_matrix) / confusion_matrix.sum(0)
    recall = torch.diag(confusion_matrix) / confusion_matrix.sum(1)
    F1 = 2 * (precision * recall) / (precision + recall)
    F1 = F1[~torch.isnan(F1)].mean()

    # 计算每类的准确率
    per_class_accuracy = torch.diag(confusion_matrix) / confusion_matrix.sum(1)

    # 计算总体准确率
    overall_accuracy = torch.diag(confusion_matrix).sum() / confusion_matrix.sum()

    # 计算平均 mIoU
    # mIoU = IoU.mean()
    score_dict = {'acc': overall_accuracy, 'preacc': per_class_accuracy, 'preIou': IoU,
                  'mIoU': mIoU,'mf1':F1}
    return score_dict
    # return mIoU, F1,IoU

# 示例使用
# preds = ... # 模型的预测结果, 尺寸为 ([4, 7, 512, 512])
# labels = ... # 标签, 尺寸为 ([4, 512, 512])


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        batch_confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        for lt, lp in zip(label_trues, label_preds):
            batch_confusion_matrix+=self._fast_hist( lt.flatten(), lp.flatten() )
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )

        return self._calculate_metrics(batch_confusion_matrix)
    def calculate_metrics(self):
        return self._calculate_metrics(self.confusion_matrix)
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                if k != "F1 Score":
                    string += "%s: %f\n" % (k, v)
                else:
                    for i in range(len(v)):
                        string += "F1_%s: %f\n" % (i, v[i])
                        # print('{0}'.format(i))
        #string+='Class IoU:\n'
        #for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def _calculate_metrics(self,hist):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        # hist = self.confusion_matrix
        acc = np.diag(hist).sum()
        per_class_accuracy = np.diag(hist) / hist.sum(1)

        # print('acc',acc.shape)
        # acc=acc/ hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        # cls_iu = dict(zip(range(self.n_classes), iu))
        # print('iu',iu)
        f1_score = []
        preIou=[]
        n = self.n_classes
        jj=0
        mf1=0
        for i in range(n):
            rowsum, colsum = sum(hist[i]), sum(hist[r][i] for r in range(n))
            precision = (hist[i][i] / float(colsum))
            recall = hist[i][i] / float(rowsum)
            f1 = 2 * precision * recall / (precision + recall)
            f1_score.append(f1)
            if f1 is not np.nan:
                mf1=mf1+f1
                jj=jj+1
            preIou.append(iu[i])
        mf1=mf1/jj


        score_dict = {'acc': acc_cls, 'preacc': per_class_accuracy, 'preIou': preIou, 'pref1': f1_score,
                      'mIoU': mean_iu, 'mf1': mf1}
        return score_dict
    def clear(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
class MetricsTracker:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.clear()

    def clear(self):
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes))
        self.total_samples = 0

    def _calculate_metrics(self, confusion_matrix):
        IoU = torch.diag(confusion_matrix) / (confusion_matrix.sum(1) + confusion_matrix.sum(0) - torch.diag(confusion_matrix)+1)
        precision = torch.diag(confusion_matrix) / confusion_matrix.sum(0)
        recall = torch.diag(confusion_matrix) / confusion_matrix.sum(1)
        F1 = 2 * (precision * recall) / (precision + recall)
        per_class_accuracy = torch.diag(confusion_matrix) / confusion_matrix.sum(1)
        overall_accuracy = torch.diag(confusion_matrix).sum() / confusion_matrix.sum()

        mIoU = IoU.mean()
        mean_F1 = F1[~torch.isnan(F1)].mean()
        score_dict = {'acc': overall_accuracy, 'preacc': per_class_accuracy, 'preIou': IoU,'pref1':F1,
                      'mIoU': mIoU, 'mf1': mean_F1}
        return score_dict


    def update(self, preds, labels):
        batch_confusion_matrix = torch.zeros((self.num_classes, self.num_classes))
        # preds = torch.argmax(preds, dim=1)

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                batch_confusion_matrix[i, j] = torch.sum((preds == i) & (labels == j))

        self.confusion_matrix += batch_confusion_matrix
        self.total_samples += labels.size(0)

        return self._calculate_metrics(batch_confusion_matrix)

    def calculate_metrics(self):
        return self._calculate_metrics(self.confusion_matrix)

# 示例使用
# tracker = MetricsTracker(num_classes=7)
# for data, labels in dataloader:
#     preds = model(data)
#     batch_metrics = tracker.update(preds, labels)
#     # Do something with batch_metrics if needed
# overall_metrics = tracker.calculate_metrics()
# tracker.clear() # 在新的轮次开始之前清空记录

