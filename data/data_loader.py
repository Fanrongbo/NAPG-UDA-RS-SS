import torch.utils.data
from data.seg_mmlab_dataset import SegmentationMMLabDataset
from option.config import cfg

def CreateDataset(opt):
    dataset = None

    dataset = SegmentationMMLabDataset()
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(torch.utils.data.Dataset):

    def initialize(self, opt):
        if opt.phase =='background':
            size=1
        else:
            size=opt.batch_size
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=size,
            shuffle=(opt.phase == 'train')or(opt.phase == 'target')or(opt.phase == 'valTr')or(opt.phase == 'background'),
            # shuffle=False,
            num_workers=int(opt.num_threads))


    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)


def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    return data_loader