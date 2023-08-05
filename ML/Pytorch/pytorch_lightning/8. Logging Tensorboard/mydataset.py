from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from typing import Any, Callable, Dict, List, Optional, Tuple

writer = SummaryWriter("logs")



class MyDataSet(Dataset):
    METAINFO = dict(
        classes=('background', 'aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor'),
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                 [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                 [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                 [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                 [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                 [0, 64, 128]])

    def __init__(self, root_dir, train: bool = True):
        self.train = train
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "JPEGImages")
        self.segimg_dir = os.path.join(root_dir, "SegmentationClass")
        if self.train:
            self.name_txt = os.path.join(root_dir, "ImageSets/Segmentation/train.txt")
        else:
            self.name_txt = os.path.join(root_dir, "ImageSets/Segmentation/val.txt")
        self.name_list = []
        with open(self.name_txt, 'r') as f:
            self.name_list = f.readlines()


    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img_name = self.name_list[idx].strip()
        img_path = self.image_dir + '/' + img_name + '.jpg'
        mask_path = self.segimg_dir + '/' + img_name + '.png'

        # label_name = self.label_list[idx]
        # img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)
        # label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        img = Image.open(img_path)
        mask = Image.open(mask_path)

        # img = np.array(img, dtype=np.float16).transpose((2, 0, 1))
        # mask = np.array(mask, dtype=np.float16)
        img = np.array(img).transpose((2, 0, 1))
        mask = np.array(mask)
        sample = {'img': img, 'label': mask}
        return sample

    def __len__(self):
        # assert len(self.image_list) == len(self.label_list)
        return len(self.name_list)

if __name__ == '__main__':
    pass


