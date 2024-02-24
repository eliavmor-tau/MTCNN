from torch.utils.data import Dataset
import pandas as pd
import torch
import os
from utils import random_crop_and_update_bbox, plot_im_with_bbox, IoU
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor,Resize


class PNetDataset(Dataset):

    def __init__(self, path: str, partition: str, transform=None, neg_th: int = 0.3, pos_th: int = 0.65,
                 min_crop: int = 12, max_crop: int = 100, out_size=12, p_prob=0.5) -> Dataset:
        super(PNetDataset, self).__init__()
        self.path = path
        data_partition = pd.read_csv(os.path.join(path, "list_bbox_celeba_align_and_crop.csv"), index_col=False)
        self.bbox = pd.read_csv(os.path.join(path, "list_bbox_celeba_align_and_crop.csv"), index_col=False)
        if partition.lower() == "train":
            partition = 0
        elif partition.lower() == "val":
            partition = 1
        elif partition.lower() == "test":
            partition = 2
        else:
            raise (f"unkonwn partition {partition}")
        self.min_crop = min_crop
        self.max_crop = max_crop
        self.neg_th = neg_th
        self.pos_th = pos_th
        self.p_prob = p_prob
        self.out_size = out_size
        self.resize_transform = Resize(out_size, antialias=True)
        self.data_split = data_partition[data_partition['partition'] == partition].image_name
        self.transform = transform

    def __generate_sample(self, im, bbox, expected_label, crop_size):
        generate_sample = True
        cropped_im, cropped_bbox = im, bbox
        tries_count = 0
        while generate_sample:
            cropped_im, cropped_bbox = random_crop_and_update_bbox(im, bbox, crop_size)
            image_bbox = torch.tensor([0, 0, cropped_im.shape[2], cropped_im.shape[1]])
            iou = IoU(image_bbox, cropped_bbox)
            if (iou >= self.pos_th and expected_label) or (iou <= self.neg_th and not expected_label):
                break
            tries_count += 1
            if tries_count >= 5:
                break
        return cropped_im, cropped_bbox

    def __getitem__(self, item):
        if 0 <= item < self.__len__():
            p = np.random.uniform(0, 1)
            expected_label = p >= self.p_prob
            crop_size = np.random.randint(self.min_crop, self.max_crop, size=1)[0]
            crop_size = [crop_size, crop_size]
            sample_name = self.data_split.iloc[item]
            bbox = torch.tensor(list(self.bbox[self.bbox['image_name'] == sample_name].to_numpy()[0][2:-1]))
            im = np.array(Image.open(os.path.join(self.path, "images", sample_name)))
            if self.transform is not None:
                im = self.transform(im)
            im, bbox = self.__generate_sample(im, bbox, expected_label, crop_size)
            im = self.resize_transform(im)
            bbox = torch.round(bbox * self.out_size / crop_size[0])
            return im, bbox, torch.tensor(int(expected_label), dtype=torch.long)

    def __len__(self):
        return len(self.data_split)


if __name__ == "__main__":
    # test dataset.
    dataset = PNetDataset(path="data/celebA", partition="val",
                          transform=ToTensor(), min_crop=30, max_crop=100,
                          out_size=12)
    for i in range(20):
        im, bbox, label = dataset[i]
        plot_im_with_bbox(im, bbox, title=f"image={i} label={label}")
