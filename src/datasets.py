import copy
import os
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from tqdm import tqdm
from utils import random_crop_and_update_bbox, IoU


class MTCNNDataset(Dataset):
    """
    Dataset class for MTCNN training.
    """

    def __init__(self, path: str, partition: str, transform=None, neg_th: int = 0.3, pos_th: int = 0.65,
                 min_crop: int = 12, max_crop: int = 100, out_size=24, n=1000, n_hard=1000,
                 previous_net=None, previous_transform=None) -> Dataset:
        """
        Initializes the MTCNNDataset.

        Args:
            path (str): Path to the dataset.
            partition (str): Dataset partition ('train', 'val', or 'test').
            transform (callable, optional): Optional transform to be applied on the sample.
            neg_th (int): Negative threshold for IoU.
            pos_th (int): Positive threshold for IoU.
            min_crop (int): Minimum crop size.
            max_crop (int): Maximum crop size.
            out_size (int): Output image size.
            n (int): Number of samples.
            n_hard (int): Number of hard samples.
            previous_net: Previous network for hard sample generation.
            previous_transform: Previous transform for hard sample generation.
        """
        super(MTCNNDataset, self).__init__()
        self.previous_net = previous_net
        self.previous_transform = previous_transform
        self.path = path
        data_partition = pd.read_csv(os.path.join(path, "../data/celebA/images/list_bbox_celeba_align_and_crop.csv"),
                                     index_col=False)
        self.bbox = pd.read_csv(os.path.join(path, "../data/celebA/images/list_bbox_celeba_align_and_crop.csv"),
                                index_col=False)
        if partition.lower() == "train":
            partition = 0
        elif partition.lower() == "val":
            partition = 1
        elif partition.lower() == "test":
            partition = 2
        else:
            raise ValueError(f"Unknown partition {partition}")

        self.min_crop = min_crop
        self.max_crop = max_crop
        self.neg_th = neg_th
        self.pos_th = pos_th
        self.out_size = out_size
        self.resize_transform = Resize(out_size, antialias=True)
        self.data_split = data_partition[data_partition['partition'] == partition].image_name
        self.transform = transform
        print(f"len(self.data_split) = {len(self.data_split)}")
        self.n = n
        self.n_hard = n_hard
        self.labels = []
        self.bboxes = []
        self.images = []
        self.__create_data()

    def __random_crop_image_and_bbox(self, im, bbox, expected_label, crop_size):
        """
        Randomly crops the image and updates the bounding box accordingly.

        Args:
            im: Input image.
            bbox: Bounding box.
            expected_label: Expected label.
            crop_size: Size of the crop.

        Returns:
            cropped_im: Cropped image.
            cropped_bbox: Cropped bounding box.
        """
        generate_sample = True
        cropped_im, cropped_bbox = im, bbox
        while generate_sample:
            cropped_im, cropped_bbox = random_crop_and_update_bbox(im, bbox, crop_size)
            image_bbox = torch.tensor([0, 0, cropped_im.shape[2], cropped_im.shape[1]])
            iou = IoU(image_bbox, cropped_bbox)
            if (iou >= self.pos_th and expected_label) or (iou <= self.neg_th and not expected_label):
                break
            crop_size[0] = max(int((crop_size[0] * 0.9)), 60)
            crop_size[1] = max(int((crop_size[1] * 0.9)), 60)
        return cropped_im, cropped_bbox

    def __generate_sample(self, label: bool):
        """
        Generates a sample.

        Args:
            label (bool): Label of the sample.

        Returns:
            im: Image.
            bbox: Bounding box.
        """
        crop_size = np.random.randint(self.min_crop, self.max_crop, size=1)[0]
        crop_size = [crop_size, crop_size]
        item = np.random.randint(0, len(self.data_split))
        sample_name = self.data_split.iloc[item]
        bbox = torch.tensor(list(self.bbox[self.bbox['image_name'] == sample_name].to_numpy()[0][2:-1]))
        im = np.array(Image.open(os.path.join(self.path, "images", sample_name)))
        if self.transform is not None:
            im = self.transform(im)
        im, bbox = self.__random_crop_image_and_bbox(im, bbox, label, crop_size)
        crop_size = [im.shape[1], im.shape[1]]
        im = self.resize_transform(im)
        bbox = bbox * 1. / float(crop_size[0])
        return im, bbox

    def __generate_hard_samples(self, label: bool):
        """
        Generates hard samples.

        Args:
            label (bool): Label of the sample.

        Returns:
            im: Image.
            bbox: Bounding box.
        """
        while True:
            crop_size = np.random.randint(self.min_crop, self.max_crop, size=1)[0]
            crop_size = [crop_size, crop_size]
            item = np.random.randint(0, len(self.data_split))
            sample_name = self.data_split.iloc[item]
            bbox = torch.tensor(list(self.bbox[self.bbox['image_name'] == sample_name].to_numpy()[0][2:-1]))
            im = np.array(Image.open(os.path.join(self.path, "images", sample_name)))
            if self.transform is not None:
                im = self.transform(im)
            im, bbox = self.__random_crop_image_and_bbox(im, bbox, label, crop_size)
            if self.previous_net is not None:
                previous_net_im = torch.unsqueeze(self.previous_transform(im), 0)
                previous_net_bbox = torch.round(bbox * self.out_size[0] / crop_size[0])
                previous_net_out = self.previous_net(previous_net_im)
                previous_net_label, previous_net_bbox = previous_net_out["y_pred"][0].argmax(), previous_net_out[
                    "bbox_pred"]
                if previous_net_label != int(label):
                    im = self.resize_transform(im)
                    bbox = bbox * 1. / float(crop_size[0])
                    break
        return im, bbox

    def __create_data(self):
        """
        Creates the dataset.
        """
        for i in tqdm(range(self.n // 2), desc="Generate positive samples", total=self.n // 2):
            im, bbox = self.__generate_sample(True)
            self.labels.append(torch.tensor(1, dtype=torch.long))
            self.bboxes.append(bbox)
            self.images.append(im)

        for i in tqdm(range(self.n // 2), desc="Generate negative samples", total=self.n // 2):
            im, bbox = self.__generate_sample(False)
            self.labels.append(torch.tensor(0, dtype=torch.long))
            self.bboxes.append(bbox)
            self.images.append(im)

        for i in tqdm(range(self.n_hard // 2), desc="Generate positive hard samples", total=self.n_hard // 2):
            im, bbox = self.__generate_hard_samples(True)
            self.labels.append(torch.tensor(1, dtype=torch.long))
            self.bboxes.append(bbox)
            self.images.append(im)

        for i in tqdm(range(self.n_hard // 2), desc="Generate negative hard samples", total=self.n_hard // 2):
            im, bbox = self.__generate_hard_samples(False)
            self.labels.append(torch.tensor(0, dtype=torch.long))
            self.bboxes.append(bbox)
            self.images.append(im)

    def __getitem__(self, item):
        """
        Gets the item at the specified index.

        Args:
            item: Index of the item.

        Returns:
            Image, bounding box, and label.
        """
        if 0 <= item < self.__len__():
            return self.images[item], self.bboxes[item], self.labels[item]

    def __len__(self):
        """
        Gets the length of the dataset.

        Returns:
            Length of the dataset.
        """
        return len(self.labels)


class CelebA(Dataset):
    """
    Dataset class for CelebA dataset.
    """

    def __init__(self, path: str, partition: str, transform=None) -> Dataset:
        """
        Initializes the CelebA dataset.

        Args:
            path (str): Path to the dataset.
            partition (str): Dataset partition ('train', 'val', or 'test').
            transform (callable, optional): Optional transform to be applied on the sample.
        """
        super(CelebA, self).__init__()
        self.path = path
        data_partition = pd.read_csv(os.path.join(path, "list_eval_partition.csv"), index_col=False)
        self.bbox = pd.read_csv(os.path.join(path, "list_bbox_celeba.csv"), index_col=False)
        if partition.lower() == "train":
            partition = 0
        elif partition.lower() == "val":
            partition = 1
        elif partition.lower() == "test":
            partition = 2
        else:
            raise ValueError(f"Unknown partition {partition}")

        self.data_split = data_partition[data_partition['partition'] == partition].image_id
        self.transform = transform

    def __getitem__(self, item):
        """
        Gets the item at the specified index.

        Args:
            item: Index of the item.

        Returns:
            Image.
        """
        if 0 <= item < self.__len__():
            sample_name = self.data_split.iloc[item]
            info = torch.tensor([float(x) for x in self.bbox[self.bbox['image_id'] == sample_name].to_numpy()[0][1:]])
            im = np.array(Image.open(os.path.join(self.path, "images", sample_name)))
            if self.transform is not None:
                im = self.transform(im)
            return im

    def __len__(self):
        """
        Gets the length of the dataset.

        Returns:
            Length of the dataset.
        """
        return len(self.data_split)
