from torch.utils.data import Dataset
import pandas as pd
import torch
import os
from utils import random_crop_and_update_bbox, plot_im_with_bbox, IoU
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor, Resize
from tqdm import tqdm
from model import PNet


class PNetDataset(Dataset):

    def __init__(self, path: str, partition: str, transform=None, neg_th: int = 0.3, pos_th: int = 0.65,
                 min_crop: int = 12, max_crop: int = 100, out_size=12, n=1000) -> Dataset:
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
        self.out_size = out_size
        self.resize_transform = Resize(out_size, antialias=True)
        self.data_split = data_partition[data_partition['partition'] == partition].image_name
        self.transform = transform
        self.n = min(n, len(self.data_split))
        self.labels = []
        self.bboxes = []
        self.images = []
        self.__create_data()

    def __random_crop_image_and_bbox(self, im, bbox, expected_label, crop_size):
        generate_sample = True
        cropped_im, cropped_bbox = im, bbox
        while generate_sample:
            cropped_im, cropped_bbox = random_crop_and_update_bbox(im, bbox, crop_size)
            image_bbox = torch.tensor([0, 0, cropped_im.shape[2], cropped_im.shape[1]])
            iou = IoU(image_bbox, cropped_bbox)
            if (iou >= self.pos_th and expected_label) or (iou <= self.neg_th and not expected_label):
                break
            crop_size[0] = max(int((crop_size[0] * 0.9)), 20)
            crop_size[1] = max(int((crop_size[0] * 0.9)), 20)
        return cropped_im, cropped_bbox

    def __generate_sample(self, label: bool):
        crop_size = np.random.randint(self.min_crop, self.max_crop, size=1)[0]
        crop_size = [crop_size, crop_size]
        item = np.random.randint(0, len(self.data_split))
        sample_name = self.data_split.iloc[item]
        bbox = torch.tensor(list(self.bbox[self.bbox['image_name'] == sample_name].to_numpy()[0][2:-1]))
        im = np.array(Image.open(os.path.join(self.path, "images", sample_name)))
        if self.transform is not None:
            im = self.transform(im)
        im, bbox = self.__random_crop_image_and_bbox(im, bbox, label, crop_size)
        im = self.resize_transform(im)
        bbox = torch.round(bbox * self.out_size / crop_size[0])
        return im, bbox

    def __create_data(self):
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

    def __getitem__(self, item):
        if 0 <= item < self.__len__():
            return self.images[item], self.bboxes[item], self.labels[item]

    def __len__(self):
        return len(self.labels)


class RNetDataset(Dataset):

    def __init__(self, pnet, path: str, partition: str, transform=None, neg_th: int = 0.3, pos_th: int = 0.65,
                 min_crop: int = 12, max_crop: int = 100, out_size=24, n=1000, n_hard=1000) -> Dataset:
        super(RNetDataset, self).__init__()
        self.pnet = pnet
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
        self.out_size = out_size
        self.resize_transform = Resize(out_size, antialias=True)
        self.resize12 = Resize((12, 12), antialias=True)
        self.data_split = data_partition[data_partition['partition'] == partition].image_name
        self.transform = transform
        print(f"len(self.data_split) = {len(self.data_split)}")
        self.n = min(n, len(self.data_split))
        self.n_hard = min(n_hard, len(self.data_split))
        self.labels = []
        self.bboxes = []
        self.images = []
        self.__create_data()

    def __random_crop_image_and_bbox(self, im, bbox, expected_label, crop_size):
        generate_sample = True
        cropped_im, cropped_bbox = im, bbox
        while generate_sample:
            cropped_im, cropped_bbox = random_crop_and_update_bbox(im, bbox, crop_size)
            image_bbox = torch.tensor([0, 0, cropped_im.shape[2], cropped_im.shape[1]])
            iou = IoU(image_bbox, cropped_bbox)
            if (iou >= self.pos_th and expected_label) or (iou <= self.neg_th and not expected_label):
                break
            crop_size[0] = max(int((crop_size[0] * 0.9)), self.min_crop)
            crop_size[1] = max(int((crop_size[1] * 0.9)), self.min_crop)
        return cropped_im, cropped_bbox

    def __generate_sample(self, label: bool):
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
        bbox = torch.round(bbox * self.out_size / crop_size[0])
        return im, bbox

    def __generate_hard_samples(self, label: bool):
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

            pnet_im = torch.unsqueeze(self.resize12(im), 0)
            pnet_bbox = torch.round(bbox * self.out_size / crop_size[0])
            pnet_out = self.pnet(pnet_im)
            pnet_label, pnet_bbox = pnet_out["y_pred"][0].argmax(), pnet_out["bbox_pred"]
            if pnet_label != int(label):
                # print("hard found hard sample")
                # print("bbox", bbox * 12 / crop_size[0])
                # print("pnet_bbox[0]", pnet_bbox[0])
                # plot_im_with_bbox(pnet_im[0], [pnet_bbox[0], bbox * 12 / crop_size[0]],
                #                   title=f"hard example pnet_label={pnet_label} expected_label={label}", iou_threshold=0)
                im = self.resize_transform(im)
                bbox = torch.round(bbox * self.out_size / crop_size[0])
                break
        return im, bbox

    def __create_data(self):
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
        if 0 <= item < self.__len__():
            return self.images[item], self.bboxes[item], self.labels[item]

    def __len__(self):
        return len(self.labels)


class FacesDataSet(Dataset):

    def __init__(self, path: str, partition: str, transform=None) -> Dataset:
        super(FacesDataSet, self).__init__()
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

        self.data_split = data_partition[data_partition['partition'] == partition].image_name
        self.transform = transform

    def __getitem__(self, item):
        if 0 <= item < self.__len__():
            sample_name = self.data_split.iloc[item]
            bbox = torch.tensor(list(self.bbox[self.bbox['image_name'] == sample_name].to_numpy()[0][2:-1]))
            im = np.array(Image.open(os.path.join(self.path, "images", sample_name)))
            if self.transform is not None:
                im = self.transform(im)
            return im

    def __len__(self):
        return len(self.data_split)


if __name__ == "__main__":
    pnet = PNet()
    # Load the checkpoint
    checkpoint = torch.load('pnet_training/checkpoint/checkpoint_epoch_150.pth')
    # Load the model state dictionary
    pnet.load_state_dict(checkpoint)
    pnet.eval()

    # test dataset.
    # dataset = PNetDataset(path="data/celebA", partition="val",
    #                       transform=ToTensor(), min_crop=30, max_crop=100,
    #                       out_size=12, n=100)

    # test dataset.
    dataset = RNetDataset(pnet=pnet, path="data/celebA", partition="train",
                          transform=ToTensor(), min_crop=80, max_crop=100,
                          out_size=24, n=0, n_hard=100)
    for i in range(len(dataset)):
        im, bbox, label = dataset[i]
        plot_im_with_bbox(im, [bbox], title=f"image={i} label={label}")
