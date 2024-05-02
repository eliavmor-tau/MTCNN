import copy

from torch.utils.data import Dataset
import pandas as pd
import torch
import os
from utils import random_crop_and_update_bbox, plot_im_with_bbox, IoU
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor, Resize, Compose
from tqdm import tqdm
from model import PNet, RNet


class MTCNNDataset(Dataset):

    def __init__(self, path: str, partition: str, transform=None, neg_th: int = 0.3, pos_th: int = 0.65,
                 min_crop: int = 12, max_crop: int = 100, out_size=24, n=1000, n_hard=1000, previous_net=None,
                 previous_transform=None) -> Dataset:
        super(MTCNNDataset, self).__init__()
        self.previous_net = previous_net
        self.previous_transform = previous_transform
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
        print(f"len(self.data_split) = {len(self.data_split)}")
        self.n = n
        self.n_hard = n_hard
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
            crop_size[0] = max(int((crop_size[0] * 0.9)), 60)
            crop_size[1] = max(int((crop_size[1] * 0.9)), 60)
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
        # bbox = torch.round(bbox * self.out_size / crop_size[0])
        bbox = bbox * 1. / float(crop_size[0])
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


class CelebA(Dataset):

    def __init__(self, path: str, partition: str, transform=None) -> Dataset:
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
            raise (f"unkonwn partition {partition}")

        self.data_split = data_partition[data_partition['partition'] == partition].image_id
        self.transform = transform

    def __getitem__(self, item):
        if 0 <= item < self.__len__():
            sample_name = self.data_split.iloc[item]
            info = torch.tensor([float(x) for x in self.bbox[self.bbox['image_id'] == sample_name].to_numpy()[0][1:]])
            im = np.array(Image.open(os.path.join(self.path, "images", sample_name)))
            if self.transform is not None:
                im = self.transform(im)
            # return im, info
            return im

    def __len__(self):
        return len(self.data_split)


class MTCNNWiderFace(Dataset):
    def __init__(self, path: str, partition: str, transform=None, neg_th: int = 0.3, pos_th: int = 0.65,
                 min_crop: int = 12, out_size=24, n=1000, n_hard=1000, previous_net=None,
                 previous_transform=None) -> Dataset:
        super(MTCNNWiderFace, self).__init__()
        partition = partition.lower()
        self.path = path
        self.data_path = os.path.join(path, f"WIDER_{partition}", "images")
        self.bbox = []
        self.im_paths = []
        with open(os.path.join(self.path, f"wider_face_split/wider_face_{partition}_bbx_gt.txt"), "r") as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            img_path = lines[i].strip("\n")
            if img_path.endswith(".jpg"):
                i += 1
                num_of_faces = int(lines[i].strip("\n"))
                i += 1
                bboxes = []
                for j in range(num_of_faces):
                    bbox = torch.tensor(
                        np.array([float(x) for x in lines[i].strip("\n").split(" ")[:4]]).reshape((1, -1)))
                    bboxes.append(bbox)
                    i += 1
                im_path = os.path.join(self.data_path, img_path)
                if 0 < num_of_faces and os.path.isfile(im_path):
                    self.im_paths.append(im_path)
                    self.bbox.append(bboxes)
            else:
                i += 1

        self.N = len(self.im_paths)
        self.transform = transform
        self.previous_net = previous_net
        self.previous_transform = previous_transform
        self.min_crop = min_crop
        self.neg_th = neg_th
        self.pos_th = pos_th
        self.out_size = out_size
        self.resize_transform = Resize(out_size, antialias=True)
        self.transform = transform
        self.n = n
        self.n_hard = n_hard
        self.labels = []
        self.bboxes = []
        self.images = []
        self.__create_data()

    def __prepare_pos_example(self, im, box):
        if len(box.shape) == 1:
            box = box.reshape((-1, 4))
        h, w = im.shape[1], im.shape[2]
        x, y, bbox_w, bbox_h = box[0]
        max_delta = min(bbox_w, bbox_h) * 1
        crop_box = copy.deepcopy(box)
        crop_im = copy.deepcopy(im)
        size = w
        status = False
        for i in range(10):
            delta = np.random.randint(-max_delta, max_delta)
            nx = max(0, x + delta)
            ny = max(0, y + delta)
            size = max(bbox_w, bbox_h)
            if nx + size - w > 0 or ny + size - h > 0:
                size -= max(nx + size - w, ny + size - h)
            crop_box = torch.tensor(np.array([nx, ny, size, size]), dtype=int).view((-1, 4))
            iou = IoU(crop_box, box)
            if iou >= self.pos_th:
                crop_x, crop_y, crop_w, crop_h = crop_box[0]
                crop_im = im[:, crop_y:crop_y + crop_h, crop_x: crop_x + crop_w]
                crop_box = torch.tensor(
                    np.array([max(x - nx, 0), max(y - ny, 0), min(size - 1, bbox_w), min(size - 1, bbox_h)]),
                    dtype=int).view((-1, 4))
                status = True
                break
        crop_im = self.resize_transform(crop_im)
        crop_box = crop_box * 1. / float(size)
        return crop_im, crop_box, status

    def __prepare_neg_example(self, im, bboxes):
        h, w = im.shape[1], im.shape[2]
        crop_box = np.array([0, 0, w, h]).reshape((1, -1))
        status = False
        for i in range(10):
            size = np.random.randint(self.min_crop, min(w, h) / 2)
            nx = np.random.randint(0, w - size)
            ny = np.random.randint(0, h - size)
            crop_box = torch.tensor(np.array([nx, ny, size, size]).reshape((-1, 4)))
            iou = IoU(crop_box, bboxes)
            if bboxes.shape[0] > 1 and (iou <= self.neg_th).all() or \
                    bboxes.shape[0] == 1 and iou <= self.neg_th:
                status = True
                break
        x, y, w, h = crop_box[0]
        crop_im = im[:, y:y + h, x: x + w]
        crop_im = self.resize_transform(crop_im)
        return crop_im, status

    def __create_data(self):
        i = 0
        neg_count, pos_count = 0, 0
        while neg_count < self.n // 2:
            idx = i % self.N
            bbox = self.bbox[i % self.N]
            im = np.array(Image.open(self.im_paths[idx]))
            if self.transform is not None:
                im = self.transform(im)
            bbox = torch.tensor(np.concatenate(bbox, axis=0))
            status = False
            while not status:
                neg_example, status = self.__prepare_neg_example(im, bboxes=bbox)
                if status:
                    self.images.append(neg_example)
                    self.bboxes.append(torch.zeros((1, 4)))
                    self.labels.append(torch.tensor(0))
                    neg_count += 1
                    i += 1
        i = 0
        while pos_count < self.n // 2:
            idx = i % self.N
            bboxes = self.bbox[i % self.N]
            im = np.array(Image.open(self.im_paths[idx]))
            if self.transform is not None:
                im = self.transform(im)
            bboxes = torch.tensor(np.concatenate(bboxes, axis=0))
            for box in bboxes:
                status = False
                if min(box[2], box[3]) > 80:
                    tries = 0
                    while not status:
                        pos_example, new_box, status = self.__prepare_pos_example(im, box)
                        if status == True:
                            self.images.append(pos_example)
                            self.bboxes.append(new_box)
                            self.labels.append(torch.tensor(1))
                            pos_count += 1
                        else:
                            tries += 1
                        if tries >= 3:
                            break
                if pos_count >= self.n // 2:
                    break
            i += 1

    def __getitem__(self, item):
        if 0 <= item < self.__len__():
            return self.images[item], self.bboxes[item], self.labels[item]

    def __len__(self):
        return len(self.labels)


class WiderFace(Dataset):
    def __init__(self, path: str, partition: str, transform=None) -> Dataset:
        super(WiderFace, self).__init__()
        partition = partition.lower()
        self.path = path
        self.data_path = os.path.join(path, f"WIDER_{partition}", "images")
        self.bbox = []
        self.im_paths = []
        with open(os.path.join(self.path, f"wider_face_split/wider_face_{partition}_bbx_gt.txt"), "r") as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            img_path = lines[i].strip("\n")
            if img_path.endswith(".jpg"):
                i += 1
                num_of_faces = int(lines[i].strip("\n"))
                i += 1
                bboxes = []
                for j in range(num_of_faces):
                    bbox = torch.tensor(
                        np.array([float(x) for x in lines[i].strip("\n").split(" ")[:4]]).reshape((1, -1)))
                    # print(bbox.dtype)
                    # exit(0)
                    bboxes.append(bbox)
                    i += 1
                if 0 < num_of_faces:
                    self.im_paths.append(os.path.join(self.data_path, img_path))
                    self.bbox.append(bboxes)
            else:
                i += 1
        self.transform = transform

    def __getitem__(self, item):
        if 0 <= item < self.__len__():
            bbox = self.bbox[item]
            im = np.array(Image.open(self.im_paths[item]))
            if self.transform is not None:
                im = self.transform(im)
            return im, bbox

    def __len__(self):
        return len(self.im_paths)


# if __name__ == "__main__":
#     transform = Compose([ToTensor()])
#     #
#     # # test dataset.
#     # dataset = MTCNNDataset(path="data/celebA", partition="train", transform=transform, min_crop=100, max_crop=180,
#     #                        # n=20000, n_hard=0, out_size=(12, 12))
#     #                        n=100, n_hard=0, out_size=(12, 12))
#     # for i in range(len(dataset)):
#     #     im, bbox, label = dataset[i]
#     #     plot_im_with_bbox(im, [bbox * 12], title=f"image={i} label={label}")
#     # dataset = WiderFace(path="data/wider_face", partition="train")
#     out_size = 48
#     dataset = MTCNNWiderFace(path="data/wider_face", partition="train", transform=transform, neg_th=0.3, pos_th=0.65,
#                              min_crop=12, out_size=(out_size, out_size), n=20, n_hard=0)
#     for i in range(len(dataset)):
#         im, bbox, label = dataset[i]
#         bbox *= out_size
#         plot_im_with_bbox(im, [bbox], title=f"image={i}")
