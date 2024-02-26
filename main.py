import numpy as np
from model import PNet, RNet
from datasets import PNetDataset, FacesDataSet
from torchvision.transforms import ToTensor, Compose, Resize
from trainer import train_pnet
import torch
from utils import plot_im_with_bbox, make_image_pyramid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def test_propose_net():
    transform = Compose([ToTensor()])

    pnet = PNet()
    # Load the checkpoint
    checkpoint = torch.load('pnet_training/checkpoint/last_epoch_checkpoint_200.pth')
    # Load the model state dictionary
    pnet.load_state_dict(checkpoint)
    pnet.eval()
    resize = Resize(size=(12, 12), antialias=True)
    dataset = FacesDataSet(path="data/celebA", partition="test", transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=1)

    for im in dataloader:
        image_pyramid = make_image_pyramid(im)
        bboxes = []
        orig_x, orig_y = im.shape[3], im.shape[2]
        for scaled_im in image_pyramid:
            scaled_im = resize(scaled_im)
            out = pnet(scaled_im)
            y, bbox = out["y_pred"], out["bbox_pred"]
            bbox[0][0] = bbox[0][0] * orig_x / float(12)
            bbox[0][2] = bbox[0][2] * orig_x / float(12)
            bbox[0][1] = bbox[0][1] * orig_y / float(12)
            bbox[0][3] = bbox[0][3] * orig_y / float(12)
            bboxes.append(bbox.detach()[0])
        plot_im_with_bbox(im[0], bboxes, scores=None, iou_threshold=0.6)


if __name__ == "__main__":
    pass
    # test_propose_net()
    # transform = Compose([ToTensor()])
    # train_dataset = PNetDataset(path="data/celebA", partition="train", transform=transform, min_crop=20, max_crop=140, n=10000)
    # val_dataset = PNetDataset(path="data/celebA", partition="val", transform=transform, min_crop=20, max_crop=140, n=1000)
    # train_params = {
    #     "lr": 1e-3,
    #     "optimizer": "adam",
    #     "n_epochs": 200,
    #     "batch_size": 128,
    # }
    # pnet = PNet()
    # Load the checkpoint
    # checkpoint = torch.load('pnet_training/checkpoint/checkpoint_epoch_100.pth')
    # Load the model state dictionary
    # pnet.load_state_dict(checkpoint)
    # rnet = RNet()
    #
    # train_pnet(pnet=pnet, train_dataset=train_dataset, val_dataset=val_dataset, train_params=train_params,
    #            out_dir="pnet_training", checkpoint_step=10, device="cuda")

