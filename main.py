import numpy as np

from model import PNet
from datasets import PNetDataset, FacesDataSet
from torchvision.transforms import ToTensor, Compose, Resize
from trainer import train_pnet, view_pnet_predictions
import torch
from utils import plot_im_with_bbox, make_image_pyramid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def test_propose_net():
    transform = Compose([ToTensor()])

    pnet = PNet()
    # Load the checkpoint
    checkpoint = torch.load('pnet/checkpoint/checkpoint_epoch_600.pth')
    # Load the model state dictionary
    pnet.load_state_dict(checkpoint)
    pnet.eval()
    resize = Resize(size=(12, 12), antialias=True)
    dataset = FacesDataSet(path="data/celebA", partition="train", transform=transform)
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
        print(bboxes)
        plot_im_with_bbox(im[0], bboxes)


if __name__ == "__main__":
    # test_propose_net()
    transform = Compose([ToTensor()])
    train_dataset = PNetDataset(path="data/celebA", partition="train", transform=transform, min_crop=20, max_crop=140)
    val_dataset = PNetDataset(path="data/celebA", partition="val", transform=transform, min_crop=20, max_crop=140)
    train_params = {
        "lr": 1e-3,
        "optimizer": "adam",
        "n_epochs": 500,
        "batch_size": 16,
    }
    pnet = PNet()
    # Load the checkpoint
    checkpoint = torch.load('pnet/checkpoint/checkpoint_epoch_600.pth')
    # Load the model state dictionary
    pnet.load_state_dict(checkpoint)
    train_pnet(pnet=pnet, train_dataset=train_dataset, val_dataset=val_dataset, train_params=train_params,
               out_dir="pnet_retrain", checkpoint_step=50, device="cuda")

    # view_pnet_predictions(pnet, train_dataset)
