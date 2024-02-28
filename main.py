import os

import numpy as np
from model import PNet, RNet
from datasets import PNetDataset, FacesDataSet, RNetDataset
from torchvision.transforms import ToTensor, Compose, Resize
from trainer import train
import torch
from utils import plot_im_with_bbox, make_image_pyramid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def test_propose_net():
    transform = Compose([ToTensor()])

    pnet = PNet()
    # Load the checkpoint
    checkpoint = torch.load('pnet_training/checkpoint/checkpoint_epoch_150.pth')
    # checkpoint = torch.load('pnet_training_2/checkpoint/last_epoch_checkpoint_200.pth')
    # Load the model state dictionary
    pnet.load_state_dict(checkpoint)
    pnet.eval()

    pnet2 = PNet()
    # Load the checkpoint
    # checkpoint = torch.load('pnet_training/checkpoint/last_epoch_checkpoint_200.pth')
    checkpoint = torch.load('pnet_training_2/checkpoint/last_epoch_checkpoint_100.pth')
    # Load the model state dictionary
    pnet2.load_state_dict(checkpoint)
    pnet2.eval()
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
        bboxes = []
        for scaled_im in image_pyramid:
            scaled_im = resize(scaled_im)
            out = pnet2(scaled_im)
            y, bbox = out["y_pred"], out["bbox_pred"]
            bbox[0][0] = bbox[0][0] * orig_x / float(12)
            bbox[0][2] = bbox[0][2] * orig_x / float(12)
            bbox[0][1] = bbox[0][1] * orig_y / float(12)
            bbox[0][3] = bbox[0][3] * orig_y / float(12)
            bboxes.append(bbox.detach()[0])
        plot_im_with_bbox(im[0], bboxes, scores=None, iou_threshold=0.6)


def test_residual_net():
    transform = Compose([ToTensor()])

    pnet = RNet()
    # Load the checkpoint
    checkpoint = torch.load('rnet_training/checkpoint/last_epoch_checkpoint_200.pth')
    # Load the model state dictionary
    pnet.load_state_dict(checkpoint)
    pnet.eval()
    resize = Resize(size=(24, 24), antialias=True)
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
        plot_im_with_bbox(im[0], bboxes, scores=None, iou_threshold=0.6)


def run_train_pnet():
    transform = Compose([ToTensor()])
    train_dataset = PNetDataset(path="data/celebA", partition="train", transform=transform, min_crop=100, max_crop=180,
                                n=10000)
    val_dataset = PNetDataset(path="data/celebA", partition="val", transform=transform, min_crop=100, max_crop=180,
                              n=1000)
    train_params = {
        "lr": 1e-2,
        "optimizer": "adam",
        "n_epochs": 100,
        "batch_size": 128,
    }
    pnet = PNet()
    # checkpoint = torch.load('pnet_training_2/checkpoint/last_epoch_checkpoint_100.pth')
    checkpoint = torch.load('pnet_training/checkpoint/checkpoint_epoch_150.pth')
    pnet.load_state_dict(checkpoint)

    train(net=pnet, train_dataset=train_dataset, val_dataset=val_dataset, train_params=train_params,
          out_dir="pnet_training_2", checkpoint_step=10, device="cuda", wd=1e-3)


def run_train_rnet():
    transform = Compose([ToTensor()])
    pnet = PNet()
    checkpoint = torch.load('pnet_training/checkpoint/checkpoint_epoch_150.pth')
    pnet.load_state_dict(checkpoint)
    train_dataset = RNetDataset(pnet=pnet, path="data/celebA", partition="train", transform=transform,
                                min_crop=100, max_crop=180, n=20, n_hard=20, out_size=24)
    val_dataset = RNetDataset(pnet=pnet, path="data/celebA", partition="val", transform=transform, min_crop=100,
                              max_crop=180, n=20, n_hard=20, out_size=24)

    train_params = {
        "lr": 1e-3,
        "optimizer": "adam",
        "n_epochs": 200,
        "batch_size": 64,
    }
    rnet = RNet()
    device = "cuda"
    train(net=rnet, train_dataset=train_dataset, val_dataset=val_dataset, train_params=train_params,
          out_dir="rnet_training", checkpoint_step=10, device=device, weights=[1.0, 0.5], wd=0)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    device = torch.device(device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    rnet.eval()
    os.makedirs("figures", exist_ok=True)
    with torch.no_grad():
        for idx, batch in enumerate(train_dataloader):
            images, bboxes, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            out = rnet(images)
            pred_bboxes = out["bbox_pred"]
            y_pred = out["y_pred"]
            plot_im_with_bbox(images[0], [pred_bboxes[0] * 24], title=f"train y={y} y_pred={y_pred[0].argmax().item()}",
                              figname=os.path.join("figures", f"train_{idx}.jpg"))

        val_dataloader = DataLoader(val_dataset, batch_size=1)
        for idx, batch in enumerate(val_dataloader):
            images, bboxes, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            out = rnet(images)
            pred_bboxes = out["bbox_pred"]
            y_pred = out["y_pred"]
            plot_im_with_bbox(images[0], [pred_bboxes[0] * 24], title=f"val y={y} y_pred={y_pred[0].argmax().item()}",
                              figname=os.path.join("figures", f"val_{idx}.jpg"))


if __name__ == "__main__":
    # test_propose_net()
    # test_residual_net()
    # run_train_pnet()
    run_train_rnet()
