import os

import numpy as np
from model import PNet, RNet, ONet
from datasets import PNetDataset, FacesDataSet, RNetDataset, ONetDataset
from torchvision.transforms import ToTensor, Compose, Resize
from trainer import train
import torch
from utils import plot_im_with_bbox, make_image_pyramid, nms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def test_propose_net():
    transform = Compose([ToTensor()])

    pnet = PNet()
    # Load the checkpoint
    checkpoint = torch.load('pnet_training_3/checkpoint/last_epoch_checkpoint_200.pth')
    # checkpoint = torch.load('pnet_training_2/checkpoint/last_epoch_checkpoint_200.pth')
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
            # bbox[0][0] = bbox[0][0] * orig_x / float(12)
            # bbox[0][2] = bbox[0][2] * orig_x / float(12)
            # bbox[0][1] = bbox[0][1] * orig_y / float(12)
            # bbox[0][3] = bbox[0][3] * orig_y / float(12)
            bbox[0][0] = bbox[0][0] * orig_x
            bbox[0][2] = bbox[0][2] * orig_x
            bbox[0][1] = bbox[0][1] * orig_y
            bbox[0][3] = bbox[0][3] * orig_y
            bboxes.append(bbox.detach()[0])
        plot_im_with_bbox(im[0], bboxes, scores=None, iou_threshold=0.6)


def test_residual_net():
    transform = Compose([ToTensor()])
    rnet = RNet()
    # Load the checkpoint
    checkpoint = torch.load('rnet_training/checkpoint/checkpoint_epoch_30.pth')
    # Load the model state dictionary
    rnet.load_state_dict(checkpoint)
    rnet.eval()
    resize = Resize(size=(24, 24), antialias=True)
    dataset = FacesDataSet(path="data/celebA", partition="val", transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=1)

    for im in dataloader:
        image_pyramid = make_image_pyramid(im)
        bboxes = []
        orig_x, orig_y = im.shape[3], im.shape[2]
        for scaled_im in image_pyramid:
            scaled_im = resize(scaled_im)
            out = rnet(scaled_im)
            y, bbox = out["y_pred"], out["bbox_pred"]
            bbox[0][0] = bbox[0][0] * orig_x
            bbox[0][2] = bbox[0][2] * orig_x
            bbox[0][1] = bbox[0][1] * orig_y
            bbox[0][3] = bbox[0][3] * orig_y
            bboxes.append(bbox.detach()[0])
        plot_im_with_bbox(im[0], bboxes, scores=None, iou_threshold=0.2)


def test_onet():
    transform = Compose([ToTensor()])
    onet = ONet()
    checkpoint = torch.load('onet_training/checkpoint/checkpoint_epoch_30.pth')
    onet.load_state_dict(checkpoint)
    onet.eval()
    resize = Resize(size=(48, 48), antialias=True)
    dataset = FacesDataSet(path="data/celebA", partition="test", transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=1)

    for im in dataloader:
        image_pyramid = make_image_pyramid(im)
        bboxes = []
        orig_x, orig_y = im.shape[3], im.shape[2]
        for scaled_im in image_pyramid:
            scaled_im = resize(scaled_im)
            out = onet(scaled_im)
            y, bbox = out["y_pred"], out["bbox_pred"]
            bbox[0][0] = bbox[0][0] * orig_x
            bbox[0][2] = bbox[0][2] * orig_x
            bbox[0][1] = bbox[0][1] * orig_y
            bbox[0][3] = bbox[0][3] * orig_y
            bboxes.append(bbox.detach()[0])
        plot_im_with_bbox(im[0], bboxes, scores=None, iou_threshold=0.2)


def test():
    import matplotlib.patches as patches
    pnet = PNet()
    pnet_resize = Resize(size=(12, 12), antialias=True)
    checkpoint = torch.load('pnet_training_3/checkpoint/last_epoch_checkpoint_200.pth')
    pnet.load_state_dict(checkpoint)
    pnet.eval()

    rnet = RNet()
    rnet_resize = Resize(size=(24, 24), antialias=True)
    checkpoint = torch.load('rnet_training/checkpoint/checkpoint_epoch_30.pth')
    rnet.load_state_dict(checkpoint)
    rnet.eval()

    onet = ONet()
    onet_resize = Resize(size=(48, 48), antialias=True)
    checkpoint = torch.load('onet_training/checkpoint/checkpoint_epoch_30.pth')
    onet.load_state_dict(checkpoint)
    onet.eval()

    transform = Compose([ToTensor()])
    dataset = FacesDataSet(path="data/celebA", partition="test", transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=1)

    for im in dataloader:
        # fig, axis = plt.subplots()
        # axis.imshow(np.transpose(im[0], axes=(1, 2, 0)))
        final_bboxes, final_scores = [], []
        image_pyramid = make_image_pyramid(im)
        orig_x, orig_y = im.shape[3], im.shape[2]
        pnet_candidates, pnet_candidates_params, pnet_bboxes, pnet_scores = [], [], [], []
        for scaled_im in image_pyramid:
            scaled_im = pnet_resize(scaled_im)
            out = pnet(scaled_im)
            y, bbox = out["y_pred"], out["bbox_pred"]
            y = torch.exp(y)
            y = y / y.sum()
            # if torch.argmax(y) == 1:
            bbox[0][0] = bbox[0][0] * orig_x
            bbox[0][2] = bbox[0][2] * orig_x
            bbox[0][1] = bbox[0][1] * orig_y
            bbox[0][3] = bbox[0][3] * orig_y
            pnet_bboxes.append(torch.round(bbox))
            pnet_scores.append(y[0][1])

        if len(pnet_bboxes):
            pnet_bboxes = torch.vstack(pnet_bboxes)
            pnet_scores = torch.tensor(pnet_scores)
            bboxes_indices = nms(boxes=pnet_bboxes, scores=pnet_scores, iou_threshold=0.3)
            pnet_bboxes = [pnet_bboxes[index] for index in bboxes_indices]
            pnet_scores = torch.tensor([pnet_scores[index] for index in bboxes_indices])
            # plot_im_with_bbox(im[0], bboxes=pnet_bboxes, scores=pnet_scores, iou_threshold=0.3)
            for bbox in pnet_bboxes:
                x, y, w, h = bbox[0].long(), bbox[1].long(), bbox[2].long(), bbox[3].long()
                x = torch.clip(x, min=0, max=orig_x)
                y = torch.clip(y, min=0, max=orig_y)
                pnet_candidates.append(im[:, :, y: y + h, x: x + w])
                pnet_candidates_params.append((x, y, w, h))
                # rec = patches.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2], height=bbox[3], linewidth=2,
                #                         edgecolor='blue',
                #                         facecolor='none')
                # axis.add_patch(rec)
                # plt.imshow(np.transpose(pnet_candidates[-1][0], axes=(1, 2, 0)))
                # plt.show()
                # plt.close()

            for pnet_candidate_idx, pnet_candidate in enumerate(pnet_candidates):
                rnet_candidates, rnet_candidates_params, rnet_bboxes, rnet_scores = [], [], [], []
                image_pyramid = make_image_pyramid(pnet_candidate)
                input_x, input_y = pnet_candidate.shape[3], pnet_candidate.shape[2]
                for scaled_im in image_pyramid:
                    scaled_im = rnet_resize(scaled_im)
                    out = rnet(scaled_im)
                    y, bbox = out["y_pred"], out["bbox_pred"]
                    y = torch.exp(y)
                    y = y / y.sum()
                    if torch.argmax(y) == 1:
                        bbox[0][0] = torch.clip(bbox[0][0] * input_x, 0, input_x)
                        bbox[0][2] = torch.clip(bbox[0][2] * input_x, 0, input_x)
                        bbox[0][1] = torch.clip(bbox[0][1] * input_y, 0, input_y)
                        bbox[0][3] = torch.clip(bbox[0][3] * input_y, 0, input_y)
                        rnet_bboxes.append(torch.round(bbox))
                        rnet_scores.append(y[0][1])

                if len(rnet_bboxes):
                    rnet_bboxes = torch.vstack(rnet_bboxes)
                    rnet_scores = torch.tensor(rnet_scores)
                    bboxes_indices = nms(boxes=rnet_bboxes, scores=rnet_scores, iou_threshold=0.3)
                    rnet_bboxes = [rnet_bboxes[index] for index in bboxes_indices]
                    rnet_scores = torch.tensor([rnet_scores[index] for index in bboxes_indices])
                    # plot_im_with_bbox(pnet_candidate[0], bboxes=rnet_bboxes, scores=rnet_scores, iou_threshold=0.3)
                    yx_offset = pnet_candidates_params[pnet_candidate_idx]
                    for bbox in rnet_bboxes:
                        x, y, w, h = bbox[0].long(), bbox[1].long(), bbox[2].long(), bbox[3].long()
                        rnet_candidates.append(pnet_candidate[:, :, y: y + h, x: x + w])
                        rnet_candidates_params.append((x + yx_offset[0], y + yx_offset[1], w, h))
                        # rec = patches.Rectangle(xy=(x + yx_offset[0], y + yx_offset[1]), width=bbox[2], height=bbox[3], linewidth=2,
                        #                         edgecolor='green',
                        #                         facecolor='none')
                        # axis.add_patch(rec)

                    for rnet_candidate_idx, rnet_candidate in enumerate(rnet_candidates):
                        onet_candidates_offsets, onet_bboxes, onet_scores = [], [], []
                        image_pyramid = make_image_pyramid(rnet_candidate)
                        input_x, input_y = rnet_candidate.shape[3], rnet_candidate.shape[2]
                        for scaled_im in image_pyramid:
                            scaled_im = onet_resize(scaled_im)
                            out = onet(scaled_im)
                            y, bbox = out["y_pred"], out["bbox_pred"]
                            y = torch.exp(y)
                            y = y / y.sum()
                            if torch.argmax(y) == 1:
                                bbox[0][0] = torch.clip(bbox[0][0] * input_x, 0, input_x)
                                bbox[0][2] = torch.clip(bbox[0][2] * input_x, 0, input_x)
                                bbox[0][1] = torch.clip(bbox[0][1] * input_y, 0, input_y)
                                bbox[0][3] = torch.clip(bbox[0][3] * input_y, 0, input_y)
                                onet_bboxes.append(torch.round(bbox))
                                onet_scores.append(y[0][1])

                        if len(onet_bboxes):
                            onet_bboxes = torch.vstack(onet_bboxes)
                            onet_scores = torch.hstack(onet_scores)
                            bboxes_indices = nms(boxes=onet_bboxes, scores=onet_scores, iou_threshold=0.3)
                            onet_bboxes = [onet_bboxes[index] for index in bboxes_indices]
                            onet_scores = torch.tensor([onet_scores[index] for index in bboxes_indices])
                            # plot_im_with_bbox(rnet_candidate[0], bboxes=onet_bboxes, scores=onet_scores,
                            #                   iou_threshold=0.3)
                            yx_offset = rnet_candidates_params[rnet_candidate_idx]
                            for idx, bbox in enumerate(onet_bboxes):
                                x, y, w, h = bbox[0].long(), bbox[1].long(), bbox[2].long(), bbox[3].long()
                                onet_candidates_offsets.append((x + yx_offset[0], y + yx_offset[1], w, h))
                                final_bboxes.append(torch.tensor([x + yx_offset[0], y + yx_offset[1], w, h]).float())
                                final_scores.append(onet_scores[idx])
                                # rec = patches.Rectangle(xy=(x + yx_offset[0], y + yx_offset[1]), width=bbox[2],
                                #                         height=bbox[3], linewidth=2,
                                #                         edgecolor='red',
                                #                         facecolor='none')
                                # axis.add_patch(rec)
        final_scores = torch.tensor(final_scores).float()
        plot_im_with_bbox(im[0], bboxes=final_bboxes, scores=final_scores, iou_threshold=0.2)


def run_train_pnet():
    transform = Compose([ToTensor()])
    train_dataset = PNetDataset(path="data/celebA", partition="train", transform=transform, min_crop=100, max_crop=180,
                                n=10000)
    val_dataset = PNetDataset(path="data/celebA", partition="val", transform=transform, min_crop=100, max_crop=180,
                              n=1000)
    train_params = {
        "lr": 1e-3,
        "optimizer": "adam",
        "n_epochs": 200,
        "batch_size": 128,
    }
    pnet = PNet()
    # checkpoint = torch.load('pnet_training_2/checkpoint/last_epoch_checkpoint_100.pth')
    # checkpoint = torch.load('pnet_training/checkpoint/checkpoint_epoch_150.pth')
    # pnet.load_state_dict(checkpoint)

    train(net=pnet, train_dataset=train_dataset, val_dataset=val_dataset, train_params=train_params,
          out_dir="pnet_training_3", checkpoint_step=10, device="cuda", wd=1e-3)


def run_train_rnet():
    transform = Compose([ToTensor()])
    pnet = PNet()
    checkpoint = torch.load('pnet_training_3/checkpoint/last_epoch_checkpoint_200.pth')
    pnet.load_state_dict(checkpoint)
    train_dataset = RNetDataset(pnet=pnet, path="data/celebA", partition="train", transform=transform,
                                min_crop=100, max_crop=180, n=10000, n_hard=1000, out_size=24)
    val_dataset = RNetDataset(pnet=pnet, path="data/celebA", partition="val", transform=transform, min_crop=100,
                              max_crop=180, n=1000, n_hard=0, out_size=24)

    train_params = {
        "lr": 1e-3,
        "optimizer": "adam",
        "n_epochs": 100,
        "batch_size": 128,
    }
    rnet = RNet()
    device = "cuda"

    def lr_step(epoch):
        if epoch <= 30:
            return 1
        else:
            return 0.1

    train(net=rnet, train_dataset=train_dataset, val_dataset=val_dataset, train_params=train_params,
          out_dir="rnet_training_3", checkpoint_step=10, lr_step=lr_step, device=device, weights=[1.0, 1.0], wd=1e-3)

    # if device == "cuda" and not torch.cuda.is_available():
    #     device = "cpu"

    # device = torch.device(device=device)
    # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    # rnet.eval()
    # os.makedirs("figures", exist_ok=True)
    # with torch.no_grad():
    #     for idx, batch in enumerate(train_dataloader):
    #         images, bboxes, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
    #         out = rnet(images)
    #         pred_bboxes = out["bbox_pred"]
    #         y_pred = out["y_pred"]
    #         plot_im_with_bbox(images[0], [pred_bboxes[0] * 24], title=f"train y={y} y_pred={y_pred[0].argmax().item()}",
    #                           figname=os.path.join("figures", f"train_{idx}.jpg"))
    #
    #     val_dataloader = DataLoader(val_dataset, batch_size=1)
    #     for idx, batch in enumerate(val_dataloader):
    #         images, bboxes, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
    #         out = rnet(images)
    #         pred_bboxes = out["bbox_pred"]
    #         y_pred = out["y_pred"]
    #         plot_im_with_bbox(images[0], [pred_bboxes[0] * 24], title=f"val y={y} y_pred={y_pred[0].argmax().item()}",
    #                           figname=os.path.join("figures", f"val_{idx}.jpg"))


def run_train_onet():
    transform = Compose([ToTensor()])

    pnet = PNet()
    # Load the checkpoint
    checkpoint = torch.load('pnet_training_3/checkpoint/last_epoch_checkpoint_200.pth')

    rnet = RNet()
    # Load the checkpoint
    checkpoint = torch.load('rnet_training/checkpoint/checkpoint_epoch_30.pth')
    # Load the model state dictionary
    rnet.load_state_dict(checkpoint)
    rnet.eval()
    train_dataset = ONetDataset(rnet=rnet, path="data/celebA", partition="train", transform=transform,
                                min_crop=40, max_crop=200, n=10000, n_hard=1000, out_size=48)
    val_dataset = ONetDataset(rnet=rnet, path="data/celebA", partition="val", transform=transform, min_crop=40,
                              max_crop=200, n=1000, n_hard=0, out_size=48)

    train_params = {
        "lr": 1e-3,
        "optimizer": "adam",
        "n_epochs": 100,
        "batch_size": 128,
    }
    onet = ONet()
    device = "cuda"

    def lr_step(epoch):
        if epoch <= 30:
            return 1
        else:
            return 0.1

    train(net=onet, train_dataset=train_dataset, val_dataset=val_dataset, train_params=train_params,
          out_dir="onet_training", checkpoint_step=10, lr_step=lr_step, device=device, weights=[1.0, 1.0], wd=1e-3)


if __name__ == "__main__":
    # test_propose_net()
    # test_residual_net()
    # test_onet()
    test()
    # run_train_pnet()
    # run_train_rnet()
    # run_train_onet()
