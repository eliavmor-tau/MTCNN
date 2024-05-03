import os

import numpy as np
from model import PNet, RNet, ONet
from datasets import MTCNNDataset, CelebA, MTCNNWiderFace
from torchvision.transforms import ToTensor, Compose, Resize
from trainer import train
import torch
from utils import plot_im_with_bbox, make_image_pyramid, nms, IoU
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2


def test_propose_net():
    transform = Compose([ToTensor()])

    pnet = PNet()
    # Load the checkpoint
    checkpoint = torch.load('pnet_training_large_celeba/checkpoint/best_checkpoint.pth')
    # Load the model state dictionary
    pnet.load_state_dict(checkpoint)
    pnet.eval()

    resize = Resize(size=(12, 12), antialias=True)
    dataset = CelebA(path="data/celebA", partition="test", transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=1)

    for im in dataloader:
        image_pyramid = make_image_pyramid(im)
        bboxes = []
        orig_x, orig_y = im.shape[3], im.shape[2]
        for scaled_im in image_pyramid:
            scaled_im = resize(scaled_im)
            out = pnet(scaled_im)
            y, bbox = out["y_pred"], out["bbox_pred"]
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
    checkpoint = torch.load('rnet_training_large_celeba/checkpoint/best_checkpoint.pth')
    # Load the model state dictionary
    rnet.load_state_dict(checkpoint)
    rnet.eval()
    resize = Resize(size=(24, 24), antialias=True)
    dataset = CelebA(path="data/celebA", partition="test", transform=transform)
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
    checkpoint = torch.load('onet_training/checkpoint/last_epoch_checkpoint_200.pth')
    onet.load_state_dict(checkpoint)
    onet.eval()
    resize = Resize(size=(48, 48), antialias=True)
    dataset = CelebA(path="data/celebA", partition="test", transform=transform)
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
    checkpoint = torch.load('pnet_training/checkpoint/last_epoch_checkpoint_200.pth')
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
    dataset = CelebA(path="data/celebA", partition="test", transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=1)

    for im in dataloader:
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
            for bbox in pnet_bboxes:
                x, y, w, h = bbox[0].long(), bbox[1].long(), bbox[2].long(), bbox[3].long()
                x = torch.clip(x, min=0, max=orig_x)
                y = torch.clip(y, min=0, max=orig_y)
                pnet_candidates.append(im[:, :, y: y + h, x: x + w])
                pnet_candidates_params.append((x, y, w, h))

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
                    yx_offset = pnet_candidates_params[pnet_candidate_idx]
                    for bbox in rnet_bboxes:
                        x, y, w, h = bbox[0].long(), bbox[1].long(), bbox[2].long(), bbox[3].long()
                        rnet_candidates.append(pnet_candidate[:, :, y: y + h, x: x + w])
                        rnet_candidates_params.append((x + yx_offset[0], y + yx_offset[1], w, h))

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
                            yx_offset = rnet_candidates_params[rnet_candidate_idx]
                            for idx, bbox in enumerate(onet_bboxes):
                                x, y, w, h = bbox[0].long(), bbox[1].long(), bbox[2].long(), bbox[3].long()
                                onet_candidates_offsets.append((x + yx_offset[0], y + yx_offset[1], w, h))
                                final_bboxes.append(torch.tensor([x + yx_offset[0], y + yx_offset[1], w, h]).float())
                                final_scores.append(onet_scores[idx])

        final_scores = torch.tensor(final_scores).float()
        plot_im_with_bbox(im[0], bboxes=final_bboxes, scores=final_scores, iou_threshold=0.2)


def sliding_window(image, window_size, stride):
    """
    Function to generate patches from an image using a sliding window approach.

    Parameters:
    - image: Input image (numpy array).
    - window_size: Tuple (w, h) specifying the size of the sliding window.
    - stride: Stride of the sliding window.

    Returns:
    - patches: List of patches extracted from the image.
    """

    patches = []
    image_height, image_width = image.shape[:2]
    window_width, window_height = window_size

    for y in range(0, image_height - window_height + 1, stride):
        for x in range(0, image_width - window_width + 1, stride):
            patch = image[y:y + window_height, x:x + window_width, :]
            patches.append((patch, x, y))

    return patches


def predict_faces_in_image(im, window_size=None, min_pyramid_size=100, reduction_factor=0.7):
    if window_size is None:
        window_size = [(i, i) for i in range(50, 501, 200)]

    pnet = PNet()
    pnet_resize = Resize(size=(12, 12), antialias=True)
    checkpoint = torch.load('pnet_training_large_celeba/checkpoint/best_checkpoint.pth')
    pnet.load_state_dict(checkpoint)
    pnet.eval()

    rnet = RNet()
    rnet_resize = Resize(size=(24, 24), antialias=True)
    checkpoint = torch.load('rnet_training_large_celeba/checkpoint/best_checkpoint.pth')
    rnet.load_state_dict(checkpoint)
    rnet.eval()

    onet = ONet()
    onet_resize = Resize(size=(48, 48), antialias=True)
    checkpoint = torch.load('onet_training/checkpoint/checkpoint_epoch_30.pth')
    onet.load_state_dict(checkpoint)
    onet.eval()
    image_patches = []
    for ws in window_size:
        image_patches += sliding_window(image=im, window_size=ws, stride=min(ws[0] // 2, 100))
    transform = Compose([ToTensor()])
    final_bboxes, final_scores = [], []
    for patch, x0, y0 in image_patches:
        patch = torch.unsqueeze(transform(patch), 0)
        image_pyramid = make_image_pyramid(patch, min_pyramid_size=min_pyramid_size, reduction_factor=reduction_factor)
        orig_x, orig_y = patch.shape[3], patch.shape[2]
        pnet_candidates, pnet_candidates_params, pnet_bboxes, pnet_scores = [], [], [], []

        for scaled_im in image_pyramid:
            scaled_im = pnet_resize(scaled_im)
            out = pnet(scaled_im)
            y, bbox = out["y_pred"], out["bbox_pred"]
            y = torch.exp(y)
            y = y / y.sum()

            if torch.argmax(y) == 1:
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
            for bbox in pnet_bboxes:
                x, y, w, h = bbox[0].long(), bbox[1].long(), bbox[2].long(), bbox[3].long()
                x = torch.clip(x, min=0, max=orig_x)
                y = torch.clip(y, min=0, max=orig_y)
                pnet_candidates.append(patch[:, :, y: y + h, x: x + w])
                pnet_candidates_params.append((x, y, w, h))

            for pnet_candidate_idx, pnet_candidate in enumerate(pnet_candidates):
                rnet_candidates, rnet_candidates_params, rnet_bboxes, rnet_scores = [], [], [], []
                image_pyramid = make_image_pyramid(pnet_candidate, min_pyramid_size=min_pyramid_size,
                                                   reduction_factor=reduction_factor)
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
                    yx_offset = pnet_candidates_params[pnet_candidate_idx]
                    for bbox in rnet_bboxes:
                        x, y, w, h = bbox[0].long(), bbox[1].long(), bbox[2].long(), bbox[3].long()
                        rnet_candidates.append(pnet_candidate[:, :, y: y + h, x: x + w])
                        rnet_candidates_params.append((x + yx_offset[0], y + yx_offset[1], w, h))

                    for rnet_candidate_idx, rnet_candidate in enumerate(rnet_candidates):
                        onet_candidates_offsets, onet_bboxes, onet_scores = [], [], []
                        image_pyramid = make_image_pyramid(rnet_candidate, min_pyramid_size=min_pyramid_size,
                                                           reduction_factor=reduction_factor)
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
                            yx_offset = rnet_candidates_params[rnet_candidate_idx]
                            for idx, bbox in enumerate(onet_bboxes):
                                x, y, w, h = bbox[0].long(), bbox[1].long(), bbox[2].long(), bbox[3].long()
                                onet_candidates_offsets.append((x + yx_offset[0], y + yx_offset[1], w, h))
                                final_bboxes.append(
                                    torch.tensor([x0 + x + yx_offset[0], y0 + y + yx_offset[1], w, h]).float())
                                final_scores.append(onet_scores[idx])

    final_scores = torch.tensor(final_scores).float()
    return final_bboxes, final_scores


def run_train_pnet():
    transform = Compose([ToTensor()])
    train_dataset = MTCNNDataset(path="data/celebA", partition="train", transform=transform, min_crop=100, max_crop=180,
                                 n=100000, n_hard=0, out_size=(12, 12))
    val_dataset = MTCNNDataset(path="data/celebA", partition="val", transform=transform, min_crop=100, max_crop=180,
                               n=10000, n_hard=0, out_size=(12, 12))

    # train_dataset = MTCNNWiderFace(path="data/wider_face", partition="train", transform=transform, neg_th=0.3,
    #                                pos_th=0.65, min_crop=12, out_size=(12, 12), n=10000, n_hard=0)
    # val_dataset = MTCNNWiderFace(path="data/wider_face", partition="val", transform=transform, neg_th=0.3,
    #                              pos_th=0.65, min_crop=12, out_size=(12, 12), n=1000, n_hard=0)

    train_params = {
        "lr": 1e-2,
        "optimizer": "adam",
        "n_epochs": 400,
        "batch_size": 128,
    }
    pnet = PNet()

    def lr_step(epoch):
        if epoch <= 10:
            return 1
        elif 10 < epoch <= 40:
            return 0.1
        else:
            return 0.01

    device = "cuda"
    train(net=pnet, train_dataset=train_dataset, val_dataset=val_dataset, train_params=train_params,
          out_dir="pnet_training_large_celeba", checkpoint_step=10, lr_step=lr_step, device=device, weights=[1.0, 1.0],
          wd=1e-3)


def run_train_rnet():
    transform = Compose([ToTensor()])
    pnet = PNet()
    checkpoint = torch.load('pnet_training_large_celeba/checkpoint/best_checkpoint.pth')
    pnet.load_state_dict(checkpoint)
    train_dataset = MTCNNDataset(previous_net=pnet, previous_transform=Resize((12, 12)), path="data/celebA",
                                 partition="train", transform=transform,
                                 min_crop=100, max_crop=180, n=100000, n_hard=10000, out_size=(24, 24))
    val_dataset = MTCNNDataset(previous_net=pnet, previous_transform=Resize((12, 12)), path="data/celebA",
                               partition="val", transform=transform, min_crop=100,
                               max_crop=180, n=2000, n_hard=0, out_size=(24, 24))

    train_params = {
        "lr": 1e-2,
        "optimizer": "adam",
        "n_epochs": 400,
        "batch_size": 128,
    }
    rnet = RNet()
    device = "cuda"

    def lr_step(epoch):
        if epoch <= 10:
            return 1
        elif 10 < epoch <= 40:
            return 0.1
        else:
            return 0.01

    train(net=rnet, train_dataset=train_dataset, val_dataset=val_dataset, train_params=train_params,
          out_dir="rnet_training_large_celeba", checkpoint_step=10, lr_step=lr_step, device=device, weights=[1.0, 1.0],
          wd=1e-3)


def run_train_onet():
    transform = Compose([ToTensor()])

    pnet = PNet()
    # Load the checkpoint
    checkpoint = torch.load('pnet_training_large_celeba/checkpoint/best_checkpoint.pth')

    rnet = RNet()
    # Load the checkpoint
    checkpoint = torch.load('rnet_training_large_celeba/checkpoint/best_checkpoint.pth')
    # Load the model state dictionary
    rnet.load_state_dict(checkpoint)
    rnet.eval()
    train_dataset = MTCNNDataset(previous_net=rnet, previous_transform=Resize((24, 24)), path="data/celebA",
                                 partition="train", transform=transform,
                                 min_crop=40, max_crop=200, n=100000, n_hard=0, out_size=(48, 48))
    val_dataset = MTCNNDataset(previous_net=rnet, previous_transform=Resize((24, 24)), path="data/celebA",
                               partition="val", transform=transform, min_crop=40,
                               max_crop=200, n=2000, n_hard=0, out_size=(48, 48))

    train_params = {
        "lr": 1e-2,
        "optimizer": "adam",
        "n_epochs": 400,
        "batch_size": 128,
    }
    onet = ONet()
    device = "cuda"

    def lr_step(epoch):
        if epoch <= 10:
            return 1
        elif 10 < epoch <= 40:
            return 0.1
        else:
            return 0.01

    train(net=onet, train_dataset=train_dataset, val_dataset=val_dataset, train_params=train_params,
          out_dir="onet_training_large_celeba", checkpoint_step=1, lr_step=lr_step, device=device, weights=[1.0, 1.0],
          wd=1e-3)


def plot_image_with_bounding_box(image, bounding_boxes, freeze=False):
    # Loop over bounding boxes and draw them on the image
    for box in bounding_boxes:
        x, y, w, h = round(box[0].item()), round(box[1].item()), round(box[2].item()), round(box[3].item())
        # cv2.rectangle(img=image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a green rectangle
        cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

    # Display the image with bounding boxes
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(f'Frame', bgr_image)
    if freeze:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.waitKey(1)


def postprocess_bboxes_and_scores(bboxes, scores, detection_th=0.96, iou_th=0.3):
    if len(scores):
        new_bboxes, new_scores = [], []
        for idx, x in enumerate(scores > detection_th):
            if x.item():
                new_bboxes.append(bboxes[idx])
                new_scores.append(scores[idx])
        if len(new_scores):
            bboxes, scores = new_bboxes, torch.hstack(new_scores)
        else:
            bboxes, scores = [], []

    if len(bboxes):
        bboxes = torch.vstack(bboxes)
        selected_indices = nms(boxes=bboxes, scores=scores, iou_threshold=iou_th)
        bboxes, scores = bboxes[selected_indices, :], scores[selected_indices]
        bboxes_iou = IoU(bboxes, bboxes)
        N = len(scores)
        keep_map = np.ones(N)
        for i in range(N):
            if keep_map[i] == 0:
                continue

            for j in range(N):
                if i == j:
                    continue

                if bboxes_iou[i, j] > iou_th:
                    keep_map[j] = 0

        if np.sum(keep_map) > 0:
            bboxes = [bboxes[idx, :] for idx, token in enumerate(keep_map) if token > 0]
            scores = torch.hstack([scores[idx] for idx, token in enumerate(keep_map) if token > 0])
        else:
            bboxes, scores = [], []
    return bboxes, scores


def live_face_detection(target_fps):
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 is the default camera, you can change it if you have multiple cameras

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open the camera")
        return

    # Calculate the delay based on the target FPS
    delay = int(1000 / target_fps)
    frame_idx = 0
    # Loop to continuously capture frames
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Check if the frame is empty
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Predict faces in the image and draw bounding boxes
        bboxes, scores = predict_faces_in_image(frame, window_size=[(400, 400)])
        bboxes, scores = postprocess_bboxes_and_scores(bboxes, scores, detection_th=0.95, iou_th=0)
        # Display the image with bounding boxes
        plot_image_with_bounding_box(frame, bboxes)
        frame_idx += 1
        # Check for the 'q' key to quit the program
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()


def find_jpg_files(folder_path, n, jpg_files=[]):
    """
    Recursively find JPG files in a folder and its subfolders.

    Args:
    - folder_path: The path to the folder to search.
    - n: The number of JPG files to pick.
    - jpg_files: A list to store the paths of JPG files found (default=[]).

    Returns:
    - A list of paths to JPG files.
    """
    if n == 0:
        return jpg_files

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            jpg_files = find_jpg_files(item_path, n, jpg_files)
        elif item.lower().endswith('.jpg'):
            jpg_files.append(os.path.abspath(item_path))
            n -= 1
            if n == 0:
                break

    return jpg_files


if __name__ == "__main__":
    # test_propose_net()
    # test_residual_net()
    # test_onet()
    # test()
    print("start Pnet training!")
    run_train_pnet()
    print("start Rnet training!")
    run_train_rnet()
    print("start Onet training!")
    run_train_onet()
    # live_face_detection(target_fps=10)
    # folder_path = "/Users/eliav/Documents/GitHub/MTCNN/MTCNN/data/wider_face/WIDER_test/images"
    # n = 5
    # files = find_jpg_files(folder_path=folder_path, n=n)
    # for path in files:
    # path = "/Users/eliav/Documents/GitHub/MTCNN/MTCNN/data/wider_face/WIDER_test/images/35--Basketball/35_Basketball_playingbasketball_35_868.jpg"
    # frame = cv2.imread(path)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # bboxes, scores = predict_faces_in_image(frame, window_size=[(i, i) for i in range(100, 101, 50)],
    #                                         min_pyramid_size=120, reduction_factor=0.01)
    # bboxes, scores = postprocess_bboxes_and_scores(bboxes, scores, detection_th=0.96, iou_th=0.5)
    # # Display the image with bounding boxes
    # plot_image_with_bounding_box(frame, bboxes, freeze=True)
