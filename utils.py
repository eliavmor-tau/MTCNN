import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from torchvision.transforms import Resize, ToTensor
from torch import Tensor
from torchvision.ops import box_iou
from torchvision.ops import nms


def plot_im_with_bbox(im: Tensor, bboxes: list, scores: [list, None] = None, iou_threshold: float = 0.6, title=""):
    if im.shape[0] == 3 and im.shape[2] != 3:
        im = np.transpose(im, axes=(1, 2, 0))
    fig, axis = plt.subplots()
    fig.suptitle(title)
    axis.imshow(im)
    for bbox in bboxes:
        rec = patches.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2], height=bbox[3], linewidth=2, edgecolor='green',
                                facecolor='none')
        axis.add_patch(rec)
    bboxes = torch.vstack(bboxes)
    if scores is None:
        scores = torch.ones(bboxes.shape[0])
    else:
        scores = torch.hstack(scores)
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    bboxes_indices = nms(bboxes, scores, iou_threshold)
    for index in bboxes_indices:
        rec = patches.Rectangle(xy=(bboxes[index][0], bboxes[index][1]), width=bboxes[index][2] - bboxes[index][0],
                                height=bboxes[index][3] - bboxes[index][1], linewidth=2, edgecolor='blue',
                                facecolor='none')
        axis.add_patch(rec)
    plt.imshow(im)
    plt.show()
    plt.close()


def make_image_pyramid(im, min_pyramid_size=12, reduction_factor=0.9):
    im_pyramid = []
    min_dim = min(im.shape[-2:])
    while min_dim > min_pyramid_size:
        im_pyramid.append(im)
        resize = Resize(round(min_dim * reduction_factor), antialias=True)
        im = resize(im)
        min_dim = min(im.shape[-2:])
    return im_pyramid


def random_crop_and_update_bbox(image, bbox, output_size):
    # Randomly crop the image
    im_height, im_width = image.shape[1], image.shape[2]
    crop_height, crop_width = output_size[0], output_size[1]
    y_crop = np.random.randint(0, max(1, im_height - output_size[0]))
    x_crop = np.random.randint(0, max(1, im_width - output_size[1]))
    cropped_image = image[:, y_crop: y_crop + crop_height, x_crop: x_crop + crop_width]
    new_bbox = [0, 0, 0, 0]
    # Update bounding box coordinates
    new_bbox[2] = min(max(bbox[0] + bbox[2] - x_crop, 0), max(x_crop + crop_width - bbox[0], 0),
                      crop_width)  # Update width
    new_bbox[3] = min(max(bbox[1] + bbox[3] - y_crop, 0), max(y_crop + crop_height - bbox[1], 0),
                      crop_height)  # Update height
    new_bbox[0] = max(0, min(bbox[0] - x_crop, crop_width))  # Update x-coordinate
    new_bbox[1] = max(0, min(bbox[1] - y_crop, crop_height))  # Update y-coordinate
    return cropped_image, torch.tensor(new_bbox)


def IoU(bbox1, bbox2):
    new_bbox1 = bbox1.detach().clone()
    if len(bbox1.shape) == 1:
        new_bbox1 = new_bbox1.view((1, -1))
    new_bbox2 = bbox2.detach().clone()
    if len(bbox2.shape) == 1:
        new_bbox2 = new_bbox2.view((1, -1))
    new_bbox1[:, 2] = new_bbox1[:, 0] + new_bbox1[:, 2]
    new_bbox1[:, 3] = new_bbox1[:, 1] + new_bbox1[:, 3]
    new_bbox2[:, 2] = new_bbox2[:, 0] + new_bbox2[:, 2]
    new_bbox2[:, 3] = new_bbox2[:, 1] + new_bbox2[:, 3]
    return box_iou(new_bbox1, new_bbox2)


if __name__ == "__main__":
    pass
    from MTCNN.datasets import PNetDataset

    dataset = PNetDataset(path="data/celebA", partition="train", transform=ToTensor())
    im, bbox = dataset[100]
    plot_im_with_bbox(im, bbox)
    new_im, new_bbox = random_crop_and_update_bbox(im, bbox, (60, 60))
    image_bbox = torch.tensor([0, 0, new_im.shape[2], new_im.shape[1]])
    iou = IoU(image_bbox, new_bbox)
    plot_im_with_bbox(new_im, new_bbox, title=f"IoU={iou[0]}")
