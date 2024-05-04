import argparse
import torch
from model import PNet, RNet, ONet
from datasets import CelebA
from torchvision.transforms import ToTensor, Compose, Resize
from utils import plot_im_with_bbox, make_image_pyramid
from torch.utils.data import DataLoader


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Test MTCNN networks.")
    parser.add_argument("--net", type=str, choices=["pnet", "rnet", "onet"], help="Network to test")
    return parser.parse_args()


def load_checkpoint(network, checkpoint_path):
    """
    Load the checkpoint for a given network.

    Args:
        network (torch.nn.Module): The network to load the checkpoint into.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        torch.nn.Module: Network with loaded checkpoint.
    """
    network.load_state_dict(torch.load(checkpoint_path))
    return network


def test_network(net, resize_size, iou_threshold):
    """
    Test the specified network.

    Args:
        net (torch.nn.Module): The network to test.
        resize_size (tuple): Size to which images will be resized.
        iou_threshold (float): Threshold for intersection over union.

    Returns:
        None
    """
    transform = Compose([ToTensor()])
    net.eval()
    resize = Resize(size=resize_size, antialias=True)
    dataset = CelebA(path="../data/celebA", partition="test", transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=1)

    for im in dataloader:
        image_pyramid = make_image_pyramid(im)
        bboxes = []
        orig_x, orig_y = im.shape[3], im.shape[2]
        for scaled_im in image_pyramid:
            scaled_im = resize(scaled_im)
            out = net(scaled_im)
            y, bbox = out["y_pred"], out["bbox_pred"]
            bbox[0][0] = bbox[0][0] * orig_x
            bbox[0][2] = bbox[0][2] * orig_x
            bbox[0][1] = bbox[0][1] * orig_y
            bbox[0][3] = bbox[0][3] * orig_y
            bboxes.append(bbox.detach()[0])
        plot_im_with_bbox(im[0], bboxes, scores=None, iou_threshold=iou_threshold)


def main():
    """
    Main function to run network testing.
    """
    args = parse_arguments()
    if args.net == "pnet":
        pnet = PNet()
        checkpoint_path = 'pnet_training_large_celeba/checkpoint/best_checkpoint.pth'
        pnet = load_checkpoint(pnet, checkpoint_path)
        test_network(pnet, resize_size=(12, 12), iou_threshold=0.6)
    elif args.net == "rnet":
        rnet = RNet()
        checkpoint_path = 'rnet_training_large_celeba/checkpoint/best_checkpoint.pth'
        rnet = load_checkpoint(rnet, checkpoint_path)
        test_network(rnet, resize_size=(24, 24), iou_threshold=0.2)
    elif args.net == "onet":
        onet = ONet()
        checkpoint_path = 'logs/onet_training/checkpoint/last_epoch_checkpoint_200.pth'
        onet = load_checkpoint(onet, checkpoint_path)
        test_network(onet, resize_size=(48, 48), iou_threshold=0.2)


if __name__ == "__main__":
    main()
