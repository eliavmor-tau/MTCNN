import argparse
import torch
from model import PNet, RNet, ONet
from datasets import MTCNNDataset
from torchvision.transforms import ToTensor, Compose, Resize
from trainer import train


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train MTCNN network.")
    parser.add_argument("--net", type=str, choices=["pnet", "rnet", "onet"], help="Network to train")
    return parser.parse_args()


def get_datasets_and_transforms(net):
    """
    Get datasets and transforms for training based on the specified network.

    Args:
        net (str): Network to train.

    Returns:
        tuple: Train and validation datasets.
    """
    transform = Compose([ToTensor()])
    if net == "pnet":
        return (
            MTCNNDataset(path="../data/celebA", partition="train", transform=transform, min_crop=100, max_crop=180,
                         n=100000, n_hard=0, out_size=(12, 12)),
            MTCNNDataset(path="../data/celebA", partition="val", transform=transform, min_crop=100, max_crop=180,
                         n=10000, n_hard=0, out_size=(12, 12))
        )
    elif net == "rnet":
        pnet = PNet()
        checkpoint = torch.load('../logs/pnet_training/checkpoint/best_checkpoint.pth')
        pnet.load_state_dict(checkpoint)
        return (
            MTCNNDataset(previous_net=pnet, previous_transform=Resize((12, 12)), path="../data/celebA",
                         partition="train", transform=transform, min_crop=100, max_crop=180, n=100000, n_hard=10000,
                         out_size=(24, 24)),
            MTCNNDataset(previous_net=pnet, previous_transform=Resize((12, 12)), path="../data/celebA",
                         partition="val", transform=transform, min_crop=100, max_crop=180, n=2000, n_hard=0,
                         out_size=(24, 24))
        )
    elif net == "onet":
        pnet = PNet()
        rnet = RNet()
        checkpoint_pnet = torch.load('../logs/pnet_training/checkpoint/best_checkpoint.pth')
        checkpoint_rnet = torch.load('../logs/rnet_training/checkpoint/best_checkpoint.pth')
        pnet.load_state_dict(checkpoint_pnet)
        rnet.load_state_dict(checkpoint_rnet)
        rnet.eval()
        return (
            MTCNNDataset(previous_net=rnet, previous_transform=Resize((24, 24)), path="../data/celebA",
                         partition="train", transform=transform, min_crop=40, max_crop=200, n=100000, n_hard=0,
                         out_size=(48, 48)),
            MTCNNDataset(previous_net=rnet, previous_transform=Resize((24, 24)), path="../data/celebA",
                         partition="val", transform=transform, min_crop=40, max_crop=200, n=2000, n_hard=0,
                         out_size=(48, 48))
        )


def run_training(net, train_dataset, val_dataset, out_dir):
    """
    Run training for the specified network.

    Args:
        net (torch.nn.Module): The network to train.
        train_dataset (torch.utils.data.Dataset): Training dataset.
        val_dataset (torch.utils.data.Dataset): Validation dataset.
        out_dir (str): Output directory for saving checkpoints and logs.

    Returns:
        None
    """
    train_params = {
        "lr": 1e-2,
        "optimizer": "adam",
        "n_epochs": 2,
        "batch_size": 128,
    }
    device = "cuda"

    def lr_step(epoch):
        if epoch <= 10:
            return 1
        elif 10 < epoch <= 40:
            return 0.1
        else:
            return 0.01

    train(net=net, train_dataset=train_dataset, val_dataset=val_dataset, train_params=train_params,
          out_dir=out_dir, checkpoint_step=1, lr_step=lr_step, device=device, weights=[1.0, 1.0], wd=1e-3)


def main():
    """
    Main function to run network training.
    """
    args = parse_arguments()
    if args.net.lower() == "pnet":
        print("Training PNet...")
        train_dataset, val_dataset = get_datasets_and_transforms("pnet")
        pnet = PNet()
        run_training(net=pnet, train_dataset=train_dataset, val_dataset=val_dataset,
                     out_dir="../logs/pnet_training")
    elif args.net.lower() == "rnet":
        print("Training RNet...")
        train_dataset, val_dataset = get_datasets_and_transforms("rnet")
        rnet = RNet()
        run_training(net=rnet, train_dataset=train_dataset, val_dataset=val_dataset,
                     out_dir="../logs/rnet_training")
    elif args.net.lower() == "onet":
        print("Training ONet...")
        train_dataset, val_dataset = get_datasets_and_transforms("onet")
        onet = ONet()
        run_training(net=onet, train_dataset=train_dataset, val_dataset=val_dataset,
                     out_dir="../logs/onet_training")


if __name__ == "__main__":
    main()
