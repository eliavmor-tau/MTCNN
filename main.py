from model import PNet
from datasets import PNetDataset
from torchvision.transforms import ToTensor, Compose
from trainer import train_pnet, view_pnet_predictions
import torch
from utils import plot_im_with_bbox

if __name__ == "__main__":
    transform = Compose([ToTensor()])
    train_dataset = PNetDataset(path="data/celebA", partition="train", transform=transform, min_crop=20, max_crop=80,
                                p_prob=0.65)
    val_dataset = PNetDataset(path="data/celebA", partition="val", transform=transform, min_crop=20, max_crop=80,
                              p_prob=0.5)
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

    # view_pnet_predictions(pnet, train_dataset)

    train_pnet(pnet=pnet, train_dataset=train_dataset, val_dataset=val_dataset, train_params=train_params,
               out_dir="pnet_retrain", checkpoint_step=50, device="cuda")
