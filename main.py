from model import PNet
from MTCNN.datasets import PNetDataset
from torchvision.transforms import ToTensor, Compose
from trainer import train_pnet

if __name__ == "__main__":
    transform = Compose([ToTensor()])
    train_dataset = PNetDataset(path="data/celebA", partition="test", transform=transform, min_crop=20, max_crop=80)
    val_dataset = PNetDataset(path="data/celebA", partition="val", transform=transform, min_crop=20, max_crop=80)
    train_params = {
        "lr": 1e-3,
        "optimizer": "adam",
        "n_epochs": 10,
        "batch_size": 16,
    }
    pnet = PNet()
    train_pnet(pnet=pnet, train_dataset=train_dataset, val_dataset=val_dataset, train_params=train_params,
               out_dir="../pnet", checkpoint_step=2)
