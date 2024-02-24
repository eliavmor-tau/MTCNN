import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam, SGD
from torch.nn.functional import cross_entropy, mse_loss
from logger import Logger

OPTIMIZERS = {
    "sgd": SGD,
    "adam": Adam
}


def load_optimizer(optimizer_name, optimizer_params):
    if optimizer_name not in OPTIMIZERS:
        raise f"cannot load optimizer {optimizer_name}. supported optimizers are {OPTIMIZERS.keys()}"
    optimizer_constructor = OPTIMIZERS[optimizer_name]
    optimizer = optimizer_constructor(**optimizer_params)
    return optimizer


def train_pnet(pnet, train_dataset, val_dataset, train_params, out_dir, checkpoint_step=None, device="cpu"):
    n_epochs = train_params.get("n_epochs")
    batch_size = train_params.get("batch_size")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    optimizer_params = {
        "lr": train_params.get("lr"),
        "params": pnet.parameters()
    }
    optimizer = load_optimizer(optimizer_name=train_params.get("optimizer"),
                               optimizer_params=optimizer_params)
    detection_loss = cross_entropy
    bbox_loss = mse_loss
    header = ["epoch", "detection_loss", "bbox_loss", "total_loss"]
    train_logger = Logger(header=header, out_dir=out_dir, log_name="train_log")
    val_logger = Logger(header=header, out_dir=out_dir, log_name="val_log")
    if device == "cuda" and not torch.cuda.is_available():
        print("Cannot train on cuda as cuda is not available. Training on CPU")
        device = "cpu"

    device = torch.device(device=device)
    pnet.to(device)
    for epoch in tqdm(range(n_epochs), desc=f"epochs", total=n_epochs):
        train_detection_loss, train_bbox_loss, train_loss = 0, 0, 0
        for batch in train_dataloader:
            images, bboxes, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            out = pnet(images)
            pred_bboxes = out["bbox_pred"]
            y_pred = out["y_pred"]
            l_bbox = bbox_loss(input=pred_bboxes, target=bboxes, reduction="mean")
            l_detect = detection_loss(input=y_pred, target=y, reduction="mean")
            loss = (l_detect + l_bbox) / 2.

            train_detection_loss += l_detect.detach().item()
            train_bbox_loss += l_bbox.detach().item()
            train_loss += loss.detach().item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_detection_loss = train_detection_loss / len(train_dataloader)
        train_bbox_loss = train_bbox_loss / len(train_dataloader)
        train_loss = train_loss / len(train_dataloader)
        train_logger.write(line=[epoch, train_detection_loss, train_bbox_loss, train_loss])
        if checkpoint_step is not None and (epoch % checkpoint_step) == 0:
            train_logger.save_model(model=pnet, checkpoint_name=f"checkpoint_epoch_{epoch}.pth")

        with torch.no_grad():
            val_detection_loss, val_bbox_loss, val_loss = 0, 0, 0
            for batch in train_dataloader:
                images, bboxes, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                out = pnet(images)
                pred_bboxes = out["bbox_pred"]
                y_pred = out["y_pred"]
                l_bbox = bbox_loss(input=pred_bboxes, target=bboxes, reduction="mean")
                l_detect = detection_loss(input=y_pred, target=y, reduction="mean")
                loss = (l_detect + l_bbox) / 2.

                val_detection_loss += l_detect.detach().item()
                val_bbox_loss += l_bbox.detach().item()
                val_loss += loss.detach().item()

            val_detection_loss = val_detection_loss / len(val_dataloader)
            val_bbox_loss = val_bbox_loss / len(val_dataloader)
            val_loss = val_loss / len(val_dataloader)
            val_logger.write(line=[epoch, val_detection_loss, val_bbox_loss, val_loss])

    train_logger.save_model(model=pnet, checkpoint_name=f"last_epoch_checkpoint_{n_epochs}.pth")
