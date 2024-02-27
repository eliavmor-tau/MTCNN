import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch.nn.functional import cross_entropy, mse_loss
from logger import Logger
from utils import plot_im_with_bbox

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


def train_pnet(pnet, train_dataset, val_dataset, train_params, out_dir, checkpoint_step=None, device="cpu", weights=[1., 0.5]):
    n_epochs = train_params.get("n_epochs")
    batch_size = train_params.get("batch_size")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    optimizer_params = {
        "lr": train_params.get("lr"),
        "params": pnet.parameters()
    }

    def lr_step(epoch):
        if epoch <= 100:
            return 1.0
        else:
            return 0.1

    optimizer = load_optimizer(optimizer_name=train_params.get("optimizer"),
                               optimizer_params=optimizer_params)
    lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_step)

    detection_loss = cross_entropy
    bbox_loss = mse_loss
    header = ["epoch", "detection_loss", "bbox_loss", "total_loss", "lr"]
    train_logger = Logger(header=header, out_dir=out_dir, log_name="train_log")
    val_logger = Logger(header=header, out_dir=out_dir, log_name="val_log")
    if device == "cuda" and not torch.cuda.is_available():
        print("Cannot train on cuda as cuda is not available.")
        device = "cpu"

    device = torch.device(device=device)
    pnet.to(device)
    print(f"Training on {device}")

    for epoch in tqdm(range(n_epochs), desc=f"epochs", total=n_epochs):
        # train epoch
        train_detection_loss, train_bbox_loss, train_loss = 0, 0, 0
        for batch in train_dataloader:
            images, bboxes, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            out = pnet(images)
            pred_bboxes = out["bbox_pred"]
            y_pred = out["y_pred"]
            l_bbox = bbox_loss(input=pred_bboxes, target=bboxes, reduction="mean")
            l_detect = detection_loss(input=y_pred, target=y, reduction="mean")
            loss = l_detect * weights[0] + l_bbox * weights[1]

            train_detection_loss += l_detect.detach().item()
            train_bbox_loss += l_bbox.detach().item()
            train_loss += loss.detach().item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_detection_loss = train_detection_loss / len(train_dataloader)
        train_bbox_loss = train_bbox_loss / len(train_dataloader)
        train_loss = train_loss / len(train_dataloader)
        current_lr = lr_scheduler.get_last_lr()[0]
        train_logger.write(line=[epoch, train_detection_loss, train_bbox_loss, train_loss, current_lr])
        if checkpoint_step is not None and (epoch % checkpoint_step) == 0:
            train_logger.save_model(model=pnet, checkpoint_name=f"checkpoint_epoch_{epoch}.pth")

        # validation epoch
        with torch.no_grad():
            val_detection_loss, val_bbox_loss, val_loss = 0, 0, 0
            for batch in val_dataloader:
                images, bboxes, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                out = pnet(images)
                pred_bboxes = out["bbox_pred"]
                y_pred = out["y_pred"]
                l_bbox = bbox_loss(input=pred_bboxes, target=bboxes, reduction="mean")
                l_detect = detection_loss(input=y_pred, target=y, reduction="mean")
                loss = l_detect * weights[0] + l_bbox * weights[1]

                val_detection_loss += l_detect.detach().item()
                val_bbox_loss += l_bbox.detach().item()
                val_loss += loss.detach().item()

            val_detection_loss = val_detection_loss / len(val_dataloader)
            val_bbox_loss = val_bbox_loss / len(val_dataloader)
            val_loss = val_loss / len(val_dataloader)
            val_logger.write(line=[epoch, val_detection_loss, val_bbox_loss, val_loss, current_lr])

        # update step size
        lr_scheduler.step()
        if epoch % 10 == 0:
            print(f"train_detection_loss={train_detection_loss}")
            print(f"train_bbox_loss={train_bbox_loss}")
            print(f"val_detection_loss={val_detection_loss}")
            print(f"val_bbox_loss={val_bbox_loss}")
            print("-"* 25)

    train_logger.save_model(model=pnet, checkpoint_name=f"last_epoch_checkpoint_{n_epochs}.pth")
