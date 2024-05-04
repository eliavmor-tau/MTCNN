import os
import pandas as pd
import csv
import torch


class Logger:
    LOGS_DIR = "../logs"
    CHECKPOINT_DIR = "checkpoint"

    def __init__(self, out_dir, header, log_name):
        self.out_dir = out_dir
        self.checkpoint_dir = os.path.join(out_dir, Logger.CHECKPOINT_DIR)
        self.log_dir = os.path.join(out_dir, Logger.LOGS_DIR)
        self.log_name = log_name
        self.log_path = os.path.join(self.log_dir, f"{self.log_name}.csv")
        self.header = header
        self.line_length = len(header)
        self.__make_dir()
        with open(self.log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            file.flush()

    def __make_dir(self):
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def write(self, line):
        if len(line) == self.line_length:
            with open(self.log_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(line)
                file.flush()
        else:
            raise f"couldn't write to Logger as line length is {len(line)} but logger header is of length {self.line_length}"

    def save_model(self, model, checkpoint_name):
        model_device = next(model.parameters()).device
        model.to("cpu")
        torch.save(model.state_dict(), os.path.join(self.checkpoint_dir, checkpoint_name))
        model.to(model_device)


