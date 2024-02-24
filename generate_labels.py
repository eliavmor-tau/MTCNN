from facenet_pytorch import MTCNN
from MTCNN.datasets import CelebA
import pandas as pd

# Load pre-trained models for face detection and face recognition
mtcnn = MTCNN(keep_all=True)

dataset_train = CelebA(path="data/celebA", partition=0)
dataset_val = CelebA(path="data/celebA", partition=1)
dataset_test = CelebA(path="data/celebA", partition=2)

data = {
        "image_name": [],
        "x_1": [],
        "y_1": [],
        "width": [],
        "height": [],
        "partition": []
    }

counter = 0
for img, bbox in dataset_train:
    image_name = bbox[0]
    print(image_name)
    # Perform face detection
    boxes, probs = mtcnn.detect(img)
    if boxes is not None:
        box = boxes[0]
        x_1, y_1, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        data["image_name"].append(image_name)
        data["x_1"].append(round(x_1))
        data["y_1"].append(round(y_1))
        data["width"].append(round(width))
        data["height"].append(round(height))
        data["partition"].append(0)
        counter += 1
    if counter == 10000:
        break

print("finish train")
counter = 0
for img, bbox in dataset_val:
    image_name = bbox[0]
    print(image_name)
    # Perform face detection
    boxes, probs = mtcnn.detect(img)
    if boxes is not None:
        box = boxes[0]
        x_1, y_1, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        data["image_name"].append(image_name)
        data["x_1"].append(round(x_1))
        data["y_1"].append(round(y_1))
        data["width"].append(round(width))
        data["height"].append(round(height))
        data["partition"].append(1)
        counter += 1
    if counter == 2000:
        break

print("finish val")
counter = 0
for img, bbox in dataset_test:
    image_name = bbox[0]
    print(image_name)
    # Perform face detection
    boxes, probs = mtcnn.detect(img)
    if boxes is not None:
        box = boxes[0]
        x_1, y_1, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        data["image_name"].append(image_name)
        data["x_1"].append(round(x_1))
        data["y_1"].append(round(y_1))
        data["width"].append(round(width))
        data["height"].append(round(height))
        data["partition"].append(2)
        counter += 1
    if counter == 1000:
        break

df = pd.DataFrame(data=data)
df.to_csv("list_bbox_celeba_align_and_crop.csv")