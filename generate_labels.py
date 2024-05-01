import torch
from facenet_pytorch import MTCNN
from datasets import CelebA
import pandas as pd
from utils import plot_im_with_bbox
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_faces(image, faces):
    # Plot the image with bounding boxes around detected faces
    plt.figure(figsize=(8, 6))
    plt.imshow(image)

    # Plot each detected face
    for face in faces:
        x, y, width, height = face
        width, height = width - x, height - y
        # Create a rectangle patch
        rect = Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        # Add the patch to the plot
        plt.gca().add_patch(rect)

    plt.axis('off')
    plt.show()

# Load pre-trained models for face detection and face recognition
mtcnn = MTCNN(keep_all=True)

dataset_train = CelebA(path="data/celebA", partition="train")
dataset_val = CelebA(path="data/celebA", partition="val")
dataset_test = CelebA(path="data/celebA", partition="test")
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
    image_name = bbox[0][0]
    print(image_name)
    # Perform face detection
    boxes, probs = mtcnn.detect(img)
    # plot_faces(img, boxes)
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
        if counter >= 100000:
            break

print("finish train", len(data["partition"]))
counter = 0
for img, bbox in dataset_val:
    image_name = bbox[0][0]
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
        if counter >= 10000:
            break

print("finish val")
counter = 0
for img, bbox in dataset_test:
    image_name = bbox[0][0]
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
        if counter >= 10000:
            break

df = pd.DataFrame(data=data)
df.to_csv("list_bbox_celeba_align_and_crop.csv")