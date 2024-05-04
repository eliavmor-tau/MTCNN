# MTCNN

This repository implements the Multitask Cascaded Convolutional Networks (MTCNN) for joint face detection and alignment, as described in the paper by Zhang et al. [\[1\]](http://dx.doi.org/10.1109/LSP.2016.2603342). This implementation focuses on face detection only, excluding the training of facial landmark detection.

## Overview

MTCNN is a popular face detection algorithm known for its accuracy and efficiency. It consists of three stages: Proposal Network (PNet), Refine Network (RNet), and Output Network (ONet). These stages progressively refine the detection results, leading to accurate bounding boxes around faces.

## Usage

### Environment Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

### Training Procedure
Enter the `src` folder via:
```
cd src
```
#### Step 1.0: Train PNet

Train the Proposal Network (PNet) using the following command:
```bash
python train_models.py --net pnet
```

#### Step 1.1: Analyze PNet Loss Curves

Analyze PNet loss curves using `notebooks/analyzer.ipynb`. You can also visualize the network's performance on the test set using:
```bash
python test_modules.py --net pnet
```

#### Step 2.0: Train RNet

Train the Refine Network (RNet) using the following command:
```bash
python train_models.py --net rnet
```

#### Step 2.1: Analyze RNet Loss Curves

Analyze RNet loss curves using `notebooks/analyzer.ipynb`. You can also visualize the network's performance on the test set using:
```bash
python test_modules.py --net rnet
```

#### Step 3.0: Train RNet

Train the Output Network (ONet) using the following command:
```bash
python train_models.py --net rnet
```

#### Step 3.1: Analyze ONet Loss Curves

Analyze ONet loss curves using `notebooks/analyzer.ipynb`. You can also visualize the network's performance on the test set using:
```bash
python test_modules.py --net rnet
```

### Live Face Detection
Enter the `src` folder via:
```
cd src
```
then use
```bash
python run_face_detection.py --fps 200
```
to start running your MTCNN on live images.

Uploading live_detection_demo.mp4…


## References

\[1\] Zhang, Kaipeng, et al. "Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks." *IEEE Signal Processing Letters*, vol. 23, no. 10, 2016, pp. 1499–1503.
