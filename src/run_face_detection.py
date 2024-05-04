import numpy as np
from model import PNet, RNet, ONet
from torchvision.transforms import ToTensor, Compose, Resize
from trainer import train
import torch
from utils import make_image_pyramid, nms, IoU
import cv2
import time
import argparse


def sliding_window(image, window_size, stride):
    """
    Function to generate patches from an image using a sliding window approach.

    Parameters:
    - image: Input image (numpy array).
    - window_size: Tuple (w, h) specifying the size of the sliding window.
    - stride: Stride of the sliding window.

    Returns:
    - patches: List of patches extracted from the image.
    """

    patches = []
    image_height, image_width = image.shape[:2]
    window_width, window_height = window_size

    for y in range(0, image_height - window_height + 1, stride):
        for x in range(0, image_width - window_width + 1, stride):
            patch = image[y:y + window_height, x:x + window_width, :]
            patches.append((patch, x, y))

    return patches


def load_model_and_resize(model_class, checkpoint_path, resize_size):
    """
    Load the specified model class and resizing function from checkpoint.

    Args:
        model_class: Class of the model to load.
        checkpoint_path (str): Path to the checkpoint file.
        resize_size (tuple): Size for resizing the input images.

    Returns:
        tuple: A tuple containing the loaded model and resizing function.
    """
    model = model_class()
    resize_fn = Resize(size=resize_size, antialias=True)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()
    return model, resize_fn


def predict_faces_in_image(im, window_size=None, min_pyramid_size=100, reduction_factor=0.7):
    """
    Predicts faces in the given image using a multi-stage face detection pipeline.

    Args:
        im (torch.Tensor): Input image tensor.
        window_size (list, optional): List of window sizes for sliding window.
            Defaults to None, in which case a default set of window sizes is used.
        min_pyramid_size (int, optional): Minimum size for the image pyramid.
            Defaults to 100.
        reduction_factor (float, optional): Reduction factor for image pyramid.
            Defaults to 0.7.

    Returns:
        tuple: A tuple containing the final bounding boxes and their corresponding scores.
    """
    # Check if window size is provided, otherwise use default range
    if window_size is None:
        window_size = [(i, i) for i in range(50, 501, 200)]

    pnet, pnet_resize = load_model_and_resize(PNet, '../logs/pnet_training/checkpoint/best_checkpoint.pth', (12, 12))
    rnet, rnet_resize = load_model_and_resize(RNet, '../logs/rnet_training/checkpoint/best_checkpoint.pth', (24, 24))
    onet, onet_resize = load_model_and_resize(ONet, '../logs/onet_training/checkpoint/best_checkpoint.pth', (48, 48))

    image_patches = []
    for ws in window_size:
        image_patches += sliding_window(image=im, window_size=ws, stride=min(ws[0] // 2, 100))

    transform = Compose([ToTensor()])
    final_bboxes, final_scores = [], []

    # Iterate through image patches
    for patch, x0, y0 in image_patches:
        patch = torch.unsqueeze(transform(patch), 0)
        image_pyramid = make_image_pyramid(patch, min_pyramid_size=min_pyramid_size, reduction_factor=reduction_factor)
        orig_x, orig_y = patch.shape[3], patch.shape[2]
        pnet_candidates, pnet_candidates_params, pnet_bboxes, pnet_scores = [], [], [], []

        # P-Net processing
        for scaled_im in image_pyramid:
            scaled_im = pnet_resize(scaled_im)
            out = pnet(scaled_im)
            y, bbox = out["y_pred"], out["bbox_pred"]
            y = torch.exp(y)
            y = y / y.sum()

            if torch.argmax(y) == 1:
                bbox[0][0] = bbox[0][0] * orig_x
                bbox[0][2] = bbox[0][2] * orig_x
                bbox[0][1] = bbox[0][1] * orig_y
                bbox[0][3] = bbox[0][3] * orig_y
                pnet_bboxes.append(torch.round(bbox))
                pnet_scores.append(y[0][1])

        # Non-maximum suppression and clipping
        if len(pnet_bboxes):
            pnet_bboxes = torch.vstack(pnet_bboxes)
            pnet_scores = torch.tensor(pnet_scores)
            bboxes_indices = nms(boxes=pnet_bboxes, scores=pnet_scores, iou_threshold=0.3)
            pnet_bboxes = [pnet_bboxes[index] for index in bboxes_indices]
            pnet_scores = torch.tensor([pnet_scores[index] for index in bboxes_indices])
            for bbox in pnet_bboxes:
                x, y, w, h = bbox[0].long(), bbox[1].long(), bbox[2].long(), bbox[3].long()
                x = torch.clip(x, min=0, max=orig_x)
                y = torch.clip(y, min=0, max=orig_y)
                pnet_candidates.append(patch[:, :, y: y + h, x: x + w])
                pnet_candidates_params.append((x, y, w, h))

            # R-Net processing
            for pnet_candidate_idx, pnet_candidate in enumerate(pnet_candidates):
                rnet_candidates, rnet_candidates_params, rnet_bboxes, rnet_scores = [], [], [], []
                image_pyramid = make_image_pyramid(pnet_candidate, min_pyramid_size=min_pyramid_size,
                                                   reduction_factor=reduction_factor)
                input_x, input_y = pnet_candidate.shape[3], pnet_candidate.shape[2]
                for scaled_im in image_pyramid:
                    scaled_im = rnet_resize(scaled_im)
                    out = rnet(scaled_im)
                    y, bbox = out["y_pred"], out["bbox_pred"]
                    y = torch.exp(y)
                    y = y / y.sum()
                    if torch.argmax(y) == 1:
                        bbox[0][0] = torch.clip(bbox[0][0] * input_x, 0, input_x)
                        bbox[0][2] = torch.clip(bbox[0][2] * input_x, 0, input_x)
                        bbox[0][1] = torch.clip(bbox[0][1] * input_y, 0, input_y)
                        bbox[0][3] = torch.clip(bbox[0][3] * input_y, 0, input_y)
                        rnet_bboxes.append(torch.round(bbox))
                        rnet_scores.append(y[0][1])

                # Non-maximum suppression
                if len(rnet_bboxes):
                    rnet_bboxes = torch.vstack(rnet_bboxes)
                    rnet_scores = torch.tensor(rnet_scores)
                    bboxes_indices = nms(boxes=rnet_bboxes, scores=rnet_scores, iou_threshold=0.3)
                    rnet_bboxes = [rnet_bboxes[index] for index in bboxes_indices]
                    rnet_scores = torch.tensor([rnet_scores[index] for index in bboxes_indices])
                    yx_offset = pnet_candidates_params[pnet_candidate_idx]
                    for bbox in rnet_bboxes:
                        x, y, w, h = bbox[0].long(), bbox[1].long(), bbox[2].long(), bbox[3].long()
                        rnet_candidates.append(pnet_candidate[:, :, y: y + h, x: x + w])
                        rnet_candidates_params.append((x + yx_offset[0], y + yx_offset[1], w, h))

                    # O-Net processing
                    for rnet_candidate_idx, rnet_candidate in enumerate(rnet_candidates):
                        onet_candidates_offsets, onet_bboxes, onet_scores = [], [], []
                        image_pyramid = make_image_pyramid(rnet_candidate, min_pyramid_size=min_pyramid_size,
                                                           reduction_factor=reduction_factor)
                        input_x, input_y = rnet_candidate.shape[3], rnet_candidate.shape[2]
                        for scaled_im in image_pyramid:
                            scaled_im = onet_resize(scaled_im)
                            out = onet(scaled_im)
                            y, bbox = out["y_pred"], out["bbox_pred"]
                            y = torch.exp(y)
                            y = y / y.sum()
                            if torch.argmax(y) == 1:
                                bbox[0][0] = torch.clip(bbox[0][0] * input_x, 0, input_x)
                                bbox[0][2] = torch.clip(bbox[0][2] * input_x, 0, input_x)
                                bbox[0][1] = torch.clip(bbox[0][1] * input_y, 0, input_y)
                                bbox[0][3] = torch.clip(bbox[0][3] * input_y, 0, input_y)
                                onet_bboxes.append(torch.round(bbox))
                                onet_scores.append(y[0][1])

                        # Non-maximum suppression
                        if len(onet_bboxes):
                            onet_bboxes = torch.vstack(onet_bboxes)
                            onet_scores = torch.hstack(onet_scores)
                            bboxes_indices = nms(boxes=onet_bboxes, scores=onet_scores, iou_threshold=0.3)
                            onet_bboxes = [onet_bboxes[index] for index in bboxes_indices]
                            onet_scores = torch.tensor([onet_scores[index] for index in bboxes_indices])
                            yx_offset = rnet_candidates_params[rnet_candidate_idx]
                            for idx, bbox in enumerate(onet_bboxes):
                                x, y, w, h = bbox[0].long(), bbox[1].long(), bbox[2].long(), bbox[3].long()
                                onet_candidates_offsets.append((x + yx_offset[0], y + yx_offset[1], w, h))
                                final_bboxes.append(
                                    torch.tensor([x0 + x + yx_offset[0], y0 + y + yx_offset[1], w, h]).float())
                                final_scores.append(onet_scores[idx])

    final_scores = torch.tensor(final_scores).float()
    return final_bboxes, final_scores


def plot_image_with_bounding_box(image, bounding_boxes, freeze=False):
    """
    Display an image with bounding boxes drawn around detected objects.

    Args:
        image (numpy.ndarray): The input image.
        bounding_boxes (list): A list of bounding boxes, where each bounding box is represented as a tuple (x, y, w, h).
        freeze (bool, optional): If True, the image will remain displayed until a key is pressed. Defaults to False.
    """
    # Loop over bounding boxes and draw them on the image
    for box in bounding_boxes:
        x, y, w, h = round(box[0].item()), round(box[1].item()), round(box[2].item()), round(box[3].item())
        cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

    # Display the image with bounding boxes
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(f'Frame', bgr_image)

    if freeze:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.waitKey(1)


def postprocess_bboxes_and_scores(bboxes, scores, detection_th=0.96, iou_th=0.3):
    """
    Postprocesses the bounding boxes and scores.

    Args:
        bboxes (list): List of bounding boxes.
        scores (torch.Tensor): Tensor containing the scores.
        detection_th (float, optional): Detection threshold. Defaults to 0.96.
        iou_th (float, optional): Intersection over Union (IoU) threshold. Defaults to 0.3.

    Returns:
        tuple: A tuple containing the postprocessed bounding boxes and scores.
    """
    # Filter out low confidence detections
    mask = scores > detection_th
    bboxes = [b for m, b in zip(mask, bboxes) if m.item()]
    scores = scores[mask]

    if len(bboxes) == 0:
        return [], []

    # Apply non-maximum suppression
    bboxes = torch.vstack(bboxes)
    selected_indices = nms(boxes=bboxes, scores=scores, iou_threshold=iou_th)
    bboxes = bboxes[selected_indices]
    scores = scores[selected_indices]

    bboxes_iou = IoU(bboxes, bboxes)
    N = len(scores)

    keep_map = np.ones(N)
    for i in range(N):
        if keep_map[i] == 0:
            continue
        for j in range(N):
            if i != j and bboxes_iou[i, j] > iou_th:
                keep_map[j] = 0

    if np.sum(keep_map) == 0:
        return [], []

    keep_indices = np.where(keep_map > 0)[0]
    bboxes = [bboxes[idx] for idx in keep_indices]
    scores = torch.hstack([scores[idx] for idx in keep_indices])

    return bboxes, scores


def live_face_detection(target_fps):
    """
    Perform live face detection using the webcam.

    Args:
        target_fps (int): Target frames per second for the live stream.
        record (bool): Whether to record the video or not.
        output_file (str): Output filename for the recorded video.
    """
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 is the default camera, you can change it if you have multiple cameras

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open the camera")
        return

    # Calculate the delay based on the target FPS
    delay = int(1000 / target_fps)
    frame_idx = 0

    # Loop to continuously capture frames
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Check if the frame is empty
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Measure the time for performance evaluation
        t = time.time()
        # Predict faces in the image and draw bounding boxes
        bboxes, scores = predict_faces_in_image(frame, window_size=[(500, 500)], reduction_factor=0.1)
        bboxes, scores = postprocess_bboxes_and_scores(bboxes, scores, detection_th=0.90, iou_th=0)

        # Print the frames per second
        if not (frame_idx % 1000):
            print(f"FPS={60 / (time.time() - t)}")

        # Display the image with bounding boxes
        plot_image_with_bounding_box(frame, bboxes)

        frame_idx += 1

        # Check for the 'q' key to quit the program
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    # Release the capture and video writer
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Live Face Detection')
    parser.add_argument('--fps', type=int, default=60, help='Target frames per second for the live stream')
    args = parser.parse_args()

    live_face_detection(target_fps=args.fps)
