import face_detection
import numpy as np
import cv2
from tqdm import tqdm
import torch
import glob
import os
from natsort import natsorted

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_squre_coords(coords, image, size=None, last_size=None):
    y1, y2, x1, x2 = coords
    w, h = x2 - x1, y2 - y1
    center = (x1 + w // 2, y1 + h // 2)
    if size is None:
        size = (w + h) // 2
    if last_size is not None:
        size = (w + h) // 2
        size = (size - last_size) // 5 + last_size
    x1, y1 = center[0] - size // 2, center[1] - size // 2
    x2, y2 = x1 + size, y1 + size
    return size, [y1, y2, x1, x2]


def get_smoothened_boxes(boxes, T=10):
    smooth_factor = 0.1
    smoothened_boxes = [boxes[0]]
    last_box = boxes[0]
    for current_box in boxes[1:]:
        change = (current_box - last_box) * smooth_factor
        current_box = last_box + change
        last_box = current_box
        smoothened_boxes.append(current_box)

    for i in range(len(smoothened_boxes)):
        if i + T > len(smoothened_boxes):
            window = smoothened_boxes[len(smoothened_boxes) - T:]
        else:
            window = smoothened_boxes[i : i + T]
        smoothened_boxes[i] = np.mean(window, axis=0)
    return np.array(smoothened_boxes).astype(np.int32)


def face_detect(images, pads):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)

    batch_size = 16 if device == "cuda" else 4
    print("face detect batch size:", batch_size)
    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i : i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError("Image too big to run face detection on GPU. Please use the --resize_factor argument")
            batch_size //= 2
            print("Recovering from OOM error; New batch size: {}".format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite(".temp/faulty_frame.jpg", image)  # check this frame where the face was not detected.
            raise ValueError("Face not detected! Ensure the video contains a face in all the frames.")

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        # y_gap, x_gap = (y2 - y1) // 2, (x2 - x1) // 2
        y_gap, x_gap = ((y2 - y1)*2)//3, ((x2 - x1)*2)//3

        coords_ = [y1 - y_gap, y2 + y_gap, x1 - x_gap, x2 + x_gap]

        _, coords = get_squre_coords(coords_, image)

        y1, y2, x1, x2 = coords
        y1 = max(0, y1)
        y2 = min(image.shape[0], y2)
        x1 = max(0, x1)
        x2 = min(image.shape[1], x2)

        results.append([x1, y1, x2, y2])

    print("Number of frames cropped: {}".format(len(results)))
    print("First coords: {}".format(results[0]))
    boxes = np.array(results)
    boxes = get_smoothened_boxes(boxes)

    del detector
    return boxes


def padding_black(imgs):
    for i in range(len(imgs)):
        imgs[i] = cv2.vconcat(
            [np.zeros((100, imgs[i].shape[1], 3), dtype=np.uint8), imgs[i], np.zeros((20, imgs[i].shape[1], 3), dtype=np.uint8)]
        )
    return imgs


def remove_padding(img):
    # remove padding, the padding settings are hardcoded in padding_black function above
    return img[100:-20]


def preprocess_video(video_path, full_frames, static=False, pads=[0, 0, 0, 0]):
    full_frames = padding_black(full_frames)
    coords_path = os.path.join(os.path.dirname(video_path), os.path.basename(video_path)[:-4]+".npz")
    if os.path.exists(coords_path):
        print("Existing coords found, loading...")
        coords = np.load(coords_path)["coords"]
        return full_frames, coords
    else:
        print("No existing coords found, running face detection...")
        if not static:
            coords = face_detect(full_frames, pads)
        else:
            coords = face_detect([full_frames[0]], pads)
        np.savez_compressed(coords_path, coords=coords)
        return full_frames, coords
