import dlib
from imutils import face_utils
import cv2
import os
from scipy.spatial import distance as dist
from tqdm import tqdm
import time
import glob
import torch


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def get_blink(lanmarks):
    leftEye = lanmarks[lStart:lEnd]
    rightEye = lanmarks[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    # average the eye aspect ratio together for both eyes
    ear = (leftEAR + rightEAR) / 2.0

    ear = (ear - 0.15) * 5
    if ear < 0:
        ear = 0.0
    if ear > 1:
        ear = 1.0

    return ear


predictor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shape_predictor_68_face_landmarks_GTX.dat")
def get_dlib_detector():
    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(predictor_path)
    return face_detector, face_predictor


def get_landmarks_fix(frame, face_detector=None, face_predictor=None):
    if face_detector is None:
        face_detector = dlib.get_frontal_face_detector()
    if face_predictor is None:
        face_predictor = dlib.shape_predictor(predictor_path)

    faces = [dlib.rectangle(64, 64, 192, 192)]
    landmark = face_predictor(frame, faces[0])
    landmark = face_utils.shape_to_np(landmark)

    return landmark


def get_landmarks_0(frame, face_detector=None, face_predictor=None):
    if face_detector is None:
        face_detector = dlib.get_frontal_face_detector()
    if face_predictor is None:
        face_predictor = dlib.shape_predictor(predictor_path)
    faces = face_detector(frame, 0)

    landmark = face_predictor(frame, faces[0])
    landmark = face_utils.shape_to_np(landmark)

    return landmark


def get_landmarks_1(frame, face_detector=None, face_predictor=None):
    if face_detector is None:
        face_detector = dlib.get_frontal_face_detector()
    if face_predictor is None:
        face_predictor = dlib.shape_predictor(predictor_path)
    faces = face_detector(frame, 1)

    landmark = face_predictor(frame, faces[0])
    landmark = face_utils.shape_to_np(landmark)

    return landmark


def process(face_detector, face_predictor, frame):
    faces = [dlib.rectangle(0, 0, 128, 128)]
    landmark = face_predictor(frame, faces[0])
    landmark = face_utils.shape_to_np(landmark)
    return get_blink(landmark)


def cal_blink_loss(face_detector, face_predictor, g_faces, gt_faces, print_value=False, use_input=None):

    g_res = [process(face_detector, face_predictor, g) for g in g_faces]
    gt_res = [process(face_detector, face_predictor, gt) for gt in gt_faces]

    g_res = torch.tensor(g_res)
    if use_input is not None:
        print("use_input: ", use_input.shape)
        # resample to shrink the size / 3.2
        g_res = torch.cat([use_input[:, :, :, int((i / 5) * 16)] for i in range(5)], dim=0).squeeze(1).squeeze(1).cpu()
    gt_res = torch.tensor(gt_res)

    if print_value:
        print("g_res: ", g_res)
        print("gt_res: ", gt_res)

    return torch.mean((g_res - gt_res) ** 2)


if __name__ == "__main__":
    face_detector, face_predictor = get_dlib_detector()
    frames = [cv2.imread(path) for path in glob.glob("~/workspace/datasets/faces/obama_processed_256/000_005/*.jpg")]
    for frame in frames:
        print(frame.shape)
        start_time = time.time()
        landmark_fix = get_landmarks_fix(frame, face_detector, face_predictor)
        blink_fix = get_blink(landmark_fix)
        landmark_1 = get_landmarks_1(frame, face_detector, face_predictor)
        blink_1 = get_blink(landmark_1)
        print(time.time() - start_time, blink_fix, blink_1)
