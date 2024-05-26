import os
import cv2
import glob
import math
import random
import imageio
import subprocess
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip, AudioFileClip

import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
use_cuda = torch.cuda.is_available()


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


logloss = nn.BCELoss()


def cosine_loss(x, v, y):  # x can be audio or emotion
    d = nn.functional.cosine_similarity(x, v)
    # loss = logloss(d.unsqueeze(1), y)
    return 1 - d


def load_checkpoint(model, path):
    # print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace("module.", "")] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model


def preprocess_mel_pose_emotion_blink(mel, pose, emotion, blink):
    mel_idx_multiplier = 80.0 / 25
    mel_step_size = 16
    mel_batch, pose_batch = [], []

    mel_chunks     = []
    pose_chunks    = []
    emotion_chunks = []
    blink_chunks   = []
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size :])
            pose_chunks.append(pose[:, len(pose[0]) - mel_step_size :])
            emotion_chunks.append(emotion[:, len(emotion[0]) - mel_step_size :])
            blink_chunks.append(blink[:, len(blink[0]) - mel_step_size :])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        pose_chunks.append(pose[:, start_idx : start_idx + mel_step_size])
        emotion_chunks.append(emotion[:, start_idx : start_idx + mel_step_size])
        blink_chunks.append(blink[:, start_idx : start_idx + mel_step_size])
        i += 1

    mel_batch     = np.asarray(mel_chunks)
    pose_batch    = np.asarray(pose_chunks)
    emotion_batch = np.asarray(emotion_chunks)
    blink_batch   = np.asarray(blink_chunks)

    mel_batch     = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
    pose_batch    = np.reshape(pose_batch, [len(pose_batch), pose_batch.shape[1], pose_batch.shape[2], 1])
    emotion_batch = np.reshape(emotion_batch, [len(emotion_batch), emotion_batch.shape[1], emotion_batch.shape[2], 1])
    blink_batch   = np.reshape(blink_batch, [len(blink_batch), blink_batch.shape[1], blink_batch.shape[2], 1])

    mel_batch     = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
    pose_batch    = torch.FloatTensor(np.transpose(pose_batch, (0, 3, 1, 2))).to(device)
    emotion_batch = torch.FloatTensor(np.transpose(emotion_batch, (0, 3, 1, 2))).to(device)
    blink_batch   = torch.FloatTensor(np.transpose(blink_batch, (0, 3, 1, 2))).to(device)

    return mel_batch, pose_batch, emotion_batch, blink_batch


def preprocess_faces(faces):
    img_batches = []
    scale_factor = faces[0].shape[0] // 128

    img_batches = np.asarray(faces)
    masked = img_batches.copy()
    masked[:, 16 * scale_factor : -16 * scale_factor, 16 * scale_factor : -16 * scale_factor] = 0.0
    # print(masked.shape)
    img_batches = np.concatenate([masked, img_batches], axis=3) / 255.0

    return torch.FloatTensor(np.transpose(img_batches, (0, 3, 1, 2))).to(device)  # torch.FloatTensor(np.array(img_batches)).to(device)


def split_batches(img_batch, mel_batch, pose_batch):
    batch_size = 16
    img_batches = []
    mel_batches = []
    pose_batches = []

    batch_nums = math.ceil(mel_batch.shape[0] / batch_size)
    for i in range(batch_nums):
        from_ = i * batch_size
        to_ = (i + 1) * batch_size
        if to_ > img_batch.shape[0]:
            to_ = img_batch.shape[0]
            from_ = to_ - batch_size
        img_batches.append(img_batch[from_:to_])
        mel_batches.append(mel_batch[from_:to_])
        pose_batches.append(pose_batch[from_:to_])

    return img_batches, mel_batches, pose_batches


def split_batches_(img_batch, mel_batch, pose_batch, blink_batch):
    batch_size = 16
    img_batches = []
    mel_batches = []
    pose_batches = []
    blink_batches = []

    batch_nums = math.ceil(mel_batch.shape[0] / batch_size)
    for i in range(batch_nums):
        from_ = i * batch_size
        to_ = (i + 1) * batch_size
        if to_ > img_batch.shape[0]:
            to_ = img_batch.shape[0]
            from_ = to_ - batch_size
        img_batches.append(img_batch[from_:to_])
        mel_batches.append(mel_batch[from_:to_])
        pose_batches.append(pose_batch[from_:to_])
        blink_batches.append(blink_batch[from_:to_])

    return img_batches, mel_batches, pose_batches, blink_batches


def split_batches_(img_batch, mel_batch, pose_batch, emotion_batch, blink_batch):
    batch_size = 16
    img_batches     = []
    mel_batches     = []
    pose_batches    = []
    emotion_batches = []
    blink_batches   = []

    batch_nums = math.ceil(mel_batch.shape[0] / batch_size)
    for i in range(batch_nums):
        from_ = i * batch_size
        to_ = (i + 1) * batch_size
        if to_ > img_batch.shape[0]:
            to_ = img_batch.shape[0]
            from_ = to_ - batch_size
        img_batches.append(img_batch[from_:to_])
        mel_batches.append(mel_batch[from_:to_])
        pose_batches.append(pose_batch[from_:to_])
        emotion_batches.append(emotion_batch[from_:to_])
        blink_batches.append(blink_batch[from_:to_])

    return img_batches, mel_batches, pose_batches, emotion_batches, blink_batches


def resample(input_arr, target_len):
    temp_arr = []
    for i_ in range(target_len):
        temp_arr.append(input_arr[int(i_ * (input_arr.shape[0] / target_len))])
    return np.array(temp_arr)


def validate_generator(
    data_root, images_lists, videos_list, generator, output_dir, img_size=256, random_mel=False, how_many_used_for_validate=10
):
    os.makedirs(output_dir, exist_ok=True)
    valid_num = 0
    temp_list = list(zip(images_lists, videos_list))
    random.shuffle(temp_list)
    temp_images_lists, temp_videos_list = zip(*temp_list)

    for i, faces in enumerate(temp_images_lists):
        if valid_num >= how_many_used_for_validate:
            break

        if random_mel:
            save_path = os.path.join(output_dir, str(valid_num).zfill(2) + "_random_temp.mp4")
        else:
            save_path = os.path.join(output_dir, str(valid_num).zfill(2) + "_temp.mp4")

        try:
            if random_mel:
                random_dir_1 = random.choice(temp_videos_list)
                random_dir_2 = random.choice(temp_videos_list)
                random_dir_3 = random.choice(temp_videos_list)
                mel     = np.load(os.path.join(random_dir_1, "mel.npy")).T
                pose    = np.load(os.path.join(random_dir_2, "pose.npy"))
                audio   = os.path.join(random_dir_1, "audio.wav")
                emotion = np.load(os.path.join(random_dir_3, "emotion_img.npy"))
                blink   = np.load(os.path.join(random_dir_3, "blink.npy")).reshape(-1, 1)
            else:
                dir_    = os.path.join(data_root, temp_videos_list[i])
                mel     = np.load(os.path.join(dir_, "mel.npy")).T
                pose    = np.load(os.path.join(dir_, "pose.npy"))
                audio   = os.path.join(dir_, "audio.wav")
                emotion = np.load(os.path.join(dir_, "emotion_img.npy"))
                blink   = np.load(os.path.join(dir_, "blink.npy")).reshape(-1, 1)

            if len(faces) > 100 and len(faces) < 1000 and mel.shape[0] > 320 and mel.shape[0] < 3200:
                valid_num += 1
                out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"DIVX"), 25, (img_size, img_size))
                writer = imageio.get_writer(save_path, fps=25, macro_block_size=1, quality=8, codec="libx264")
            else:
                continue

            len_    = mel.shape[0]
            pose    = resample(pose, len_)
            emotion = resample(emotion, len_)
            blink   = resample(blink, len_)
            # print(len(faces), mel.shape, pose.shape)

            faces = np.array([cv2.resize(cv2.imread(f), (img_size, img_size)) for f in faces])
            img_batch = preprocess_faces(faces)

            mel_batch, pose_batch, emotion_batch, blink_batch = preprocess_mel_pose_emotion_blink(mel.T, pose.T, emotion.T, blink.T)

            # print(img_batch.shape, mel_batch.shape, pose_batch.shape)
            if mel_batch.shape[0] > img_batch.shape[0]:
                mel_batch     = mel_batch[: img_batch.shape[0]]
                pose_batch    = pose_batch[: img_batch.shape[0]]
                emotion_batch = emotion_batch[: img_batch.shape[0]]
                blink_batch   = blink_batch[: img_batch.shape[0]]

            else:
                img_batch = img_batch[: mel_batch.shape[0]]
            # print(img_batch.shape, mel_batch.shape, pose_batch.shape)

            img_batches, mel_batches, pose_batches, emotion_batches, blink_batches = split_batches_(
                img_batch, mel_batch, pose_batch, emotion_batch, blink_batch
            )
            for i in tqdm(range(len(img_batches))):
                pred = generator(mel_batches[i], img_batches[i], pose_batches[i], emotion_batches[i], blink_batches[i])
                pred = pred.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
                # print(mel_batches.shape, img_batches.shape, v.shape, y.shape)
                for p in pred:
                    p = cv2.resize(p.astype(np.uint8), (img_size, img_size))
                    writer.append_data(cv2.cvtColor(p, cv2.COLOR_RGB2BGR))
            writer.close()
            video = VideoFileClip(save_path)
            audio = AudioFileClip(audio)
            video = video.set_audio(audio)
            video.write_videofile(save_path.replace("_temp", ""), codec="libx264", audio_codec="aac")
            os.remove(save_path)
        except Exception as e:
            print(e)


# if __name__ == "__main__":
#     from models import Generator_pose_emotion_blink as Generator
#     from utils.data_utils import get_image_lists

#     # test code
#     data_root = '~/workspace/datasets/faces/obama_processed_256'
#     images_lists, videos_list = get_image_lists(data_root, 'val')
#     generator = Generator().to(device)
#     for p in generator.parameters():
#         p.requires_grad = False

#     checkpoint_path = ''
#     save_dir = './.temp/val_videos'
#     os.makedirs(save_dir, exist_ok=True)
#     generator = load_checkpoint(generator, checkpoint_path)
#     validate_generator(data_root, images_lists, videos_list, generator, save_dir)
