import os

# set CUDA_MODULE_LOADING=LAZY to speed up the serverless function
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
# set SAFETENSORS_FAST_GPU=1 to speed up the serverless function
os.environ["SAFETENSORS_FAST_GPU"] = "1"

import cv2
import time
import torch
import imageio
import numpy as np
from tqdm import tqdm
from glob import glob
from moviepy.editor import AudioFileClip, VideoFileClip

from utils.audio_utils import load_wav, melspectrogram
from models import Generator_pose_emotion_blink as Generator
from preprocess_video import preprocess_video, remove_padding


fps = 25
mel_step_size = 16
sample_rate = 16000
mel_idx_multiplier = 80.0 / fps
gen_batch_size = 64 if torch.cuda.is_available() else 4
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} for inference.".format(device))
use_fp16 = True if torch.cuda.is_available() else False
print("Using FP16 for inference.") if use_fp16 else None
torch.backends.cudnn.benchmark = True if device == "cuda" else False


def frames_and_audio_to_video(frames, audio, output_path, fps=25):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path.replace(".mp4", "_temp.mp4"), fps=fps, quality=8, macro_block_size=1)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    audio_clip = AudioFileClip(audio)
    video_clip = VideoFileClip(output_path.replace(".mp4", "_temp.mp4"))
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(output_path)
    os.remove(output_path.replace(".mp4", "_temp.mp4"))
    return output_path


def make_mask(image_size=256, border_size=32):
    mask_bar = np.linspace(1, 0, border_size).reshape(1, -1).repeat(image_size, axis=0)
    mask = np.zeros((image_size, image_size), dtype=np.float32)
    mask[-border_size:, :] += mask_bar.T[::-1]
    mask[:, :border_size] = mask_bar
    mask[:, -border_size:] = mask_bar[:, ::-1]
    mask[-border_size:, :][mask[-border_size:, :] < 0.6] = 0.6
    mask = np.stack([mask] * 3, axis=-1).astype(np.float32)
    return mask


face_mask = make_mask()


def blend_images(foreground, background):
    # Blend the foreground and background images using the mask
    temp_mask  = cv2.resize(face_mask, (foreground.shape[1], foreground.shape[0]))
    blended    = cv2.multiply(foreground.astype(np.float32), temp_mask)
    blended   += cv2.multiply(background.astype(np.float32), 1 - temp_mask)
    blended    = np.clip(blended, 0, 255).astype(np.uint8)
    return blended


def resample(input_attributes, length):
    input_attributes = np.array(input_attributes)
    resized_attributes = [input_attributes[int(i_ * (input_attributes.shape[0] / length))] for i_ in range(length)]
    return np.array(resized_attributes).T


def preprocess_batch(batch):
    return torch.FloatTensor(np.reshape(batch, [len(batch), 1, batch[0].shape[0], batch[0].shape[1]])).to(device)


def datagen(face_path, frames, mels, poses, emotions, blinks, static=False, img_size=256, pads=[0, 0, 0, 0]):
    img_batch, mel_batch, pose_batch, emotion_batch, blink_batch, frame_batch, coords_batch = [], [], [], [], [], [], []
    scale_factor = img_size // 128

    frames = frames[: len(mels)]
    frames, coords = preprocess_video(face_path, frames, static, pads)
    face_det_results = [[image[y1:y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(frames, coords)]
    face_det_results = face_det_results[: len(mels)]

    while len(frames) < len(mels):
        face_det_results = face_det_results + face_det_results[::-1]
        frames = frames + frames[::-1]
    else:
        face_det_results = face_det_results[: len(mels)]
        frames = frames[: len(mels)]

    for i in range(len(mels)):
        idx = 0 if static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()
        face = cv2.resize(face, (img_size, img_size))

        img_batch.append(face)
        mel_batch.append(mels[i])
        pose_batch.append(poses[i])
        emotion_batch.append(emotions[i])
        blink_batch.append(blinks[i])
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        # print(m.shape, poses[i].shape)
        # (80, 16) (3, 16)
        if len(img_batch) >= gen_batch_size:
            img_masked = np.asarray(img_batch).copy()
            img_masked[:, 16 * scale_factor : -16 * scale_factor, 16 * scale_factor : -16 * scale_factor] = 0.0

            img_batch     = np.concatenate((img_masked, img_batch), axis=3) / 255.0
            img_batch     = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch     = preprocess_batch(mel_batch)
            pose_batch    = preprocess_batch(pose_batch)
            emotion_batch = preprocess_batch(emotion_batch)
            blink_batch   = preprocess_batch(blink_batch)

            if use_fp16:
                yield (
                    img_batch.half(),
                    mel_batch.half(),
                    pose_batch.half(),
                    emotion_batch.half(),
                    blink_batch.half(),
                ), frame_batch, coords_batch
            else:
                yield (img_batch, mel_batch, pose_batch, emotion_batch, blink_batch), frame_batch, coords_batch
            img_batch, mel_batch, pose_batch, emotion_batch, blink_batch, frame_batch, coords_batch = [], [], [], [], [], [], []

    if len(img_batch) > 0:
        img_masked = np.asarray(img_batch).copy()
        img_masked[:, 16 * scale_factor : -16 * scale_factor, 16 * scale_factor : -16 * scale_factor] = 0.0

        img_batch     = np.concatenate((img_masked, img_batch), axis=3) / 255.0
        img_batch     = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch     = preprocess_batch(mel_batch)
        pose_batch    = preprocess_batch(pose_batch)
        emotion_batch = preprocess_batch(emotion_batch)
        blink_batch   = preprocess_batch(blink_batch)

        if use_fp16:
            yield (img_batch.half(), mel_batch.half(), pose_batch.half(), emotion_batch.half(), blink_batch.half()), frame_batch, coords_batch
        else:
            yield (img_batch, mel_batch, pose_batch, emotion_batch, blink_batch), frame_batch, coords_batch


def _load(checkpoint_path):
    if device == "cuda":
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def load_generator(checkpoint_path):
    generator = Generator()
    if checkpoint_path.endswith(".pth") or checkpoint_path.endswith(".ckpt"):
        if device == "cuda":
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        s = checkpoint["state_dict"]
    else:
        from safetensors import safe_open

        s = {}
        with safe_open(checkpoint_path, framework="pt", device=device) as f:
            for key in f.keys():
                s[key] = f.get_tensor(key)
    new_s = {}
    for k, v in s.items():
        new_s[k.replace("module.", "")] = v
    generator.load_state_dict(new_s)

    generator = generator.to(device)
    generator.eval()
    print("Model loaded")
    if use_fp16:
        for name, module in generator.named_modules():
            if ".query_conv" in name or ".key_conv" in name or ".value_conv" in name:
                # keep attention layers in full precision to avoid error
                module.to(torch.float)
            else:
                module.to(torch.half)
        print("Generator converted to half precision to accelerate inference")
    return generator


def output_chunks(input_attributes):
    output_chunks = []
    len_ = len(input_attributes[0])

    i = 0
    # print(mel.shape, pose.shape)
    # (80, 801) (3, 801)
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len_:
            output_chunks.append(input_attributes[:, len_ - mel_step_size :])
            break
        output_chunks.append(input_attributes[:, start_idx : start_idx + mel_step_size])
        i += 1
    return output_chunks


def prepare_data(face_path, audio_path, pose, emotion, blink, img_size=256, pads=[0, 0, 0, 0]):
    if os.path.isfile(face_path) and face_path.split(".")[1] in ["jpg", "png", "jpeg"]:
        static = True
        full_frames = [cv2.imread(face_path)]
    else:
        static = False
        video_stream = cv2.VideoCapture(face_path)

        # print('Reading video frames...')
        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            full_frames.append(frame)
    print("Number of frames available for inference: " + str(len(full_frames)))

    wav  = load_wav(audio_path, sample_rate)
    mel  = melspectrogram(wav=wav)
    len_ = mel.shape[1]
    mel  = mel[:, :len_]
    # print('>>>', mel.shape)

    pose    = resample(pose, len_)
    emotion = resample(emotion, len_)
    blink   = resample(blink, len_)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError("Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again")

    mel_chunks     = output_chunks(mel)
    pose_chunks    = output_chunks(pose)
    emotion_chunks = output_chunks(emotion)
    blink_chunks   = output_chunks(blink)

    gen = datagen(face_path, full_frames, mel_chunks, pose_chunks, emotion_chunks, blink_chunks, static=static, img_size=img_size, pads=pads)
    steps = int(np.ceil(float(len(mel_chunks)) / gen_batch_size))

    return gen, steps


def inference(checkpoint_path, gen, steps, audio_path, outfile, subtitle):
    n = 0
    output_frames = []
    generator = load_generator(checkpoint_path)
    print("Generator loaded")
    for i, (inputs, frames, coords) in enumerate(tqdm(gen, total=steps)):
        with torch.no_grad():
            pred = generator(*inputs)
        if i == 0:
            frame_h, frame_w = frames[0].shape[:-1]
            # remove the padding, if any
            frame_h -= 120
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            y1, y2, x1, x2 = int(y1), int(y2), int(x1), int(x2)
            y = round(y2 - y1)
            x = round(x2 - x1)
            p = cv2.resize(p.astype(np.uint8), (x, y))
            try:
                f[y1 : y1 + y, x1 : x1 + x] = blend_images(f[y1 : y1 + y, x1 : x1 + x], p)
            except Exception as e:
                print(e)
                f[y1 : y1 + y, x1 : x1 + x] = p
            f = cv2.resize(remove_padding(f), (frame_w, frame_h))
            if subtitle is not None:
                cv2.putText(f, subtitle[n], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            output_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            n += 1

    frames_and_audio_to_video(output_frames, audio_path, outfile, fps=fps)
    print("Inference completed. Video saved at", outfile)


