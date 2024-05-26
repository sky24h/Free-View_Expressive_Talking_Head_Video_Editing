import os
import cv2
import time
import random
import argparse
import traceback
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.train_utils import Timer, check_saved_checkpoints
from utils.data_utils import create_image_lists, get_image_lists
from utils.log_utils import create_logger


global global_step, global_epoch
global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print("use_cuda: {}".format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16

logger = create_logger("train_pose_emotion")
timer = Timer()


class Dataset(object):
    def __init__(self, hparams, data_root, checkpoint_dir, split):
        self.all_images_lists, self.all_videos = get_image_lists(data_root, split)
        self.error_videos = []
        logger.info("{} videos in the {} split".format(len(self.all_videos), split))
        self.hparams = hparams

    def _get_frame_id(self, frame):
        return int(os.path.basename(frame).split(".")[0])

    def _get_window(self, start_frame):
        start_id = self._get_frame_id(start_frame)
        img_extension = os.path.os.path.basename(start_frame).split(".")[-1]
        vidname = os.path.dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = os.path.join(vidname, "{}.{}".format(frame_id, img_extension))
            if not os.path.isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def _crop_audio_window(self, spec, start_frame):
        start_frame_num = self._get_frame_id(start_frame)
        start_idx = int(80.0 * (start_frame_num / float(self.hparams.fps)))
        end_idx = start_idx + syncnet_mel_step_size
        return spec[start_idx:end_idx, :]

    def _resample(self, input_arr, target_len):
        temp_arr = []
        for i_ in range(target_len):
            temp_arr.append(input_arr[int(i_ * (input_arr.shape[0] / target_len))])
        return np.array(temp_arr)

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            # timer.start()
            idx = random.randint(0, len(self.all_videos) - 1)
            img_names = self.all_images_lists[idx]
            vidname = self.all_videos[idx]

            # img_names = list(glob(join(vidname, '*.png')))
            if len(img_names) <= 3 * syncnet_T:
                continue

            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            if random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = img_name
            else:
                y = torch.zeros(1).float()
                chosen = wrong_img_name

            window_fnames = self._get_window(chosen)
            if window_fnames is None:
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)  # [16:-16,16:-16]
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (self.hparams.img_size, self.hparams.img_size))
                except Exception as e:
                    all_read = False
                    traceback.print_exc()
                    break

                window.append(img)

            if not all_read:
                continue

            try:
                melpath = os.path.join(vidname, "mel.npy")
                orig_mel = np.load(melpath).T

                len_        = orig_mel.shape[0]
                pose_ori    = np.load(os.path.join(vidname, "pose.npy"))
                emotion_ori = np.load(os.path.join(vidname, "emotion_face.npy"))
                pose_ori    = self._resample(pose_ori, len_)
                emotion_ori = self._resample(emotion_ori, len_)

            except Exception as e:
                logger.debug(e)
                self.error_videos.append(vidname)
                traceback.print_exc()
                continue

            mel     = self._crop_audio_window(orig_mel.copy(), img_name)
            pose    = self._crop_audio_window(pose_ori.copy(), img_name)
            emotion = self._crop_audio_window(emotion_ori.copy(), img_name)

            if mel.shape[0] != syncnet_mel_step_size:
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.0
            x = x.transpose(2, 0, 1)
            # x = x[:, x.shape[1]//2:]
            scale_factor = self.hparams.img_size // 128
            x = x[:, 64 * scale_factor : -16 * scale_factor, 16 * scale_factor : -16 * scale_factor]

            x       = torch.FloatTensor(x)
            mel     = torch.FloatTensor(mel.T).unsqueeze(0)
            pose    = torch.FloatTensor(pose.T).unsqueeze(0)
            emotion = torch.FloatTensor(emotion.T).unsqueeze(0)
            return x, mel, y, pose, emotion


logloss = nn.BCELoss()


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss


def train(device, syncnet, train_data_loader, test_data_loader, optimizer, audio_wt, pose_wt, emotion_wt, hparams, checkpoint_dir=None):
    global global_step, global_epoch
    resumed_step = global_step

    nepochs                    = hparams.syncnet_nepochs
    syncnet_eval_interval      = hparams.syncnet_eval_interval
    syncnet_save_interval      = hparams.syncnet_initial_save_interval
    syncnet_save_after_nepochs = hparams.syncnet_save_after_nepochs
    syncnet_not_improved_limit = hparams.syncnet_not_improved_limit
    min_loss = float("inf")
    not_improved_count = 0

    # clean logdir
    logdir = os.path.join(checkpoint_dir, "logdir", "syncnet")
    timer.start()
    if checkpoint_dir is not None:
        if not os.path.exists(logdir):
            os.makedirs(logdir)
    else:
        logger.critical("No checkpoint dir")
        exit("checkpoint_dir is None")
    writer = SummaryWriter(log_dir=logdir)
    while global_epoch <= nepochs:
        logger.info("Start training epoch: {}".format(global_epoch))
        running_loss = 0.0
        running_loss_a, running_loss_p, running_loss_e = 0.0, 0.0, 0.0
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, mel, y, pose, emotion) in prog_bar:
            syncnet.train()
            optimizer.zero_grad()

            # Transform data to CUDA device
            x       = x.to(device)
            pose    = pose.to(device)
            emotion = emotion.to(device)
            mel     = mel.to(device)

            a, v, p, e = syncnet(mel, x, pose, emotion)
            y = y.to(device)

            audio_loss   = cosine_loss(a, v, y)  # audio-video
            pose_loss    = cosine_loss(p, v, y)  # pose-video
            emotion_loss = cosine_loss(e, v, y)  # emotion-video
            loss         = audio_wt * audio_loss + pose_wt * pose_loss + emotion_wt * emotion_loss

            loss.backward()
            optimizer.step()

            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()
            running_loss_a += audio_loss.item()
            running_loss_p += pose_loss.item()
            running_loss_e += emotion_loss.item()

            if global_step % hparams.log_interval == 0:
                time_elapsed = timer.stop()
                log_info = "Step: {}, Time elapsed: {:.2f}, Loss: {:.4f}, Audio_loss: {:.4f}, Pose_loss: {:.4f}, Emotion_loss: {:.4f}".format(
                        global_step,
                        time_elapsed,
                        running_loss / (step + 1),
                        running_loss_a / (step + 1),
                        running_loss_p / (step + 1),
                        running_loss_e / (step + 1),
                    )
                prog_bar.set_description(log_info)
                logger.info(log_info)
                timer.start()  # reset timer

            writer.add_scalar("train_loss/loss", running_loss / (step + 1), global_step)
            writer.add_scalar("train_loss/audio_loss", running_loss_a / (step + 1), global_step)
            writer.add_scalar("train_loss/pose_loss", running_loss_p / (step + 1), global_step)
            writer.add_scalar("train_loss/emotion_loss", running_loss_e / (step + 1), global_step)

            if global_step % syncnet_eval_interval == 0:
                with torch.no_grad():
                    averaged_loss, averaged_loss_a, averaged_loss_p, averaged_loss_e = eval_syncnet(
                        test_data_loader, global_epoch, global_step, device, syncnet, audio_wt, pose_wt, emotion_wt, checkpoint_dir
                    )
                writer.add_scalar("test_loss/loss", averaged_loss, global_step)
                writer.add_scalar("test_loss/audio_loss", averaged_loss_a, global_step)
                writer.add_scalar("test_loss/pose_loss", averaged_loss_p, global_step)
                writer.add_scalar("test_loss/emotion_loss", averaged_loss_e, global_step)
                if global_epoch >= syncnet_save_after_nepochs or global_epoch == 0:
                    if min_loss > averaged_loss * 0.99:
                        min_loss = averaged_loss
                        logger.info("saving best model")
                        save_checkpoint(syncnet, optimizer, hparams.save_optimizer_state, global_step, checkpoint_dir, global_epoch, is_best=True)
                        not_improved_count = 0
                    else:
                        logger.info("not improving")
                        not_improved_count += 1

                    if not_improved_count > syncnet_not_improved_limit:
                        logger.info("Early stopped at epoch {} step {}".format(global_step, global_epoch))
                        break
                    else:
                        logger.debug("early stopping not triggered")
                    if global_step % syncnet_save_interval == 0 or global_epoch == 0:
                        save_checkpoint(syncnet, optimizer, hparams.save_optimizer_state, global_step, checkpoint_dir, global_epoch)
                        temp_save_interval = check_saved_checkpoints(checkpoint_dir, prefix="sync")
                        syncnet_save_interval = temp_save_interval if temp_save_interval is not None else syncnet_save_interval
            global_step += 1
        global_epoch += 1
    else:
        if global_epoch == nepochs:
            logger.info("Training completed")
        else:
            logger.info("early stopping triggered")


def eval_syncnet(test_data_loader, global_epoch, global_step, device, syncnet, audio_wt, pose_wt, emotion_wt, checkpoint_dir):
    eval_steps = 1000
    logger.info("Evaluating at epoch {} step {}".format(global_epoch, global_step))
    losses, losses_a, losses_p, losses_e = [], [], [], []

    while 1:
        for step, (x, mel, y, pose, emotion) in enumerate(test_data_loader):
            syncnet.eval()

            # Transform data to CUDA device
            x       = x.to(device)
            pose    = pose.to(device)
            emotion = emotion.to(device)
            mel     = mel.to(device)

            a, v, p, e = syncnet(mel, x, pose, emotion)
            y = y.to(device)

            audio_loss   = cosine_loss(a, v, y)  # audio-video
            pose_loss    = cosine_loss(p, v, y)  # pose-video
            emotion_loss = cosine_loss(e, v, y)  # emotion-video
            loss         = audio_wt * audio_loss + pose_wt * pose_loss + emotion_wt * emotion_loss

            losses.append(loss.item())
            losses_a.append(audio_loss.item())
            losses_p.append(pose_loss.item())
            losses_e.append(emotion_loss.item())

            if step > eval_steps:
                break

        averaged_loss = sum(losses) / len(losses)
        averaged_loss_a = sum(losses_a) / len(losses_a)
        averaged_loss_p = sum(losses_p) / len(losses_p)
        averaged_loss_e = sum(losses_e) / len(losses_e)
        # print(averaged_loss)

        return averaged_loss, averaged_loss_a, averaged_loss_p, averaged_loss_e


def save_checkpoint(model, optimizer, save_optimizer_state, step, checkpoint_dir, epoch, is_best=False):
    if is_best:
        checkpoint_path = os.path.join(checkpoint_dir, "sync_best_model_epoch{:05d}_step{:09d}.pth".format(epoch, step))
    else:
        checkpoint_path = os.path.join(checkpoint_dir, "sync_checkpoint_epoch{:05d}_step{:09d}.pth".format(epoch, step))

    optimizer_state = optimizer.state_dict() if save_optimizer_state else None
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer_state,
            "global_step": step,
            "global_epoch": epoch,
        },
        checkpoint_path,
    )
    logger.info("Saved checkpoint: {}".format(checkpoint_path))


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    logger.info("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            logger.info("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model


if __name__ == "__main__":
    from models import SyncNet_pose_emotion as SyncNet
    from hparams import hparams

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_root", help="Root folder of the preprocessed dataset", required=True)
    parser.add_argument("--checkpoint_dir", help="Save checkpoints to this directory", required=True, type=str)
    parser.add_argument("--checkpoint_path", help="Resumed from this checkpoint", default=None, type=str)
    parser.add_argument("--logdir", help="Logdir", default="./logs", type=str)
    parser.add_argument("--audio_wt", help="audio_loss", required=True, type=float)
    parser.add_argument("--pose_wt", help="pose_loss", required=True, type=float)
    parser.add_argument("--emotion_wt", help="emotion_loss", required=True, type=float)
    args = parser.parse_args()
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir, exist_ok=True)
    logger = create_logger("sync_train_pose_emotion", os.path.join(args.logdir, "sync_train_pose_emotion.log"))

    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path
    data_root = args.data_root

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    print(data_root, checkpoint_dir, hparams.syncnet_batch_size)
    valid_list = create_image_lists(data_root, checkpoint_dir, hparams.syncnet_batch_size)

    # Dataset and Dataloader setup
    train_dataset = Dataset(hparams, data_root, checkpoint_dir, "train")
    test_dataset  = Dataset(hparams, data_root, checkpoint_dir, "val")

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=hparams.syncnet_batch_size,
        shuffle=True,
        num_workers=hparams.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    test_data_loader = DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size, num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True
    )

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    syncnet = SyncNet().to(device)
    print("total trainable params {}".format(sum(p.numel() for p in syncnet.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in syncnet.parameters() if p.requires_grad], lr=hparams.syncnet_lr)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, syncnet, optimizer, reset_optimizer=True)

    if torch.cuda.device_count() > 1:
        syncnet = nn.DataParallel(syncnet).to(device)
    else:
        syncnet = syncnet.to(device)

    train(
        device,
        syncnet,
        train_data_loader,
        test_data_loader,
        optimizer,
        args.audio_wt,
        args.pose_wt,
        args.emotion_wt,
        hparams,
        checkpoint_dir=checkpoint_dir,
    )
