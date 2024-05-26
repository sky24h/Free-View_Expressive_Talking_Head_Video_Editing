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
from utils.log_utils import create_logger
from utils.data_utils import create_image_lists, get_image_lists
from utils.eye_utils import get_dlib_detector, cal_blink_loss
from validate_generator import validate_generator

logger = create_logger("train_pose_emotion_blink")
timer = Timer()

global global_step, global_epoch
global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
logger.info("use_cuda: {}".format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



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

    def _read_window(self, window_fnames):
        if window_fnames is None:
            return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)  # [16:-16,16:-16]
            if img is None:
                return None
            try:
                img = cv2.resize(img, (self.hparams.img_size, self.hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def _crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self._get_frame_id(start_frame)
        start_idx = int(80.0 * (start_frame_num / float(self.hparams.fps)))
        end_idx = start_idx + syncnet_mel_step_size
        return spec[start_idx:end_idx, :]

    def _get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self._get_frame_id(start_frame) + 1  # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0:
            return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self._crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def _prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.0
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def _resample(self, input_arr, target_len):
        temp_arr = []
        for i_ in range(target_len):
            temp_arr.append(input_arr[int(i_ * (input_arr.shape[0] / target_len))])
        return np.array(temp_arr)

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            img_names = self.all_images_lists[idx]
            vidname = self.all_videos[idx]

            if len(img_names) <= 3 * syncnet_T:
                continue

            img_name       = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            window_fnames       = self._get_window(img_name)
            wrong_window_fnames = self._get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue

            window = self._read_window(window_fnames)
            if window is None:
                continue

            wrong_window = self._read_window(wrong_window_fnames)
            if wrong_window is None:
                continue

            try:
                melpath = os.path.join(vidname, "mel.npy")
                orig_mel = np.load(melpath).T

                len_        = orig_mel.shape[0]
                pose_ori    = np.load(os.path.join(vidname, "pose.npy"))
                emotion_ori = np.load(os.path.join(vidname, "emotion_img.npy"))
                blink_ori   = np.load(os.path.join(vidname, "blink.npy")).reshape(-1, 1)
                pose_ori    = self._resample(pose_ori, len_)
                emotion_ori = self._resample(emotion_ori, len_)
                blink_ori   = self._resample(blink_ori, len_)
            except Exception as e:
                logger.debug(e)
                self.error_videos.append(vidname)
                traceback.print_exc()
                continue

            mel     = self._crop_audio_window(orig_mel.copy(), img_name)
            pose    = self._crop_audio_window(pose_ori.copy(), img_name)
            emotion = self._crop_audio_window(emotion_ori.copy(), img_name)
            blink   = self._crop_audio_window(blink_ori.copy(), img_name)

            if mel.shape[0] != syncnet_mel_step_size:
                continue

            indiv_mels     = self._get_segmented_mels(orig_mel.copy(), img_name)
            indiv_poses    = self._get_segmented_mels(pose_ori.copy(), img_name)
            indiv_emotions = self._get_segmented_mels(emotion_ori.copy(), img_name)
            indiv_blinks   = self._get_segmented_mels(blink_ori.copy(), img_name)
            if indiv_mels is None:
                continue
            if indiv_poses is None:
                continue
            if indiv_emotions is None:
                continue
            if indiv_blinks is None:
                continue

            window = self._prepare_window(window)
            y = window.copy()
            # window[:, :, window.shape[2]//2:] = 0.
            # window[:, :, :, :] = 0.
            scale_factor = self.hparams.img_size // 128
            window[:, :, 16 * scale_factor : -16 * scale_factor, 16 * scale_factor : -16 * scale_factor] = 0.0

            wrong_window = self._prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)

            x       = torch.FloatTensor(x)
            mel     = torch.FloatTensor(mel.T).unsqueeze(0)
            pose    = torch.FloatTensor(pose.T).unsqueeze(0)
            emotion = torch.FloatTensor(emotion.T).unsqueeze(0)
            blink   = torch.FloatTensor(blink.T).unsqueeze(0)

            indiv_mels     = torch.FloatTensor(indiv_mels).unsqueeze(1)
            indiv_poses    = torch.FloatTensor(indiv_poses).unsqueeze(1)
            indiv_emotions = torch.FloatTensor(indiv_emotions).unsqueeze(1)
            indiv_blinks   = torch.FloatTensor(indiv_blinks).unsqueeze(1)
            y              = torch.FloatTensor(y)

            return x, indiv_mels, indiv_poses, indiv_emotions, indiv_blinks, mel, pose, emotion, blink, y


def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    x  = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.0).astype(np.uint8)
    g  = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.0).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.0).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = os.path.join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder):
        os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite("{}/{}_{}.jpg".format(folder, batch_idx, t), c[t], [int(cv2.IMWRITE_JPEG_QUALITY), 95])


logloss = nn.BCELoss()


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)
    return loss


recon_loss = nn.L1Loss()
face_detector, face_predictor = get_dlib_detector()


def get_blink_loss(face_detector, face_predictor, g_faces, gt_faces):
    g_faces  = torch.cat([g_faces[:, :, i] for i in range(g_faces.size(2))], dim=0)
    gt_faces = torch.cat([gt_faces[:, :, i] for i in range(gt_faces.size(2))], dim=0)
    g_faces  = np.array(g_faces[:, :, 64:-64, 64:-64].permute(0, 2, 3, 1).detach().cpu().numpy() * 255, dtype=np.uint8)
    gt_faces = np.array(gt_faces[:, :, 64:-64, 64:-64].permute(0, 2, 3, 1).detach().cpu().numpy() * 255, dtype=np.uint8)
    loss     = cal_blink_loss(face_detector, face_predictor, g_faces, gt_faces)
    return loss


def get_sync_loss(syncnet, mel, g, pose, emotion):
    g = g[:, :, :, g.size(3) // 2 :]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v, t, e = syncnet(mel, g, pose, emotion)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y), cosine_loss(t, v, y), cosine_loss(e, v, y)


def train(
    device,
    data_root,
    generator,
    syncnet,
    vgg_model,
    train_data_loader,
    test_data_loader,
    optimizer,
    audio_wt,
    pose_wt,
    emotion_wt,
    vgg_wt,
    hparams,
    checkpoint_dir=None,
):
    global global_step, global_epoch
    resumed_step = global_step

    nepochs                      = hparams.generator_nepochs
    generator_save_interval      = hparams.generator_initial_save_interval
    generator_eval_interval      = hparams.generator_eval_interval
    generator_sample_interval    = hparams.generator_sample_interval
    generator_save_after_nepochs = hparams.generator_save_after_nepochs
    images_lists, videos_list = get_image_lists(data_root, "val")


    logdir = os.path.join(checkpoint_dir, "logdir", "generator")
    if checkpoint_dir is not None:
        if not os.path.exists(logdir):
            os.makedirs(logdir)
    else:
        logger.critical("No checkpoint dir")
        exit("checkpoint_dir is None")

    writer = SummaryWriter(log_dir=logdir)
    timer.start()
    min_loss = float("inf")
    blink_wt = 0.0 # is initially zero, will be set automatically later. Leads to faster convergence.
    generator_syncnet_wt = 0.0 # is initially zero, will be set automatically later. Leads to faster convergence.

    while global_epoch < nepochs:
        logger.info("Starting Epoch: {}".format(global_epoch))
        running_sync_loss, running_l1_loss = 0.0, 0.0
        running_sync_loss_a, running_sync_loss_p, running_sync_loss_e = 0.0, 0.0, 0.0
        running_blink_loss, running_vgg_loss = 0.0, 0.0
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, indiv_mels, indiv_poses, indiv_emotions, indiv_blinks, mel, pose, emotion, blink, gt) in prog_bar:
            generator.train()

            x              = x.to(device)
            mel            = mel.to(device)
            pose           = pose.to(device)
            emotion        = emotion.to(device)
            blink          = blink.to(device)
            gt             = gt.to(device)

            indiv_mels     = indiv_mels.to(device)
            indiv_poses    = indiv_poses.to(device)
            indiv_emotions = indiv_emotions.to(device)
            indiv_blinks   = indiv_blinks.to(device)

            ### Train generator now. Remove ALL grads.
            optimizer.zero_grad()

            g = generator(indiv_mels, x, indiv_poses, indiv_emotions, indiv_blinks)

            if generator_syncnet_wt > 0.0:
                scale_factor = hparams.img_size // 128
                sync_loss_a, sync_loss_p, sync_loss_e = get_sync_loss(
                    syncnet, mel, g[:, :, :, 16 * scale_factor : -16 * scale_factor, 16 * scale_factor : -16 * scale_factor], pose, emotion
                )
                sync_loss = audio_wt * sync_loss_a + pose_wt * sync_loss_p + emotion_wt * sync_loss_e
            else:
                sync_loss = 0.0

            if blink_wt > 0.0:
                blink_loss = get_blink_loss(face_detector, face_predictor, g, gt)
            else:
                blink_loss = 0.0

            l1loss = recon_loss(g, gt)
            vgg_loss = vgg_model(g, gt).mean()

            loss = (
                generator_syncnet_wt * sync_loss
                + vgg_wt * vgg_loss
                + blink_wt * blink_loss
                + (20.0 - generator_syncnet_wt - vgg_wt) * l1loss
            )

            loss.backward()
            optimizer.step()

            # Logs
            running_l1_loss  += l1loss.detach().item()
            running_vgg_loss += vgg_loss.detach().item()

            if generator_syncnet_wt > 0.0:
                running_sync_loss   += sync_loss.detach().item()
                running_sync_loss_a += sync_loss_a.detach().item()
                running_sync_loss_p += sync_loss_p.detach().item()
                running_sync_loss_e += sync_loss_e.detach().item()
            else:
                running_sync_loss   += 0.0
                running_sync_loss_a += 0.0
                running_sync_loss_p += 0.0
                running_sync_loss_e += 0.0

            if blink_wt > 0.0:
                running_blink_loss += blink_loss.detach().item()
            else:
                running_blink_loss += 0.0

            if global_step % hparams.log_interval == 0:
                time_elapsed = timer.stop()
                log_info = "Step: {}, Time elapsed: {:.2f}, L1: {:.4f}, Sync_audio: {:.4f}, Sync_pose: {:.4f}, Sync_emotion: {:.4f}, Blink: {:.4f}, VGG: {:.4f}".format(
                        global_step,
                        time_elapsed,
                        running_l1_loss / (step + 1),
                        running_sync_loss_a / (step + 1),
                        running_sync_loss_p / (step + 1),
                        running_sync_loss_e / (step + 1),
                        running_blink_loss / (step + 1),
                        running_vgg_loss / (step + 1),
                    )
                prog_bar.set_description(log_info)
                logger.info(log_info)
                timer.start()

            writer.add_scalar("train_loss/l1loss", running_l1_loss / (step + 1), global_step)
            writer.add_scalar("train_loss/sync_loss", running_sync_loss / (step + 1), global_step)
            writer.add_scalar("train_loss/sync_loss_audio", running_sync_loss_a / (step + 1), global_step)
            writer.add_scalar("train_loss/sync_loss_pose", running_sync_loss_p / (step + 1), global_step)
            writer.add_scalar("train_loss/sync_loss_emotion", running_sync_loss_e / (step + 1), global_step)
            writer.add_scalar("train_loss/blink_loss", running_blink_loss / (step + 1), global_step)
            writer.add_scalar("train_loss/vgg_loss", running_vgg_loss / (step + 1), global_step)

            if global_step % generator_eval_interval == 0:
                save_sample_images(x, g, gt, global_step, checkpoint_dir)
                with torch.no_grad():
                    (
                        averaged_l1_loss,
                        averaged_sync_loss,
                        averaged_sync_loss_a,
                        averaged_sync_loss_p,
                        averaged_sync_loss_e,
                        averaged_blink_loss,
                        averaged_vgg_loss,
                    ) = eval_generator(
                        test_data_loader,
                        global_epoch,
                        global_step,
                        device,
                        generator,
                        generator_syncnet_wt,
                        audio_wt,
                        pose_wt,
                        vgg_wt,
                        emotion_wt,
                        hparams,
                        syncnet,
                        vgg_model,
                    )
                if averaged_sync_loss_a < 0.6:
                    generator_syncnet_wt = hparams.generator_syncnet_wt
                    blink_wt = hparams.blink_wt

                writer.add_scalar("test_loss/l1_loss", averaged_l1_loss, global_step)
                writer.add_scalar("test_loss/sync_loss", averaged_sync_loss, global_step)
                writer.add_scalar("test_loss/sync_loss_audio", averaged_sync_loss_a, global_step)
                writer.add_scalar("test_loss/sync_loss_pose", averaged_sync_loss_p, global_step)
                writer.add_scalar("test_loss/sync_loss_emotion", averaged_sync_loss_e, global_step)
                writer.add_scalar("test_loss/vgg_loss", averaged_vgg_loss, global_step)
                if global_epoch > generator_save_after_nepochs or global_epoch == 0:
                    if min_loss > averaged_sync_loss:
                        min_loss = averaged_sync_loss
                        logger.info("saving best model")
                        save_checkpoint(
                            generator, optimizer, hparams.save_optimizer_state, global_step, checkpoint_dir, global_epoch, is_best=True
                        )
                        not_improved_count = 0
                    else:
                        logger.info("not improving")
                        not_improved_count += step

            if global_step % generator_sample_interval == 0:
                try:
                    save_dir = os.path.join(checkpoint_dir, "samples_step{:09d}".format(global_step), "videos")
                    validate_generator(data_root, images_lists, videos_list, generator, save_dir, random_mel=False)
                    validate_generator(data_root, images_lists, videos_list, generator, save_dir, random_mel=True)
                except Exception as e:
                    logger.error("Error in saving videos: {}".format(e))

            if global_step % generator_save_interval == 0 or global_step == 0:
                save_checkpoint(generator, optimizer, hparams.save_optimizer_state, global_step, checkpoint_dir, global_epoch, prefix="gen_")
                temp_save_interval = check_saved_checkpoints(checkpoint_dir, prefix="gen")
                generator_save_interval = temp_save_interval if temp_save_interval is not None else generator_save_interval

            global_step += 1
        global_epoch += 1

        if not_improved_count > 100000:
            logger.info('Early stopped at step {}, epoch {}'.format(global_step, global_epoch))
            break
        else:
            logger.debug('early stopping not triggered')
    else:
        if global_epoch == nepochs:
            logger.info("Training completed")
        else:
            logger.info("early stopping triggered")

def eval_generator(
    test_data_loader, global_epoch, global_step, device, generator, generator_syncnet_wt, audio_wt, pose_wt, emotion_wt, vgg_wt, hparams, syncnet, vgg_model
):
    eval_steps = 200
    logger.info("Evaluating at epoch {} step {}".format(global_epoch, global_step))
    # emotion_wt = 1.0 - audio_wt - pose_wt

    l1_losses, vgg_losses = [], []
    sync_losses, sync_losses_e = [], []
    sync_losses_a, sync_losses_p = [], []
    blink_losses = []

    while 1:
        for step, (x, indiv_mels, indiv_poses, indiv_emotions, indiv_blinks, mel, pose, emotion, blink, gt) in enumerate(test_data_loader):
            generator.eval()

            x       = x.to(device)
            mel     = mel.to(device)
            pose    = pose.to(device)
            emotion = emotion.to(device)
            blink   = blink.to(device)
            gt      = gt.to(device)

            indiv_mels     = indiv_mels.to(device)
            indiv_poses    = indiv_poses.to(device)
            indiv_emotions = indiv_emotions.to(device)
            indiv_blinks   = indiv_blinks.to(device)

            g = generator(indiv_mels, x, indiv_poses, indiv_emotions, indiv_blinks)

            scale_factor = hparams.img_size // 128
            sync_loss_a, sync_loss_p, sync_loss_e = get_sync_loss(
                syncnet, mel, g[:, :, :, 16 * scale_factor : -16 * scale_factor, 16 * scale_factor : -16 * scale_factor], pose, emotion
            )
            sync_loss  = audio_wt * sync_loss_a + pose_wt * sync_loss_p + emotion_wt * sync_loss_e
            blink_loss = get_blink_loss(face_detector, face_predictor, g, gt)
            l1loss     = recon_loss(g, gt)
            vgg_loss   = vgg_model(g, gt).mean()

            loss = (
                generator_syncnet_wt * sync_loss
                + vgg_wt * vgg_loss
                + 10 * blink_loss
                + (20.0 - generator_syncnet_wt - vgg_wt) * l1loss
            )

            l1_losses.append(l1loss.detach().item())
            vgg_losses.append(vgg_loss.detach().item())
            sync_losses.append(sync_loss.detach().item())
            sync_losses_a.append(sync_loss_a.detach().item())
            sync_losses_p.append(sync_loss_p.detach().item())
            sync_losses_e.append(sync_loss_e.detach().item())
            blink_losses.append(blink_loss.detach().item())

            if step > eval_steps:
                break

            averaged_l1_loss     = sum(l1_losses) / len(l1_losses)
            averaged_vgg_loss    = sum(vgg_losses) / len(vgg_losses)
            averaged_sync_loss   = sum(sync_losses) / len(sync_losses)
            averaged_sync_loss_a = sum(sync_losses_a) / len(sync_losses_a)
            averaged_sync_loss_p = sum(sync_losses_p) / len(sync_losses_p)
            averaged_sync_loss_e = sum(sync_losses_e) / len(sync_losses_e)
            averaged_blink_loss  = sum(blink_losses) / len(blink_losses)

        logger.info(
            "L1: {:.4f}, Sync_audio: {:.4f}, Sync_pose: {:.4f}, Sync_emotion: {:.4f}, Blink: {:.4f}, VGG: {:.4f}".format(
                averaged_l1_loss,
                averaged_sync_loss_a,
                averaged_sync_loss_p,
                averaged_sync_loss_e,
                averaged_blink_loss,
                averaged_vgg_loss,
            )
        )

        return (
            averaged_l1_loss,
            averaged_sync_loss,
            averaged_sync_loss_a,
            averaged_sync_loss_p,
            averaged_sync_loss_e,
            averaged_blink_loss,
            averaged_vgg_loss,
        )


def save_checkpoint(generator, optimizer, save_optimizer_state, step, checkpoint_dir, epoch, prefix="", is_best=False):
    if is_best:
        checkpoint_path = os.path.join(checkpoint_dir, "{}best_checkpoint_epoch{:05d}_step{:09d}.pth".format(prefix, epoch, step))
    else:
        checkpoint_path = os.path.join(checkpoint_dir, "{}checkpoint_epoch{:05d}_step{:09d}.pth".format(prefix, epoch, step))

    optimizer_state = optimizer.state_dict() if save_optimizer_state else None
    torch.save(
        {
            "state_dict": generator.state_dict(),
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


def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    logger.info("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace("module.", "")] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            logger.info("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])

    if overwrite_global_states:
        global_step  = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model


if __name__ == "__main__":
    from models.syncnet_pose_emotion import SyncNet_pose_emotion as SyncNet
    from models.gen_pose_emotion_blink import Generator_pose_emotion_blink as Generator
    from models.networks import VGGLoss
    from hparams import hparams

    logger = create_logger("train_pose_emotion_blink", "logs/train_pose_emotion_blink.log")

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_root", help="Root folder of the preprocessed dataset", required=True, type=str)
    parser.add_argument("--checkpoint_dir", help="Save checkpoints to this directory", required=True, type=str)
    parser.add_argument("--syncnet_checkpoint_path", help="Load the pre-trained Syncnet", required=True, type=str)
    parser.add_argument("--checkpoint_path", help="Resume generator from this checkpoint", default=None, type=str)
    parser.add_argument("--audio_wt", help="audio_loss", required=True, type=float)
    parser.add_argument("--vgg_wt", help="vgg_loss", required=True, type=float)
    parser.add_argument("--emotion_wt", help="emotion_loss", required=True, type=float)
    parser.add_argument("--pose_wt", help="pose_loss", required=True, type=float)
    args = parser.parse_args()

    data_root               = args.data_root
    checkpoint_dir          = args.checkpoint_dir
    syncnet_checkpoint_path = args.syncnet_checkpoint_path
    checkpoint_path         = args.checkpoint_path
    audio_wt                = args.audio_wt
    vgg_wt                  = args.vgg_wt
    pose_wt                 = args.pose_wt
    emotion_wt              = args.emotion_wt


    # Dataset and Dataloader setup
    train_dataset = Dataset(hparams, data_root, checkpoint_dir, "train")
    test_dataset  = Dataset(hparams, data_root, checkpoint_dir, "val")
    valid_list    = create_image_lists(data_root, checkpoint_dir, hparams.generator_batch_size)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=hparams.generator_batch_size,
        shuffle=True,
        num_workers=hparams.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    test_data_loader = DataLoader(
        test_dataset, batch_size=hparams.generator_batch_size, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Model
    generator = Generator()
    vgg_model = VGGLoss()
    for p in vgg_model.parameters():
        p.requires_grad = False
    syncnet = SyncNet()
    for p in syncnet.parameters():
        p.requires_grad = False

    print("total trainable params {}".format(sum(p.numel() for p in generator.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in generator.parameters() if p.requires_grad], lr=hparams.generator_lr, betas=(0.5, 0.999))

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, generator, optimizer, reset_optimizer=False)

    if not os.path.exists(syncnet_checkpoint_path):
        raise FileNotFoundError("Syncnet checkpoint not found")
    else:
        load_checkpoint(syncnet_checkpoint_path, syncnet, None, reset_optimizer=True, overwrite_global_states=False)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    if torch.cuda.device_count() > 1:
        generator = nn.DataParallel(generator).to(device)
        vgg_model = nn.DataParallel(vgg_model).to(device)
        syncnet   = nn.DataParallel(syncnet).to(device)
    else:
        generator = generator.to(device)
        vgg_model = vgg_model.to(device)
        syncnet   = syncnet.to(device)

    # Train!
    print("start training")
    train(
        device,
        data_root,
        generator,
        syncnet,
        vgg_model,
        train_data_loader,
        test_data_loader,
        optimizer,
        audio_wt,
        pose_wt,
        emotion_wt,
        vgg_wt,
        hparams,
        checkpoint_dir=checkpoint_dir,
    )
