import os
import glob
import random
import shutil
import pickle
from tqdm import tqdm
from natsort import natsorted


def create_file_lists(data_root, filename, videos_list):
    images_lists = []
    for vidname in tqdm(videos_list):
        images_list = natsorted(glob.glob(os.path.join(data_root, vidname, "*.png")) + glob.glob(os.path.join(data_root, vidname, "*.jpg")))
        images_lists.append(list(images_list))
    with open(filename, "wb") as fp:
        pickle.dump(images_lists, fp)


def create_image_lists(data_root, checkpoint_dir, batch_size, overwrite=False):
    if len(os.path.basename(data_root)) > 0:
        pass
    else:
        # ../a/b/ -> ../a/b
        data_root = data_root[:-1]
    if len(os.path.basename(checkpoint_dir)) > 0:
        pass
    else:
        # ../a/b/ -> ../a/b
        checkpoint_dir = checkpoint_dir[:-1]

    filelist_save_dir = os.path.join(".filelists", os.path.basename(data_root))

    if not os.path.exists(filelist_save_dir) or overwrite:
        os.makedirs(filelist_save_dir)
        all_videos = os.listdir(data_root)

        remove_list = []
        for vid in all_videos:
            if len(glob.glob(os.path.join(data_root, vid, "*.png"))) + len(glob.glob(os.path.join(data_root, vid, "*.jpg"))) == 0:
                remove_list.append(vid)
            elif not os.path.exists(os.path.join(data_root, vid, "mel.npy")):
                remove_list.append(vid)
            elif not os.path.exists(os.path.join(data_root, vid, "pose.npy")):
                remove_list.append(vid)
            elif not os.path.exists(os.path.join(data_root, vid, "blink.npy")):
                remove_list.append(vid)
            elif not os.path.exists(os.path.join(data_root, vid, "emotion_img.npy")):
                remove_list.append(vid)

        print("total num for videos: {}".format(len(all_videos)))
        for vid in remove_list:
            all_videos.remove(vid)
        print("total num for videos after removing empty videos: {}".format(len(all_videos)))

        while len(all_videos) < 8000 and len(all_videos) > 0:
            all_videos += all_videos
        all_videos = all_videos[:8000]
        random.shuffle(all_videos)

        print(filelist_save_dir)
        if not os.path.exists(filelist_save_dir):
            os.makedirs(filelist_save_dir, exist_ok=True)

        len_ = len(all_videos)
        for_traing_num = (int(0.90 * len_) // batch_size) * batch_size - 1
        for_test_num = (int(0.95 * len_) // batch_size) * batch_size - 1
        create_file_lists(data_root, os.path.join(filelist_save_dir, "train"), all_videos[:for_traing_num])
        create_file_lists(data_root, os.path.join(filelist_save_dir, "test"), all_videos[for_traing_num:for_test_num])
        create_file_lists(data_root, os.path.join(filelist_save_dir, "val"), all_videos[for_test_num:])
        print("filelist created", filelist_save_dir)
    else:
        print("filelist already exists", filelist_save_dir)


def get_image_lists(data_root, split):
    if len(os.path.basename(data_root)) > 0:
        pass
    else:
        data_root = data_root[:-1]

    filelist_path = os.path.join(".filelists", os.path.basename(data_root), split)
    with open(filelist_path, "rb") as fp:
        images_lists = pickle.load(fp)
    video_lists = [os.path.dirname(x[0]) for x in images_lists]

    print("filelist loaded", filelist_path)
    print("total num for {}: {}".format(split, len(images_lists)))

    return images_lists, video_lists

