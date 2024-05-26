import os
import cv2
import sys
import librosa
import imageio
import numpy as np
import soundfile as sf


sample_rate = 16000
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def input_pose(pose_select="front"):
    step = 1
    scale = 10

    if pose_select == "front":
        pose = [[0.0, 0.0, 0.0] for i in range(0, 10, step)]  # -20 to 20
    elif pose_select == "left_right_shaking":
        pose = [[-i, 0.0, 0.0] for i in range(0, scale*2, step)]  # 0 to -20
        pose += [[i - scale*2, 0.0, 0.0] for i in range(0, scale*4, step)]  # -20 to 20
        pose += [[scale*2 - i, 0.0, 0.0] for i in range(0, scale*2, step)]  # 20 to 0
        pose = pose + pose
        pose = pose + pose
        pose = pose + pose
    else:
        raise ValueError("pose_select Error")

    return np.array(pose)


def input_emotion(emotion_select="neutral"):
    sacle_factor = 1.4
    if emotion_select == "neutral":
        emotion = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for _ in range(2)]
    elif emotion_select == "happy":
        emotion = [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] for _ in range(2)]
    elif emotion_select == "angry":
        emotion = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(2)]
    elif emotion_select == "surprised":
        emotion = [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] for _ in range(2)]
    else:
        raise ValueError("emotion_select Error")

    return np.array(emotion) * sacle_factor


def input_blink(blink_select="yes"):
    if blink_select == "yes":
        blink = [[1.0] for _ in range(25)]
        blink += [[0.8], [0.6], [0.0], [0.0]]
        blink += [[1.0] for _ in range(5)]
        blink = blink + blink + blink
    else:
        blink = [[1.0] for _ in range(2)]
    return np.array(blink)


def showcase(face_path, audio_path, wav):
    audio_len = round(librosa.get_duration(y=wav, sr=sample_rate))

    # this function is used to generate the same results as our demo video.
    step = 1
    scale = 10

    yaw = (
        [[-i, 0.0, 0.0] for i in range(0, scale * 2, step)]
        + [[i - scale * 2, 0.0, 0.0] for i in range(0, scale * 4, step)]
        + [[scale * 2 - i, 0.0, 0.0] for i in range(0, scale * 2, step)]
    )
    # divide every value by 2
    half_yaw = [[i[0] / 2.0, i[1] / 2.0, i[2] / 2.0] for i in yaw]

    pitch = (
        [[0.0, -i, 0.0] for i in range(0, scale, step)]
        + [[0.0, i - scale, 0.0] for i in range(0, scale * 2, step)]
        + [[0.0, scale - i, 0.0] for i in range(0, scale, step)]
    )
    # divide every value by 2
    half_pitch = [[i[0] / 2.0, i[1] / 2.0, i[2] / 2.0] for i in pitch]

    roll = (
        [[0.0, 0.0, (-i) * 1] for i in range(0, scale, step)]
        + [[0.0, 0.0, (i - scale) * 1] for i in range(0, scale * 2, step)]
        + [[0.0, 0.0, (scale - i) * 1] for i in range(0, scale, step)]
    )  # 20 to 0
    # divide every value by 2
    half_roll = [[i[0] / 2.0, i[1] / 2.0, i[2] / 2.0] for i in roll]
    pose = []
    pose += yaw + pitch + roll
    len_per_5sec = len(pose)

    pose += [[0.0, 0.0, 0.0] for i in range(0, len_per_5sec * 3)]  # 15 sec
    pose += half_yaw + half_pitch + half_roll
    if audio_len > 5:
        pose += [[0.0, 0.0, 0.0] for i in range(0, int((audio_len - 5) / 5) * len_per_5sec)]
    pose = np.array(pose)
    # print("pose", pose.shape, pose[:10])

    quick_blink = [[1.0], [0.8], [0.4], [0.0], [1.0]]
    slow_blink = [[1.0], [1.0], [0.8], [0.6], [0.4], [0.2], [0.2], [0.0], [0.8], [1.0]]
    len_per_sec = len(slow_blink)
    blink = [[1.0] for i in range(0, len_per_sec * 5)]
    # blink = [[0.05*i] for i in range(0, 20, step)]#-20 to 20
    blink += slow_blink
    blink += slow_blink
    blink += [[1.0] for i in range(0, len_per_sec * 1)]
    blink += quick_blink + quick_blink
    blink += [[1.0] for i in range(0, len_per_sec * 11)]

    blink += [[1.0] for i in range(0, len_per_sec * 1)]
    blink += quick_blink
    blink += [[1.0] for i in range(0, len_per_sec * 1)]
    blink += [[1.0] for i in range(0, len_per_sec * 1)]
    blink += quick_blink
    blink += [[1.0] for i in range(0, len_per_sec * 1)]
    blink += quick_blink
    blink += quick_blink
    blink += [[1.0] for i in range(0, int(len_per_sec * (audio_len - 6)))]
    blink = np.array(blink)
    # print("blink", blink.shape, blink[:10])

    len_per_sec = 5
    normal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    happy = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    angry = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    surprise = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]

    # normal to surprise, interpolate in 5 elements
    n2s = []
    for i in range(5):
        n2s.append(np.array(normal) * (4 - i) / 4 + np.array(surprise) * (i) / 4)
    s2n = n2s[::-1]
    # print(n2s)

    n2h = []
    for i in range(5):
        n2h.append(np.array(normal) * (4 - i) / 4 + np.array(happy) * (i) / 4)
    h2n = n2h[::-1]

    n2a = []
    for i in range(5):
        n2a.append(np.array(normal) * (4 - i) / 4 + np.array(angry) * (i) / 4)
    a2n = n2a[::-1]

    emotion = [normal for _ in range(len_per_sec * 10)]
    emotion += n2s + [surprise for _ in range(len_per_sec * 1)] + s2n
    emotion += n2h + [happy for _ in range(len_per_sec * 1)] + h2n
    emotion += n2a + [angry for _ in range(len_per_sec * 1)] + a2n
    emotion += [normal for _ in range(len_per_sec * (audio_len - 3))]
    emotion += n2a + [angry for _ in range(len_per_sec * 2)]

    emotion = np.array(emotion) * 1.4  # amplify the emotion, can be adjusted

    subtitle = []
    # first 5 sec
    subtitle += ["Head Pose (Yaw, Pitch, Roll)"] * 5 * 25
    subtitle += ["Eye Blink (Slow, Quick)"] * 5 * 25
    subtitle += ["Facial Emotion (Surprise, Happy, Angry)"] * 10 * 25
    subtitle += ["Editing in Real Video"] * int((audio_len) * 25)  # add 10 sec at the end to avoid error

    # add 20 seconds of silence at the beginning
    wav = np.concatenate((np.zeros((sample_rate * 20)), wav))
    sf.write(audio_path.replace(".wav", "_showcase.wav"), wav, sample_rate)

    # add 20 seconds static at the beginning
    frames = []
    cap = cv2.VideoCapture(face_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    first_frame = frames[0]
    frames = [first_frame for _ in range(25 * 20)] + frames
    writer = imageio.get_writer(face_path.replace(".mp4", "_showcase.mp4"), fps=25, quality=8, macro_block_size=1)
    for frame in frames:
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    writer.close()

    # print("emotion", emotion.shape, emotion[:10])
    return pose, emotion, blink, subtitle, face_path.replace(".mp4", "_showcase.mp4"), audio_path.replace(".wav", "_showcase.wav")