import os
import argparse
from inference_utils import inference, prepare_data
from utils.attributes_utils import showcase, input_pose, input_emotion, input_blink
from utils.audio_utils import load_wav

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference code")

    parser.add_argument("--checkpoint_path", type=str, help="Name of saved checkpoint to load weights from", default="./checkpoints/obama-fp16.safetensors")
    parser.add_argument("--face", type=str, help="Filepath of video/image that contains faces to use", required=True)
    parser.add_argument("--audio", type=str, help="Filepath of video/audio file to use as raw audio source", required=True)
    parser.add_argument("--pose", type=str, help="pose", default=None)
    parser.add_argument("--blink", type=str, help="pose", default=None)
    parser.add_argument("--emotion", type=str, help="emotion", default=None)
    parser.add_argument("--outfile", type=str, help="Video path to save result. See default for an e.g.", default="results/res.mp4")
    parser.add_argument("--image_size", type=int, help="Size of image to use for inference", default=256)
    parser.add_argument("--showcase", action="store_true", help="Add 20 seconds of silence, to showcase the attributes editing")
    args = parser.parse_args()
    audio_path = args.audio
    face_path = args.face
    checkpoint_path = args.checkpoint_path

    if os.path.exists(checkpoint_path):
        print("Checkpoint found")
    else:
        print("Checkpoint not found, try downloading ...")
        from huggingface_hub import hf_hub_download
        repo_id = "sky24h/Free-View_Expressive_Talking_Head_Video_Editing"
        subfolder = "checkpoints"
        filename = "obama-fp16.safetensors"
        hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder, local_dir=".", local_dir_use_symlinks=False, repo_type="space")
    sample_rate = 16000

    pads = [0, 0, 0, 0]

    if audio_path.endswith(".wav") or audio_path.endswith(".mp3"):
        wav = load_wav(audio_path, sample_rate)
    else:
        # extract audio from video
        import moviepy.editor as mp
        video = mp.VideoFileClip(audio_path)
        audio = video.audio
        audio.write_audiofile(audio_path.replace(".mp4", ".wav"), fps=sample_rate)
        audio_path = audio_path.replace(".mp4", ".wav")
        wav = load_wav(audio_path, sample_rate)

    if args.showcase:
        print("Running in showcase mode, ingoring provided attributes")
        pose, emotion, blink, subtitle, face_path, audio_path = showcase(face_path, audio_path, wav)
    else:
        # you can modify the following functions to input your own attributes.
        # pose
        pose = input_pose()
        # emotion
        emotion = input_emotion()
        # blink
        blink = input_blink()
        subtitle = None

    gen, steps = prepare_data(face_path, audio_path, pose, emotion, blink, img_size=args.image_size, pads=pads)
    inference(args.checkpoint_path, gen, steps, audio_path, args.outfile, subtitle)
