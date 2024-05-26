# Free-View_Expressive_Talking_Head_Video_Editing
Code for the paper "Free-View Expressive Talking Head Video Editing" (ICASSP 2023)

Project Page: [https://sky24h.github.io/websites/icassp2023_free-view_video-editing](https://sky24h.github.io/websites/icassp2023_free-view_video-editing)

Huggingface Demo: [https://huggingface.co/spaces/sky24h/Free-View_Expressive_Talking_Head_Video_Editing](https://huggingface.co/spaces/sky24h/Free-View_Expressive_Talking_Head_Video_Editing)


# Dependencies
Python >= 3.9
```bash
pip install -r requirements.txt
```


# Data Preparation
Due to licensing issues, we cannot provide the full dataset.
Instead, the URLs of videos and preprocessing scripts will be provided soon.



# Inference
An example is provided in inference.sh:

```bash
bash inference.sh
```
Pretrained models will be automatically downloaded when running the code.

Please check inference.py for more details.

# Training
Our training includes two stages: 1) training the Multi-Attribute Discriminator for syncing the audio, attributes, and video, and 2) training the Generator for generating the talking head video.


### Stage 1. Multi-Attribute Discriminator
```bash
python train_sync.sh
```

### Stage 2. Generator Model
```bash
python train_gen.sh
```

Please check corresponding scripts for more details.

# Citation
If you find this code useful, please cite our paper:

```
@inproceedings{Huang2023FETE,
  author = {Huang, Yuantian and Iizuka, Satoshi and Fukui, Kazuhiro},
  booktitle = {ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title = {Free-View Expressive Talking Head Video Editing},
  year = {2023},
  pages = {1-5},
  doi = {10.1109/ICASSP49357.2023.10095745},
}
```


