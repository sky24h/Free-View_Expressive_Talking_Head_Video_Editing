import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv2d

class SyncNet_pose_emotion(nn.Module):
    def __init__(self):
        super(SyncNet_pose_emotion, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=2, padding=3),#96

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.pose_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(1, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.emotion_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=1, stride=2, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)


    def forward(self, audio_sequences, face_sequences, pose_sequences, emotion_sequences): # audio_sequences := (B, dim, T)
        # print('audio_sequences.shape:', audio_sequences.shape, 'face_sequences.shape:', face_sequences.shape, 'pose_sequences.shape:', pose_sequences.shape)

        face_embedding    = self.face_encoder(face_sequences)
        audio_embedding   = self.audio_encoder(audio_sequences)
        pose_embedding    = self.pose_encoder(pose_sequences)
        emotion_embedding = self.emotion_encoder(emotion_sequences)
        # print('face_embedding', face_embedding.shape, 'audio_embedding', audio_embedding.shape, 'pose_embedding', pose_embedding.shape)

        audio_embedding   = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding    = face_embedding.view(face_embedding.size(0), -1)
        pose_embedding    = pose_embedding.view(pose_embedding.size(0), -1)
        emotion_embedding = emotion_embedding.view(emotion_embedding.size(0), -1)
        # print('face_embedding', face_embedding.shape, 'audio_embedding', audio_embedding.shape, 'pose_embedding', pose_embedding.shape)

        audio_embedding   = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding    = F.normalize(face_embedding, p=2, dim=1)
        pose_embedding    = F.normalize(pose_embedding, p=2, dim=1)
        emotion_embedding = F.normalize(emotion_embedding, p=2, dim=1)
        # print('face_embedding', face_embedding.shape, 'audio_embedding', audio_embedding.shape, 'pose_embedding', pose_embedding.shape)

        return audio_embedding, face_embedding, pose_embedding, emotion_embedding
