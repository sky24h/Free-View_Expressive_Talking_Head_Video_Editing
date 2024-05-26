import torch
from torch import nn

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d


class Generator_pose_emotion_blink(nn.Module):
    def __init__(self):
        super(Generator_pose_emotion_blink, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=2, padding=3)), # 256,256 -> 128,128

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 64,64
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # 32,32
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # 16,16
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),       # 8,8
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),     # 4,4
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=2, padding=0),     # 1, 1
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),])

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
            Conv2d(1, 32, kernel_size=7, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(1, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.blink_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=(1, 2), padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(1, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=1, stride=(1, 2), padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(2048, 512, kernel_size=1, stride=1, padding=0),),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=4, stride=1, padding=0), # 4,4
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), # 8,8
            Self_Attention(512, 512),),

            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True), # 16, 16
            Self_Attention(384, 384),),

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 32, 32
            Self_Attention(256, 256),),

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),), # 64, 64

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),),]) # 128,128

        # self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
        #     nn.Sigmoid())

        self.output_block = nn.Sequential(Conv2dTranspose(80, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())

    def forward(self, face_sequences, audio_sequences, pose_sequences, emotion_sequences, blink_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences   = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            pose_sequences    = torch.cat([pose_sequences[:, i] for i in range(pose_sequences.size(1))], dim=0)
            emotion_sequences = torch.cat([emotion_sequences[:, i] for i in range(emotion_sequences.size(1))], dim=0)
            blink_sequences   = torch.cat([blink_sequences[:, i] for i in range(blink_sequences.size(1))], dim=0)
            face_sequences    = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        # print(audio_sequences.size(), face_sequences.size(), pose_sequences.size(), emotion_sequences.size())

        audio_embedding   = self.audio_encoder(audio_sequences) # B,                                                     512,  1, 1
        pose_embedding    = self.pose_encoder(pose_sequences) # B,                                                       512,  1, 1
        emotion_embedding = self.emotion_encoder(emotion_sequences) # B,                                                 512,  1, 1
        blink_embedding   = self.blink_encoder(blink_sequences) # B,                                                     512,  1, 1
        inputs_embedding  = torch.cat((audio_embedding, pose_embedding, emotion_embedding, blink_embedding), dim=1) # B, 1536, 1, 1
        # print(audio_embedding.size(), pose_embedding.size(), emotion_embedding.size(), inputs_embedding.size())

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            # print(x.shape)
            feats.append(x)

        x = inputs_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            # print(x.shape)

            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e
            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
            outputs = x

        return outputs


class Self_Attention(nn.Module):
    """
    Source-Reference Attention Layer
    """

    def __init__(self, in_planes_s, in_planes_r):
        """
        Parameters
        ----------
            in_planes_s: int
                Number of input source feature vector channels.
            in_planes_r: int
                Number of input reference feature vector channels.
        """
        super(Self_Attention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_planes_s, out_channels=in_planes_s // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_planes_r, out_channels=in_planes_r // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_planes_r, out_channels=in_planes_r, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, source):
        source = source.float() if isinstance(source, torch.cuda.HalfTensor) else source
        reference = source
        """
        Parameters
        ----------
            source : torch.Tensor
                Source feature maps (B x Cs x Ts x Hs x Ws)
            reference : torch.Tensor
                Reference feature maps (B x Cr x Tr x Hr x Wr )
         Returns :
            torch.Tensor
                Source-reference attention value added to the input source features
            torch.Tensor
                Attention map (B x Ns x Nt) (Ns=Ts*Hs*Ws, Nr=Tr*Hr*Wr)
        """
        s_batchsize, sC, sH, sW = source.size()
        r_batchsize, rC, rH, rW = reference.size()

        proj_query = self.query_conv(source).view(s_batchsize, -1, sH * sW).permute(0, 2, 1)
        proj_key = self.key_conv(reference).view(r_batchsize, -1, rW * rH)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(reference).view(r_batchsize, -1, rH * rW)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(s_batchsize, sC, sH, sW)
        out = self.gamma * out + source
        return out.half() if isinstance(source, torch.cuda.FloatTensor) else out


