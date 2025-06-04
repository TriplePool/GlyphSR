import math
from turtle import forward
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import sys
from torch.nn import init
import numpy as np

sys.path.append('./')
sys.path.append('../')
from model.tps_spatial_transformer import TPSSpatialTransformer
from model.stn_head import STNHead
from model.dcn import DSTA

from einops import rearrange

SHUT_BN = False

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class UpsampleBLockOld(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLockOld, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        # self.prelu = nn.ReLU()
        self.prelu = mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class RecurrentResidualBlockTL(nn.Module):
    def __init__(self, channels, text_channels):
        super(RecurrentResidualBlockTL, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = GruBlock(channels + text_channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = GruBlock(channels, channels)

        # self.concat_conv = nn.Conv2d(channels + text_channels, channels, kernel_size=3, padding=1)

    def forward(self, x, text_emb):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        ############ Fusing with TL ############
        cat_feature = torch.cat([residual, text_emb], 1)
        # residual = self.concat_conv(cat_feature)
        ########################################

        residual = self.gru1(cat_feature.transpose(-1, -2)).transpose(-1, -2)
        # residual = self.non_local(residual)

        return self.gru2(x + residual)

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size], T = num_steps.
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(
            input
        )  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output

class CharSegHead(nn.Module):
    def __init__(self, input_channels, max_seq_len, num_classes) -> None:
        super().__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(input_channels, 2 * input_channels, 3, 1, 1),
            nn.BatchNorm2d(2 * input_channels),
            nn.ReLU()
        )
        self.pool0 = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * input_channels, 2 * 2 * input_channels, 3, 1, 1),
            nn.BatchNorm2d(2 * 2 * input_channels),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * 2 * input_channels, 2 * 2 * input_channels, 3, 1, 1),
            nn.BatchNorm2d(2 * 2 * input_channels),
            nn.ReLU()
        )

        self.gru = GruBlock(2 * 2 * input_channels, 2 * 2 * input_channels)

        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * 2 * input_channels, 2 * 2 * input_channels, 3, 1, 1),
            nn.BatchNorm2d(2 * 2 * input_channels),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(2 * 2 * input_channels, 2 * input_channels, 3, 1, 1),
            nn.BatchNorm2d(2 * input_channels),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(2 * input_channels, input_channels, 3, 1, 1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU()
        )

        self.rec_proj = nn.Sequential(
            nn.Conv2d(input_channels, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.char_heed = nn.Conv2d(input_channels, max_seq_len, 3, 1, 1)
        self.seg_heed = nn.Conv2d(input_channels, 1, 3, 1, 1)
        self.char_cls = nn.Linear(256, num_classes)
        
    
    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def _apply_attention(self, x, attn):
        feat = rearrange(x, 'b c h w -> b (h w) c')
        attn = rearrange(attn, 'b l h w -> b l (h w)').softmax(dim=-1)
        # print(feat.shape, attn.shape)
        res = torch.bmm(attn, feat) # b l c
        return res
    
    def forward(self, x):
        x_raw = x
        x = self.conv0(x)
        r0 = x
        x = self.pool0(x)
        x = self.conv1(x)
        r1 = x
        x = self.conv2(x)
        x = self.gru(x)
        x = self.conv3(x)
        x = self._upsample_add(x, r1)
        x = self.conv4(x)
        x = self._upsample_add(x, r0)
        x = self.conv5(x)
        char_pred = self.char_heed(x)
        seg_pred = self.seg_heed(x)

        char_attn = char_pred * seg_pred
        x_rec = self.rec_proj(x_raw)
        char_attn_feat = self._apply_attention(x_rec, char_attn)
        char_cls = self.char_cls(char_attn_feat)

        return seg_pred, char_pred, char_cls, char_attn_feat

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class MergeBlock(nn.Module):
    def __init__(self, attn_channels, hidden_channels) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(hidden_channels, attn_channels, 1, 1, bias=False),nn.BatchNorm2d(attn_channels))
        self.conv2 = nn.Sequential(nn.Conv2d(attn_channels, attn_channels, 3, 1, 1, bias=True),nn.BatchNorm2d(attn_channels))

        self.channel_wise_attn = ChannelAttention(attn_channels, 1)
        self.conv3 = nn.Sequential(nn.Conv2d(attn_channels, hidden_channels, 1, 1, bias=False),nn.BatchNorm2d(hidden_channels))
        self.dsta = DSTA(hidden_channels)
    def forward(self, x, attn_scores):
        residual = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        attn = self.channel_wise_attn(attn_scores)
        x = x * attn
        x = self.conv3(x)
        x = self.dsta(x)
        x = x + residual
        x = F.relu(x)
        return x

class GlyphSR(nn.Module):
    def __init__(self,
                 scale_factor=2,
                 width=128,
                 height=32,
                 STN=False,
                 srb_nums=5,
                 mask=True,
                 hidden_units=32,
                 word_vec_d=300,
                 text_emb=37, #37, #26+26+1 3965
                 out_text_channels=32,
                 triple_clues=False,
                 char_block=6): #256 32
        super(GlyphSR, self).__init__()
        
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        self.text_emb = text_emb
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.srb_nums = srb_nums
        self.char_block = char_block
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlockTL(2 * hidden_units, out_text_channels)) #RecurrentResidualBlockTL

        self.feature_enhancer = None
        # From [1, 1] -> [16, 16]

        self.infoGen = InfoGen(text_emb, out_text_channels)

        if not SHUT_BN:
            setattr(self, 'block%d' % (srb_nums + 2),
                    nn.Sequential(
                        nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                        nn.BatchNorm2d(2 * hidden_units)
                    ))
        else:
            setattr(self, 'block%d' % (srb_nums + 2),
                    nn.Sequential(
                        nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                        # nn.BatchNorm2d(2 * hidden_units)
                    ))

        block_ = [UpsampleBLockOld(2 * hidden_units, 2) for _ in range(upsample_block_num)]
        setattr(self, 'block_up', nn.Sequential(*block_))
        block_ = []
        block_.append(nn.Conv2d(2 * hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block_out', nn.Sequential(*block_))
        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height // scale_factor, width // scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none',
                input_size=self.tps_inputsize)

        self.block_range = [k for k in range(2, self.srb_nums+2)]
        assert self.char_block in self.block_range

        self.dsta_rec = DSTA(hidden_units)
        for idx in range(self.char_block, self.block_range[-1] + 1):
            setattr(self, 'block_merge%d' % idx, MergeBlock(16, 2 * hidden_units))

        self.char_seg = CharSegHead(2 * hidden_units, 16, 36 + 3)

    def forward(self, x, text_emb=None):

        block = {'1': self.block1(x)}

        if self.stn and self.training:
            # x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)

        if text_emb is None:
            text_emb = torch.zeros(1, self.text_emb, 1, 26).to(x.device) # 37 or 3965

        spatial_t_emb_, pr_weights = self.infoGen(text_emb)  # # ,block['1'], block['1'],

        spatial_t_emb = F.interpolate(spatial_t_emb_, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)

        hint_rec = self.dsta_rec(spatial_t_emb)

        hint = hint_rec

        seg_pred, char_pred, char_cls = None, None, None

        # Reasoning block: [2, 3, 4, 5, 6]
        for i in range(self.srb_nums + 1):
            if i + 2 in self.block_range:
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)], hint)
                if i + 2 == self.char_block:
                    seg_pred, char_pred, char_cls, char_attn_feat = self.char_seg(block[str(i + 2)])
                if i + 2 >= self.char_block:
                    block[str(i + 2)] = getattr(self, 'block_merge%d' % (i + 2))(block[str(i + 2)], char_pred)
            else:
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        self.block = block

        seg_res = (seg_pred, char_pred, char_cls)
        output = getattr(self, 'block_up')((block['1'] + block[str(self.srb_nums + 2)]))
        output = getattr(self, 'block_out')(output)
        output = torch.tanh(output)
        return output, seg_res

class InfoGen(nn.Module):
    def __init__(
                self,
                t_emb,
                output_size
                 ):
        super(InfoGen, self).__init__()

        self.tconv1 = nn.ConvTranspose2d(t_emb, 512, 3, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        self.tconv2 = nn.ConvTranspose2d(512, 128, 3, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.tconv3 = nn.ConvTranspose2d(128, 64, 3, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.tconv4 = nn.ConvTranspose2d(64, output_size, 3, (2, 1), padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(output_size)

    def forward(self, t_embedding):

        # t_embedding += noise.to(t_embedding.device)

        x = F.relu(self.bn1(self.tconv1(t_embedding)))
        # print(x.shape)
        x = F.relu(self.bn2(self.tconv2(x)))
        # print(x.shape)
        x = F.relu(self.bn3(self.tconv3(x)))
        # print(x.shape)
        x = F.relu(self.bn4(self.tconv4(x)))
        # print(x.shape)

        return x, torch.zeros((x.shape[0], 1024, t_embedding.shape[-1])).to(x.device)

class mish(nn.Module):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (torch.tanh(F.softplus(x)))
        return x

class GruBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GruBlock, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.gru = nn.GRU(out_channels, out_channels // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        # x = self.gru(x)[0]
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2)
        return x

