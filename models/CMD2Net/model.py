import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import SwinV2Small
from .channel_attention import ChannelAttention
from .multi_scale_feature_fusion import MultiScaleFeatureFusion
from .dilated_temporal_frequency_fusion import DilatedTemporalFrequencyFusion
from .decoder import Decoder

class CMD2Net(nn.Module):
    def __init__(self, mid_c=128):
        super(CMD2Net, self).__init__()
        self.mid_c = mid_c

        self.backbone = SwinV2Small(pretrained=True)
        self.channels = self.backbone.out_channels
        
        self.channel_attention = ChannelAttention(self.channels)
        self.multiscale_feature_fusion = MultiScaleFeatureFusion(self.channels, self.mid_c)
        self.temporal_frequency_refinement = DilatedTemporalFrequencyFusion(self.mid_c, self.mid_c)
        self.decoder = Decoder(self.mid_c)

    def forward(self, x1, x2):
        # Backbone
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone(x2)
        # Channel Attention
        x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5 = self.channel_attention(
            x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5
        )
        # Multi Scale Feature Fusion
        x1_2, x1_3, x1_4, x1_5 = self.multiscale_feature_fusion(x1_2, x1_3, x1_4, x1_5)
        x2_2, x2_3, x2_4, x2_5 = self.multiscale_feature_fusion(x2_2, x2_3, x2_4, x2_5)
        # Dilated Temporal Frequency Fusion
        c2, c3, c4, c5 = self.temporal_frequency_refinement(
            x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5
        )
        # Decoder
        out_p2, out_p3, out_p4, out_p5 = self.decoder(c2, c3, c4, c5)

        out_p2 = torch.sigmoid(nn.functional.interpolate(out_p2, scale_factor=(4, 4), mode='bilinear'))
        out_p3 = torch.sigmoid(nn.functional.interpolate(out_p3, scale_factor=(8, 8), mode='bilinear'))
        out_p4 = torch.sigmoid(nn.functional.interpolate(out_p4, scale_factor=(16, 16), mode='bilinear'))
        out_p5 = torch.sigmoid(nn.functional.interpolate(out_p5, scale_factor=(32, 32), mode='bilinear'))

        return {"main_predictions": out_p2,"aux_predictions": [out_p3, out_p4, out_p5]}
