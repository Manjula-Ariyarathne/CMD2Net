import torch
import torch.nn as nn

class DilatedTemporalDifferenceFusion(nn.Module):
    def __init__(self, in_c, out_c):
        super(DilatedTemporalDifferenceFusion, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.relu = nn.ReLU(inplace=True)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(self.in_c, self.in_c, kernel_size=3, stride=1, padding=9, dilation=9),
            nn.BatchNorm2d(self.in_c)
        )
        self.conv_2 = nn.Conv2d(self.in_c, self.in_c, kernel_size=1)
        self.conv_2_f = nn.Sequential(
            nn.Conv2d(self.in_c, self.in_c, kernel_size=3, stride=1, padding=7, dilation=7),
            nn.BatchNorm2d(self.in_c)
        )
        self.conv_3 = nn.Conv2d(self.in_c, self.in_c, kernel_size=1)
        self.conv_3_f = nn.Sequential(
            nn.Conv2d(self.in_c, self.in_c, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(self.in_c)
        )
        self.conv_4 = nn.Conv2d(self.in_c, self.in_c, kernel_size=1)
        self.conv_4_f = nn.Sequential(
            nn.Conv2d(self.in_c, self.in_c, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(self.in_c)
        )
        self.conv_5 = nn.Conv2d(self.in_c, self.in_c, kernel_size=1)
        self.conv_5_f = nn.Sequential(
            nn.Conv2d(self.in_c, self.out_c, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.out_c)
        )
        self.conv_6 = nn.Conv2d(self.in_c, self.out_c, kernel_size=1)

    def forward(self, x1, x2):
        x = torch.abs(x1 - x2)
        x_1 = self.conv_1(x)
        x_2 = self.relu(self.conv_2(x) + x_1)
        x_2 = self.conv_2_f(x_2)
        x_3 = self.relu(self.conv_3(x) + x_2)
        x_3 = self.conv_3_f(x_3)
        x_4 = self.relu(self.conv_4(x) + x_3)
        x_4 = self.conv_4_f(x_4)
        x_5 = self.relu(self.conv_5(x) + x_4)
        x_5 = self.conv_5_f(x_5)
        x_out = self.relu(self.conv_6(x) + x_5)

        return x_out


class TemporalFeatureFusion(nn.Module):
    def __init__(self, in_c=128, out_c=128):
        super(TemporalFeatureFusion, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.temporal_fusion_x2 = DilatedTemporalDifferenceFusion(self.in_c, self.out_c)
        self.temporal_fusion_x3 = DilatedTemporalDifferenceFusion(self.in_c, self.out_c)
        self.temporal_fusion_x4 = DilatedTemporalDifferenceFusion(self.in_c, self.out_c)
        self.temporal_fusion_x5 = DilatedTemporalDifferenceFusion(self.in_c, self.out_c)

    def forward(self, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5):
        c2 = self.temporal_fusion_x2(x1_2, x2_2)
        c3 = self.temporal_fusion_x3(x1_3, x2_3)
        c4 = self.temporal_fusion_x4(x1_4, x2_4)
        c5 = self.temporal_fusion_x5(x1_5, x2_5)

        return c2, c3, c4, c5


class FrequencyRefinement(nn.Module):
    def __init__(self, channels):
        super(FrequencyRefinement, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        freq = torch.fft.rfft2(x.float(), norm="ortho")
        magnitude = torch.log1p(torch.abs(freq))
        magnitude = nn.functional.interpolate(magnitude, size=x.shape[-2:], mode='bilinear')
        return self.proj(x + x * self.gate(magnitude.to(dtype=x.dtype)))


class DilatedTemporalFrequencyFusion(nn.Module):
    def __init__(self, in_c=128, out_c=128):
        super(DilatedTemporalFrequencyFusion, self).__init__()
        self.temporal_fusion = TemporalFeatureFusion(in_c, out_c)
        self.frequency_refinement = nn.ModuleList([
            FrequencyRefinement(out_c),
            FrequencyRefinement(out_c),
            FrequencyRefinement(out_c),
            FrequencyRefinement(out_c),
        ])

    def forward(self, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5):
        c2, c3, c4, c5 = self.temporal_fusion(x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5)
        c2 = self.frequency_refinement[0](c2)
        c3 = self.frequency_refinement[1](c3)
        c4 = self.frequency_refinement[2](c4)
        c5 = self.frequency_refinement[3](c5)

        return c2, c3, c4, c5
