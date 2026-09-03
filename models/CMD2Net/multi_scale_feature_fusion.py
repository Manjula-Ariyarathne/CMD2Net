import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualFeatureFusionBlock(nn.Module):
    def __init__(self, fuse_d, in_c, out_c=128):
        super(ResidualFeatureFusionBlock, self).__init__()
        self.fuse_d = fuse_d
        self.in_c = in_c
        self.out_c = out_c
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_d, self.out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_c, self.out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_c)
        )
        self.conv_identity = nn.Conv2d(self.in_c, self.out_c, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, c_fuse, c):
        c_fuse = self.conv_fuse(c_fuse)
        c_out = self.relu(c_fuse + self.conv_identity(c))

        return c_out


class AdjacentScaleFeatureAggregation(nn.Module):
    def __init__(self, in_c=None, out_c=128):
        super(AdjacentScaleFeatureAggregation, self).__init__()
        self.in_c = in_c
        self.mid_d = out_c // 2
        self.out_c = out_c
        self.s2_c2_proj = nn.Sequential(
            nn.Conv2d(self.in_c[0], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.s2_c3_proj = nn.Sequential(
            nn.Conv2d(self.in_c[1], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.s2_fuse = ResidualFeatureFusionBlock(self.mid_d * 2, self.in_c[0], self.out_c)
        self.s3_c2_proj = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_c[0], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.s3_c3_proj = nn.Sequential(
            nn.Conv2d(self.in_c[1], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.s3_c4_proj = nn.Sequential(
            nn.Conv2d(self.in_c[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.s3_fuse = ResidualFeatureFusionBlock(self.mid_d * 3, self.in_c[1], self.out_c)
        self.s4_c3_proj = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_c[1], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.s4_c4_proj = nn.Sequential(
            nn.Conv2d(self.in_c[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.s4_c5_proj = nn.Sequential(
            nn.Conv2d(self.in_c[3], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.s4_fuse = ResidualFeatureFusionBlock(self.mid_d * 3, self.in_c[2], self.out_c)
        self.s5_c4_proj = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_c[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.s5_c5_proj = nn.Sequential(
            nn.Conv2d(self.in_c[3], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.s5_fuse = ResidualFeatureFusionBlock(self.mid_d * 2, self.in_c[3], self.out_c)

    def forward(self, c2, c3, c4, c5):
        c2_s2 = self.s2_c2_proj(c2)
        c3_s2 = self.s2_c3_proj(c3)
        c3_s2 = F.interpolate(c3_s2, scale_factor=(2, 2), mode='bilinear')
        s2 = self.s2_fuse(torch.cat([c2_s2, c3_s2], dim=1), c2)
        
        c2_s3 = self.s3_c2_proj(c2)
        c3_s3 = self.s3_c3_proj(c3)
        c4_s3 = self.s3_c4_proj(c4)
        c4_s3 = F.interpolate(c4_s3, scale_factor=(2, 2), mode='bilinear')

        s3 = self.s3_fuse(torch.cat([c2_s3, c3_s3, c4_s3], dim=1), c3)
        c3_s4 = self.s4_c3_proj(c3)
        c4_s4 = self.s4_c4_proj(c4)
        c5_s4 = self.s4_c5_proj(c5)
        c5_s4 = F.interpolate(c5_s4, scale_factor=(2, 2), mode='bilinear')

        s4 = self.s4_fuse(torch.cat([c3_s4, c4_s4, c5_s4], dim=1), c4)
        c4_s5 = self.s5_c4_proj(c4)
        c5_s5 = self.s5_c5_proj(c5)
        s5 = self.s5_fuse(torch.cat([c4_s5, c5_s5], dim=1), c5)

        return s2, s3, s4, s5


class MultiScaleContextRefinement(nn.Module):
    def __init__(self, channels=128, dilations=(1, 3, 5)):
        super(MultiScaleContextRefinement, self).__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=d, dilation=d, groups=channels),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
            for d in dilations
        ])
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * (len(dilations) + 1), channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        global_context = nn.functional.interpolate(
            self.global_branch(x),
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )
        refined = self.fuse(torch.cat(outs + [global_context], dim=1))
        return x + refined * self.gate(refined)


class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_c=None, out_c=128):
        super(MultiScaleFeatureFusion, self).__init__()
        self.adjacent_scale_aggregation = AdjacentScaleFeatureAggregation(in_c, out_c)
        self.refine_x2 = MultiScaleContextRefinement(out_c, dilations=(1, 3, 5))
        self.refine_x3 = MultiScaleContextRefinement(out_c, dilations=(1, 3, 5))
        self.refine_x4 = MultiScaleContextRefinement(out_c, dilations=(1, 3, 5))
        self.refine_x5 = MultiScaleContextRefinement(out_c, dilations=(1, 2, 3))

    def forward(self, c2, c3, c4, c5):
        s2, s3, s4, s5 = self.adjacent_scale_aggregation(c2, c3, c4, c5)
        return self.refine_x2(s2), self.refine_x3(s3), self.refine_x4(s4), self.refine_x5(s5)
