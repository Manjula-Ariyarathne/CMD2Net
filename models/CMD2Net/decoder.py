import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderHierarchicalFeatureFusion(nn.Module):
    def __init__(self, in_c, out_c=128):
        super(DecoderHierarchicalFeatureFusion, self).__init__()
        self.out_c = out_c

        self.d_in1 = nn.ModuleDict({
            "conv": nn.Conv2d(in_c[1], out_c // 2, 1, 1, 0, bias=False),
            "bn": nn.BatchNorm2d(out_c // 2),
            "act": nn.SiLU()
        })
        self.d_in2 = nn.ModuleDict({
            "conv": nn.Conv2d(in_c[0], out_c // 2, 1, 1, 0, bias=False),
            "bn": nn.BatchNorm2d(out_c // 2),
            "act": nn.SiLU()
        })
        self.conv = nn.ModuleDict({
            "conv": nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            "bn": nn.BatchNorm2d(out_c),
            "act": nn.SiLU()
        })

        l_hidden_c = int(2 * in_c[1] / 3)
        self.glu_l = nn.ModuleDict({
            "fc1": nn.Conv2d(in_c[1], l_hidden_c * 2, kernel_size=1),
            "dwconv": nn.Sequential(
                nn.Conv2d(
                    l_hidden_c,
                    l_hidden_c,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=l_hidden_c
                ),
                nn.GELU()
            ),
            "fc2": nn.Conv2d(l_hidden_c, out_c // 2, kernel_size=1),
            "drop": nn.Dropout(0.)
        })

        h_hidden_c = int(2 * in_c[0] / 3)
        self.glu_h = nn.ModuleDict({
            "fc1": nn.Conv2d(in_c[0], h_hidden_c * 2, kernel_size=1),
            "dwconv": nn.Sequential(
                nn.Conv2d(
                    h_hidden_c,
                    h_hidden_c,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=h_hidden_c
                ),
                nn.GELU()
            ),
            "fc2": nn.Conv2d(h_hidden_c, out_c // 2, kernel_size=1),
            "drop": nn.Dropout(0.)
        })

    def forward(self, x):
        h_feature, l_feature = x

        g_l_feature, l_gate = self.glu_l["fc1"](l_feature).chunk(2, dim=1)
        g_l_feature = self.glu_l["dwconv"](g_l_feature) * l_gate
        g_l_feature = self.glu_l["drop"](g_l_feature)
        g_l_feature = self.glu_l["fc2"](g_l_feature)
        g_l_feature = torch.sigmoid(self.glu_l["drop"](g_l_feature))

        g_h_feature, h_gate = self.glu_h["fc1"](h_feature).chunk(2, dim=1)
        g_h_feature = self.glu_h["dwconv"](g_h_feature) * h_gate
        g_h_feature = self.glu_h["drop"](g_h_feature)
        g_h_feature = self.glu_h["fc2"](g_h_feature)
        g_h_feature = torch.sigmoid(self.glu_h["drop"](g_h_feature))

        l_feature = self.d_in1["act"](self.d_in1["bn"](self.d_in1["conv"](l_feature)))
        h_feature = self.d_in2["act"](self.d_in2["bn"](self.d_in2["conv"](h_feature)))

        l_feature = l_feature + l_feature * g_l_feature + (1 - g_l_feature) * F.interpolate(
            g_h_feature * h_feature,
            size=l_feature.size()[2:],
            mode="bilinear",
            align_corners=False
        )
        h_feature = h_feature + h_feature * g_h_feature + (1 - g_h_feature) * F.interpolate(
            g_l_feature * l_feature,
            size=h_feature.size()[2:],
            mode="bilinear",
            align_corners=False
        )

        h_feature = F.interpolate(
            h_feature,
            size=l_feature.size()[2:],
            mode="bilinear",
            align_corners=False
        )
        out = torch.cat([h_feature, l_feature], dim=1)
        return self.conv["act"](self.conv["bn"](self.conv["conv"](out)))


class DecoderPredictionFusionBlock(nn.Module):
    def __init__(self, high_c, low_c, out_c):
        super(DecoderPredictionFusionBlock, self).__init__()
        self.hierarchical_fusion = DecoderHierarchicalFeatureFusion([high_c, low_c], out_c=out_c)
        self.cls = nn.Conv2d(out_c, 1, kernel_size=1)

    def forward(self, high_feature, low_feature):
        out = self.hierarchical_fusion([high_feature, low_feature])
        mask = self.cls(out)
        return out, mask


class Decoder(nn.Module):
    def __init__(self, mid_c=128):
        super(Decoder, self).__init__()
        self.mid_c = mid_c

        self.p5_refine = nn.Sequential(
            nn.Conv2d(self.mid_c, self.mid_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_c),
            nn.ReLU(inplace=True)
        )
        self.decoder_fusion_p4 = DecoderPredictionFusionBlock(self.mid_c, self.mid_c, self.mid_c)
        self.decoder_fusion_p3 = DecoderPredictionFusionBlock(self.mid_c, self.mid_c, self.mid_c)
        self.decoder_fusion_p2 = DecoderPredictionFusionBlock(self.mid_c, self.mid_c, self.mid_c)
        self.cls_p5 = nn.Conv2d(self.mid_c, 1, kernel_size=1)

    def forward(self, d2, d3, d4, d5):
        p5 = self.p5_refine(d5)
        out_p5 = self.cls_p5(p5)

        p4, out_p4 = self.decoder_fusion_p4(p5, d4)
        p3, out_p3 = self.decoder_fusion_p3(p4, d3)
        p2, out_p2 = self.decoder_fusion_p2(p3, d2)

        return out_p2, out_p3, out_p4, out_p5
