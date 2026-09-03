import torch
import torch.nn as nn

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_c):
        super(ChannelAttentionModule, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_c, in_c, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_c, in_c, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        avg_out = self.global_avg_pool(x).view(batch_size, channels)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)
        avg_out = self.sigmoid(avg_out).view(batch_size, channels, 1, 1)

        return x * avg_out


class ChannelAttention(nn.Module):
    def __init__(self, feature_channels=None):
        super(ChannelAttention, self).__init__()
        self.feature_channels = feature_channels
        self.attention_x2 = ChannelAttentionModule(self.feature_channels[0])
        self.attention_x3 = ChannelAttentionModule(self.feature_channels[1])
        self.attention_x4 = ChannelAttentionModule(self.feature_channels[2])
        self.attention_x5 = ChannelAttentionModule(self.feature_channels[3])

    def forward(self, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5):
        x1_2 = self.attention_x2(x1_2)
        x1_3 = self.attention_x3(x1_3)
        x1_4 = self.attention_x4(x1_4)
        x1_5 = self.attention_x5(x1_5)

        x2_2 = self.attention_x2(x2_2)
        x2_3 = self.attention_x3(x2_3)
        x2_4 = self.attention_x4(x2_4)
        x2_5 = self.attention_x5(x2_5)

        return x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5
