from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwinV2Small(nn.Module):
    """SwinV2 Small."""

    model_name = "swinv2_small_window16_256"
    pretrained_path = Path(__file__).resolve().parents[2] / "pretrained" / "swinv2_small_patch4_window16_256.pth"
    target_reductions = (4, 8, 16, 32)

    def __init__(self, pretrained=True, pretrained_path=None):
        super(SwinV2Small, self).__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "CMD2Net requires timm for local SwinV2 Small ImageNet-1K. "
                "Install dependencies with `uv pip install -r requirements.txt`."
            ) from exc

        self.backbone = timm.create_model(
            self.model_name,
            pretrained=False,
            features_only=True,
            img_size=256,
        )
        if pretrained:
            self._load_microsoft_pretrained(pretrained_path or self.pretrained_path)

        reductions = list(self.backbone.feature_info.reduction())
        channels = list(self.backbone.feature_info.channels())
        self.selected_indices = self._select_feature_indices(reductions)
        self.out_channels = [channels[index] for index in self.selected_indices]

    def _load_microsoft_pretrained(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SwinV2 checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        own_state = self.backbone.state_dict()
        converted_state = {}
        skipped_names = ("relative_position_index", "relative_coords_table", "attn_mask")
        for key, value in state_dict.items():
            if any(name in key for name in skipped_names):
                continue
            converted_key = self._convert_microsoft_key(key)
            if converted_key in own_state and own_state[converted_key].shape == value.shape:
                converted_state[converted_key] = value

        incompatible = self.backbone.load_state_dict(converted_state, strict=False)
        if not converted_state:
            raise RuntimeError(f"No compatible SwinV2 weights were loaded from {checkpoint_path}")
        if incompatible.missing_keys:
            missing = ", ".join(incompatible.missing_keys[:5])
            print(f"Loaded local SwinV2 checkpoint with missing keys: {missing} ...")

    def _convert_microsoft_key(self, key):
        for index in range(3):
            key = key.replace(f"layers.{index}.downsample.", f"layers_{index + 1}.downsample.")
        for index in range(4):
            key = key.replace(f"layers.{index}.", f"layers_{index}.")
        return key

    def _select_feature_indices(self, reductions):
        selected_indices = []
        for target_reduction in self.target_reductions:
            candidates = [
                (abs(reduction - target_reduction), index)
                for index, reduction in enumerate(reductions)
                if index not in selected_indices
            ]
            if not candidates:
                raise RuntimeError(
                    f"{self.model_name} does not expose enough feature stages: {reductions}"
                )
            selected_indices.append(min(candidates)[1])
        return selected_indices

    def _to_channels_first(self, feature, expected_channels):
        if feature.shape[1] == expected_channels:
            return feature
        if feature.shape[-1] == expected_channels:
            return feature.permute(0, 3, 1, 2).contiguous()
        raise RuntimeError(
            f"{self.model_name} returned an unexpected feature shape: {feature.shape}"
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        features = self.backbone(x)
        selected_features = [
            self._to_channels_first(features[index], channels)
            for index, channels in zip(self.selected_indices, self.out_channels)
        ]

        normalized_features = []
        for feature, target_reduction in zip(selected_features, self.target_reductions):
            target_size = (
                max(1, input_size[0] // target_reduction),
                max(1, input_size[1] // target_reduction),
            )
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            normalized_features.append(feature)

        c2, c3, c4, c5 = normalized_features
        return None, c2, c3, c4, c5
