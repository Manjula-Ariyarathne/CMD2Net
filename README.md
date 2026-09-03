# CMD2Net

CMD2Net is a remote sensing mining change detection model. This repository contains the code for training and testing it on the MineNetCD dataset.

## Pretrained SwinV2 weights

Before training or testing, download Microsoft's pretrained `swinv2_small_patch4_window16_256.pth` checkpoint from the following link:

[Download the pretrained SwinV2 Small checkpoint](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_small_patch4_window16_256.pth)

Create a `pretrained` directory in the repository root and place the downloaded file inside it:

```text
pretrained/
└── swinv2_small_patch4_window16_256.pth
```

## Dataset structure

By default, `configs/CMD2Net.yml` expects MineNetCD in the repository root with this structure:

```text
MineNetCD/
├── train/
│   ├── imageA/
│   ├── imageB/
│   └── label/
├── val/
│   ├── imageA/
│   ├── imageB/
│   └── label/
└── test/
    ├── imageA/
    ├── imageB/
    └── label/
```

Corresponding images and labels must have the same filename.

## Training

From the repository root, run:

```bash
accelerate launch train.py --config configs/CMD2Net.yml
```

The best checkpoint is saved by default under:

```text
checkpoints/MineNetCD/CMD2Net/BestF1/
```

## Testing

Test the best saved checkpoint with:

```bash
accelerate launch test.py --model checkpoints/MineNetCD/CMD2Net/BestF1/
```

Training parameters, dataset paths, batch sizes, and checkpoint paths can be changed in `configs/CMD2Net.yml`.
