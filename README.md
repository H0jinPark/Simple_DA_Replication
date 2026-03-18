# Office-31 Simple Model Training

This workspace contains the Office-31 dataset directories (`amazon/`, `dslr/`, `webcam/`).

## Quick Start

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Train a model on one domain and optionally evaluate on another:

```bash
python train.py --source amazon --target webcam --epochs 10 --batch-size 32
```

3. The script saves a checkpoint (`checkpoint.pth` by default) containing the model weights and class mapping.

## Notes

- The script uses a pretrained `resnet18` backbone and replaces the final classifier for the Office-31 classes.
- It expects the dataset directory structure to follow `ImageFolder` layout (one folder per class).
