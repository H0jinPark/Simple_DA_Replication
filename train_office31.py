#!/usr/bin/env python3
"""Train a simple classifier on Office-31 (amazon/dslr/webcam).

This script trains a ResNet-18 model on one domain (source) and optionally evaluates on another (target).

Usage:
  python train_office31.py --source amazon --target webcam --epochs 10

The dataset folders should be laid out like:
  office-31/
    amazon/<class>/*
    dslr/<class>/*
    webcam/<class>/*

"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_dataloader(root: str, batch_size: int, train: bool, num_workers: int):
    # Standard ImageNet-style preprocessing for ResNet
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    dataset = datasets.ImageFolder(root, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)
    return loader, dataset


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    count = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        count += inputs.size(0)

    epoch_loss = running_loss / count
    epoch_acc = running_correct / count
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    count = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            count += inputs.size(0)

    loss = running_loss / count
    acc = running_correct / count
    return loss, acc


def main():
    parser = argparse.ArgumentParser(description="Train a simple Office-31 classifier")
    parser.add_argument("--root", default=".", help="Path to Office-31 root folder")
    parser.add_argument("--source", default="amazon", choices=["amazon", "dslr", "webcam"], help="Domain to train on")
    parser.add_argument("--target", default=None, choices=["amazon", "dslr", "webcam"], help="Domain to evaluate on (optional)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--workers", type=int, default=4, help="Data loader worker count")
    parser.add_argument("--device", default=None, help="Device to use (cpu or cuda). Defaults to cuda if available")
    parser.add_argument("--save", default="checkpoint.pth", help="Path to save model checkpoint")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    source_root = os.path.join(args.root, args.source)
    if not os.path.isdir(source_root):
        raise FileNotFoundError(f"Source directory not found: {source_root}")

    train_loader, train_dataset = get_dataloader(source_root, args.batch_size, train=True, num_workers=args.workers)
    print(f"Source ({args.source}) classes: {train_dataset.classes}")

    target_loader = None
    if args.target:
        target_root = os.path.join(args.root, args.target)
        if not os.path.isdir(target_root):
            raise FileNotFoundError(f"Target directory not found: {target_root}")
        target_loader, target_dataset = get_dataloader(target_root, args.batch_size, train=False, num_workers=args.workers)

        if train_dataset.classes != target_dataset.classes:
            print("Warning: Class ordering differs between domains. Evaluation results may be incorrect." )

    num_classes = len(train_dataset.classes)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} - train loss: {train_loss:.4f} acc: {train_acc:.4f} ({time.time()-start:.1f}s)")

        if target_loader is not None:
            val_loss, val_acc = evaluate(model, target_loader, criterion, device)
            print(f"  target({args.target}) loss: {val_loss:.4f} acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "classes": train_dataset.classes,
                }, args.save)
                print(f"  Saved best model (acc={best_val_acc:.4f}) to {args.save}")

    if args.target is None:
        torch.save({
            "epoch": args.epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "classes": train_dataset.classes,
        }, args.save)
        print(f"Saved model to {args.save}")


if __name__ == "__main__":
    main()
