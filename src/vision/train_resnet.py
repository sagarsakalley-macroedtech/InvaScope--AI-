import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "image_dataset"
OUT = ROOT / "models" / "vision"
REPORTS = ROOT / "reports" / "model_metrics"

OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 8
EPOCHS = 5
IMAGE_SIZE = 224

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_dir = DATA / "train"
val_dir = DATA / "val"
test_dir = DATA / "test"

for folder in (train_dir, val_dir, test_dir):
    if not folder.is_dir():
        raise SystemExit(
            f"Missing folder: {folder}\n"
            "Run src\\data\\prepare_training_data.py first."
        )

weights = models.ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

# Use pretrained-model normalization for all splits.
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.meta["mean"],
        std=weights.meta["std"],
    ),
])

val_transform = preprocess
test_transform = preprocess

train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
val_ds = datasets.ImageFolder(val_dir, transform=val_transform)
test_ds = datasets.ImageFolder(test_dir, transform=test_transform)

classes = train_ds.classes

if len(classes) < 2:
    raise SystemExit(
        f"Need at least two training classes; found: {classes}"
    )

if val_ds.classes != classes or test_ds.classes != classes:
    raise SystemExit(
        "Class folders differ between train, val and test.\n"
        f"Train: {classes}\n"
        f"Val: {val_ds.classes}\n"
        f"Test: {test_ds.classes}\n"
        "Collect more images or correct the dataset split first."
    )

if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
    raise SystemExit("One of the dataset splits is empty.")

print("Classes:", classes)
print("Images — train:", len(train_ds))
print("Images — validation:", len(val_ds))
print("Images — test:", len(test_ds))

# Windows-safe DataLoader configuration.
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

model = models.resnet50(weights=weights)

# Freeze the pretrained feature extractor for the baseline.
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

best_val_loss = float("inf")
best_path = OUT / "resnet50_best.pth"

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item() * images.size(0)

    train_loss = train_loss_sum / len(train_ds)

    model.eval()
    val_loss_sum = 0.0
    val_preds, val_true = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            val_loss_sum += loss.item() * images.size(0)
            val_preds.extend(logits.argmax(1).cpu().tolist())
            val_true.extend(labels.cpu().tolist())

    val_loss = val_loss_sum / len(val_ds)
    val_acc = accuracy_score(val_true, val_preds)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"train loss={train_loss:.4f} | "
        f"val loss={val_loss:.4f} | "
        f"val accuracy={val_acc:.3f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "image_size": IMAGE_SIZE,
            "architecture": "resnet50",
            "weights": "ResNet50_Weights.DEFAULT",
        }, best_path)

# Load the best validation checkpoint for final test evaluation.
checkpoint = torch.load(
    best_path, map_location=device, weights_only=True
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_preds, test_true = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images)
        test_preds.extend(logits.argmax(1).cpu().tolist())
        test_true.extend(labels.tolist())

accuracy = accuracy_score(test_true, test_preds)
macro_f1 = f1_score(
    test_true, test_preds, average="macro", zero_division=0
)

report = classification_report(
    test_true,
    test_preds,
    labels=list(range(len(classes))),
    target_names=classes,
    output_dict=True,
    zero_division=0,
)

metrics = {
    "architecture": "ResNet-50",
    "classes": classes,
    "train_images": len(train_ds),
    "validation_images": len(val_ds),
    "test_images": len(test_ds),
    "test_accuracy": float(accuracy),
    "test_macro_f1": float(macro_f1),
    "classification_report": report,
}

metrics_path = REPORTS / "resnet50_metrics.json"
metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

cm = confusion_matrix(
    test_true, test_preds, labels=list(range(len(classes)))
)
display = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=classes
)
fig, ax = plt.subplots(figsize=(10, 8))
display.plot(ax=ax, xticks_rotation=45, colorbar=False)
fig.tight_layout()
fig.savefig(REPORTS / "resnet50_confusion_matrix.png", dpi=150)
plt.close(fig)

print("\nTEST RESULTS")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")
print("Model saved:", best_path)
print("Metrics saved:", metrics_path)
print("Confusion matrix saved.")
print("These results are meaningful only if the test split is independent.")
