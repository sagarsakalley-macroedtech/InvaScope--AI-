import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "image_dataset"
OUT = ROOT / "models" / "vision"
REPORTS = ROOT / "reports" / "model_metrics"

OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

EPOCHS = 5
BATCH_SIZE = 8
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device:", device)

weights = models.ResNet50_Weights.DEFAULT

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

train_dir = DATA / "train"
val_dir = DATA / "val"
test_dir = DATA / "test"

for folder in (train_dir, val_dir, test_dir):
    if not folder.is_dir():
        raise SystemExit(
            f"Missing {folder}. Run prepare_training_data.py first."
        )

train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
val_ds = datasets.ImageFolder(val_dir, transform=weights.transforms())
test_ds = datasets.ImageFolder(test_dir, transform=weights.transforms())

classes = train_ds.classes

if len(classes) < 2:
    raise SystemExit(f"Need at least 2 training classes. Found: {classes}")

if val_ds.classes != classes or test_ds.classes != classes:
    raise SystemExit("Train, validation and test classes do not match.")

if not len(train_ds) or not len(val_ds) or not len(test_ds):
    raise SystemExit("One dataset split is empty.")

print("Classes:", classes)
print("Train:", len(train_ds), "| Validation:", len(val_ds),
      "| Test:", len(test_ds))

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

for parameter in model.parameters():
    parameter.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
best_val_loss = float("inf")
best_path = OUT / "resnet50_best.pth"

for epoch in range(EPOCHS):
    model.train()
    train_total_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_total_loss += loss.item() * len(labels)

    model.eval()
    val_total_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            val_total_loss += loss.item() * len(labels)
            val_correct += (logits.argmax(1) == labels).sum().item()

    train_loss = train_total_loss / len(train_ds)
    val_loss = val_total_loss / len(val_ds)
    val_accuracy = val_correct / len(val_ds)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_accuracy={val_accuracy:.3f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "architecture": "resnet50",
        }, best_path)

checkpoint = torch.load(
    best_path, map_location=device, weights_only=True
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

confusion = np.zeros((len(classes), len(classes)), dtype=int)
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images).argmax(1)

        for actual, predicted in zip(
            labels.cpu().tolist(), predictions.cpu().tolist()
        ):
            confusion[actual, predicted] += 1

        correct += (predictions == labels).sum().item()
        total += len(labels)

accuracy = correct / total if total else 0.0
class_f1 = []

for index in range(len(classes)):
    tp = confusion[index, index]
    fp = confusion[:, index].sum() - tp
    fn = confusion[index, :].sum() - tp
    denominator = 2 * tp + fp + fn
    class_f1.append(
        2 * tp / denominator if denominator else 0.0
    )

macro_f1 = float(np.mean(class_f1))

metrics = {
    "architecture": "ResNet-50",
    "classes": classes,
    "train_images": len(train_ds),
    "validation_images": len(val_ds),
    "test_images": total,
    "test_accuracy": accuracy,
    "test_macro_f1": macro_f1,
    "confusion_matrix": confusion.tolist(),
    "per_class_f1": dict(zip(classes, class_f1)),
}

(REPORTS / "resnet50_metrics.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)

fig, ax = plt.subplots(figsize=(9, 7))
ax.imshow(confusion)
ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
ax.set_yticks(range(len(classes)), classes)
ax.set_xlabel("Predicted species")
ax.set_ylabel("Actual species")
ax.set_title("ResNet-50 Test Confusion Matrix")

for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, str(confusion[i, j]), ha="center", va="center")

fig.tight_layout()
fig.savefig(REPORTS / "resnet50_confusion_matrix.png", dpi=150)
plt.close(fig)

print("\nTest accuracy:", round(accuracy, 4))
print("Test macro-F1:", round(macro_f1, 4))
print("Model saved:", best_path)
print("Metrics saved:", REPORTS / "resnet50_metrics.json")

