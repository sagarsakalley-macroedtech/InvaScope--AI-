import argparse
from pathlib import Path

import torch
from torch import nn
from PIL import Image
from torchvision import models

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "vision" / "resnet50_best.pth"

parser = argparse.ArgumentParser(
    description="Predict a candidate plant species from an image."
)
parser.add_argument("image", help="Path to the plant image")
args = parser.parse_args()

image_path = Path(args.image).resolve()

if not MODEL_PATH.exists():
    raise SystemExit(f"Model not found: {MODEL_PATH}")

if not image_path.is_file():
    raise SystemExit(f"Image not found: {image_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    MODEL_PATH, map_location=device, weights_only=True
)
classes = checkpoint["classes"]

weights = models.ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

with Image.open(image_path) as source:
    image = source.convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)

with torch.no_grad():
    probabilities = torch.softmax(model(tensor), dim=1)[0]
    confidence, index = probabilities.max(dim=0)

print("\nInvaScope AI — Plant Image Prediction")
print("------------------------------------")
print("Predicted class:", classes[index.item()])
print(f"Model score: {confidence.item():.2%}")
print("\nScores for all trained classes:")

for i in torch.argsort(probabilities, descending=True):
    print(f"  {classes[i.item()]}: {probabilities[i].item():.2%}")

print(
    "\nResearch note: this score is not a calibrated probability "
    "that the species identification is correct. The model always "
    "selects one of its trained classes, even for an unrelated plant."
)
