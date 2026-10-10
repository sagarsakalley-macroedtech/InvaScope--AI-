from pathlib import Path

import streamlit as st
import torch
from torch import nn
from PIL import Image
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "vision" / "resnet50_best.pth"

st.set_page_config(
    page_title="InvaScope AI",
    page_icon="🌿",
    layout="centered",
)

st.title("🌿 InvaScope AI")
st.subheader("Invasive Plant Image Classifier")
st.write(
    "Upload a plant photograph to see the predictions "
    "from your trained ResNet-50 model."
)

if not MODEL_PATH.exists():
    st.error(f"Model checkpoint not found: {MODEL_PATH}")
    st.stop()


@st.cache_resource
def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(
        MODEL_PATH, map_location=device, weights_only=True
    )
    classes = checkpoint["classes"]

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, classes, device


model, classes, device = load_model()
weights = models.ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

uploaded = st.file_uploader(
    "Choose a plant image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded plant photograph",
             use_container_width=True)

    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    ranking = torch.argsort(probabilities, descending=True)
    best_index = ranking[0].item()

    st.markdown("### Model prediction")
    st.write(f"**Top class:** {classes[best_index]}")
    st.write(
        f"**Model score:** "
        f"{probabilities[best_index].item():.2%}"
    )

    st.markdown("### Scores for all trained classes")
    for index in ranking:
        i = index.item()
        st.write(classes[i])
        st.progress(probabilities[i].item())
        st.caption(f"{probabilities[i].item():.2%}")

    st.warning(
        "Research prototype only. Scores are not calibrated "
        "probabilities of correct identification. The model may "
        "misclassify unfamiliar plants or species outside its "
        "training classes. Verify results with botanical experts."
    )
else:
    st.info("Upload an image to begin.")
