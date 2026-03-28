import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile
from PIL import Image
import numpy as np

# Load model
num_classes = 6  # update if needed

# load checkpoint
checkpoint = torch.load("model.pth", map_location="cpu")

# class names from model
class_names = checkpoint["class_names"]
num_classes = len(class_names)

# rebuild model
model = models.convnext_tiny(weights=None)

num_ftrs = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

# load weights
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()


app = FastAPI()

@app.get("/")
def home():
    return {"message": "Backend is working"}

import random

def run_model(image):

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    return (
        class_names[predicted.item()],
        probs[0][predicted].item()
    )

@app.post("/predict")
async def predict(file: UploadFile):

    image = Image.open(file.file).convert("RGB")

    breed, confidence = run_model(image)

    return {
        "status": "success",
        "breed": breed,
        "confidence": round(confidence, 3)
    }
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
