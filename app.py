import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile
from PIL import Image
import numpy as np
from database import SessionLocal
from models import User, Cattle, Prediction
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Form

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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Backend is working"}


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
#PREDICTIONNNNNNN
@app.post("/predict")

@app.post("/predict")
async def predict(file: UploadFile, cattle_id: int = Form(...)):

    print("API HIT")  

    image = Image.open(file.file).convert("RGB")

    breed, confidence = run_model(image)

    print(breed, confidence)  
    db = SessionLocal()

    new_prediction = Prediction(
        cattle_id=cattle_id,
        predicted_breed=breed,
        confidence=confidence
    )

    db.add(new_prediction)
    db.commit()
    db.close()

    return {
        "status": "success",
        "breed": breed,
        "confidence": round(confidence, 3)
    }
    print("API HIT")  

    image = Image.open(file.file).convert("RGB")

    breed, confidence = run_model(image)

    print(breed, confidence)  
    db = SessionLocal()

    new_prediction = Prediction(
        cattle_id=cattle_id,
        predicted_breed=breed,
        confidence=confidence
    )

    db.add(new_prediction)
    db.commit()
    db.close()

    return {
        "status": "success",
        "breed": breed,
        "confidence": round(confidence, 3)
    }

# all the APIs are here for db
#First Real Feature → USER REGISTER
class UserCreate(BaseModel):
    name: str
    email: str
    password: str

@app.post("/register")
def register(user: UserCreate):
    db = SessionLocal()

    new_user = User(
        name=user.name,
        email=user.email,
        password=user.password
    )

    db.add(new_user)
    db.commit()
    db.close()

    return {"message": "User registered"}
#Add Cattle API

class CattleCreate(BaseModel):
    user_id: int
    breed: str
    count: int

@app.post("/add-cattle")
def add_cattle(data: CattleCreate):
    db = SessionLocal()

    cattle = Cattle(
        user_id=data.user_id,
        breed=data.breed,
        count=data.count
    )

    db.add(cattle)
    db.commit()
    db.close()

    return {"message": "Cattle added"}

#Get All Cattle of a User
@app.get("/get-cattle/{user_id}")
def get_cattle(user_id: int):

    db = SessionLocal()

    cattle = db.query(Cattle).filter(Cattle.user_id == user_id).all()

    db.close()

    return cattle
#get prediction history@app.get("/history/{cattle_id}")
def get_history(cattle_id: int):

    db = SessionLocal()

    predictions = db.query(Prediction).filter(
        Prediction.cattle_id == cattle_id
    ).all()

    db.close()

    return predictions
@app.get("/get-users")
def get_users():
    db = SessionLocal()
    users = db.query(User).all()
    db.close()
    return users

