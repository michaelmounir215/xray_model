
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import io
import os
import gdown

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def read_root():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return f.read()
    return "<h1>API is running, but index.html not found.</h1>"

diseases = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural Thickening", "Hernia", "Fracture"
]

model_path = "chest_xray_chexnet.pth"
drive_file_id = "1O0Lw3IrdMXgxKTIM9VZ66kZ6Dq5zKsS5"  # Replace with your actual file ID

if not os.path.isfile(model_path):
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={drive_file_id}"
    gdown.download(url, model_path, quiet=False)

model = models.densenet121(weights=None)
model.classifier = nn.Linear(1024, 15)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            output = torch.sigmoid(output)

        probs = output[0].tolist()
        max_idx = probs.index(max(probs))
        top_disease = diseases[max_idx]
        top_probability = round(probs[max_idx], 4)

        return {
            "top_disease": top_disease,
            "probability": top_probability
        }

    except Exception as e:
        return {"error": str(e)}
