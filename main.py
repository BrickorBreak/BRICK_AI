from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from model import model, transform
import io

app = FastAPI()

@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return {
        "category": f"class_{predicted.item()}",
        "confidence": float(confidence.item())
    }
