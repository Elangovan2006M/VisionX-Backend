# app/services/disease_service.py
import os
import torch
import tempfile
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from io import BytesIO
from gradio_client import Client, handle_file

# Config
DISEASE_MODEL = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
HF_SPACE = "muxiddin19/blip-image-captioning-api"
HF_API_NAME = "/predict"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Disease classifier
disease_processor = AutoImageProcessor.from_pretrained(DISEASE_MODEL)
disease_model = AutoModelForImageClassification.from_pretrained(DISEASE_MODEL).to(device).eval()

# Hugging Face Space client
hf_client = Client(HF_SPACE)


async def analyze_plant_image(image_file=None):
    """
    Takes raw image bytes.
    - Runs disease classification (MobileNet)
    - Gets caption from Hugging Face Space (BLIP)
    """
    if not image_file:
        return {
            "predicted_class": "Unknown",
            "confidence": 0.0,
            "description": "No image provided"
        }

    # Read image
    image_bytes = await image_file.read() if hasattr(image_file, "read") else image_file
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # 1️⃣ Disease prediction
    inputs = disease_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = disease_model(**inputs).logits
        pred_idx = outputs.argmax(-1).item()
        confidence = torch.nn.functional.softmax(outputs, dim=-1)[0][pred_idx].item() * 100
        predicted_class = disease_model.config.id2label[pred_idx]

    # 2️⃣ Image caption from Hugging Face Space
    description = "No caption available"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        result = hf_client.predict(
            input=handle_file(tmp_path),
            api_name=HF_API_NAME
        )

        # print(" Raw HuggingFace Space response:", result)

        if isinstance(result, str) and result.strip():
            description = result.strip()

        # cleanup
        os.remove(tmp_path)

    except Exception as e:
        print("Captioning failed:", e)

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "description": description,
    }
