
from pathlib import Path
from threading import Lock
from time import perf_counter

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from ultralytics import YOLO
from app.visualization import draw_detections, encode_image


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "last.pt"

IMAGE_SIZE = 640
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# --------------------------------------------------
# Load model once when the API starts
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

# Prevent concurrent requests from using the same model
# simultaneously. We can revisit this when optimizing serving.
inference_lock = Lock()

# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="PIDray Inference API",
    description=(
        "X-ray threat detection API powered by a "
        "YOLOv8 model trained on the PIDray dataset."
    ),
    version="0.1.0"
)


@app.get("/")
def root():
    return {
        "project": "PIDray",
        "version": "0.1.0",
        "message": "PIDray inference API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.get("/model-info")
def model_info():
    return {
        "model": MODEL_PATH.name,
        "framework": "Ultralytics YOLO",
        "image_size": IMAGE_SIZE,
        "number_of_classes": len(model.names),
        "classes": model.names
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    conf: float = Query(default=0.25, ge=0.0, le=1.0),
    include_image: bool = Query(default=True)
):
    # Check filename extension
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if Path(file.filename or "").suffix.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format."
        )

    # Read uploaded image
    image_bytes = file.file.read(MAX_FILE_SIZE + 1)

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit."
        )

    # Decode image bytes into an OpenCV image
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode uploaded image."
        )

    height, width = image.shape[:2]

    if height == 0 or width == 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid image dimensions."
        )

    # Run inference
    try:
        start_time = perf_counter()

        with inference_lock:
            results = model.predict(
                source=image,
                imgsz=IMAGE_SIZE,
                conf=conf,
                verbose=False
            )

        inference_time_ms = (perf_counter() - start_time) * 1000

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(exc)}"
        )

    # Extract detections
    result = results[0]
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            coordinates = box.xyxy[0].tolist()

            detections.append({
                "class_id": class_id,
                "class_name": model.names[class_id],
                "confidence": round(confidence, 4),
                "bbox": [
                    round(float(value), 2)
                    for value in coordinates
                ]
            })

    annotated_image = None

    if include_image:
        try:
            visual = draw_detections(result)
            annotated_image = encode_image(visual)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Visualization failed: {str(exc)}"
            )

    return {
        "success": True,
        "image_size": {
            "width": width,
            "height": height
        },
        "detections": detections,
        "detection_count": len(detections),
        "inference_time_ms": round(inference_time_ms, 2),
        "annotated_image": annotated_image
    }
