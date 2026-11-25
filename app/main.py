
from pathlib import Path
from threading import Lock
from time import perf_counter

import cv2
import numpy as np

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    Query
)

from ultralytics import YOLO

from app.visualization import draw_detections, encode_image
from app.explainability.gradcam import GradCAM


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "last.pt"

IMAGE_SIZE = 640
MAX_FILE_SIZE = 10 * 1024 * 1024


# --------------------------------------------------
# Load model
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

# Initialize Grad-CAM using the same loaded model
gradcam = GradCAM(model, target_layer_index=20)

# Protect shared model access
inference_lock = Lock()


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="PIDray Inference API",
    description="X-ray threat detection with optional Grad-CAM explainability.",
    version="0.3.0"
)


@app.get("/")
def root():
    return {
        "project": "PIDray",
        "version": "0.3.0",
        "message": "PIDray inference API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "explainability_available": gradcam is not None
    }


@app.get("/model-info")
def model_info():
    return {
        "model": MODEL_PATH.name,
        "framework": "Ultralytics YOLO",
        "image_size": IMAGE_SIZE,
        "number_of_classes": len(model.names),
        "classes": model.names,
        "explainability": {
            "available": True,
            "method": "Grad-CAM",
            "target_layer": 20
        }
    }


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    conf: float = Query(default=0.25, ge=0.0, le=1.0),
    include_image: bool = Query(default=True),
    explain: bool = Query(default=False)
):

    # Validate extension
    allowed_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }

    extension = Path(file.filename or "").suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format."
        )

    # Read image
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

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode uploaded image."
        )

    height, width = image.shape[:2]

    # --------------------------------------------------
    # YOLO inference
    # --------------------------------------------------

    try:
        start_time = perf_counter()

        with inference_lock:
            results = model.predict(
                source=image,
                imgsz=IMAGE_SIZE,
                conf=conf,
                rect=False,
                verbose=False
            )

        inference_time_ms = (
            perf_counter() - start_time
        ) * 1000

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(exc)}"
        )

    result = results[0]

    # --------------------------------------------------
    # Extract detections
    # --------------------------------------------------

    detections = []

    if result.boxes is not None:
        for box in result.boxes:

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            coordinates = [
                round(float(value), 2)
                for value in box.xyxy[0].tolist()
            ]

            detections.append({
                "class_id": class_id,
                "class_name": model.names[class_id],
                "confidence": round(confidence, 4),
                "bbox": coordinates
            })

    # --------------------------------------------------
    # Annotated image
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Grad-CAM explanation
    # --------------------------------------------------

    explanation = None
    explainability_time_ms = None

    if explain and detections:

        # Explain the highest-confidence detection
        selected_index = max(
            range(len(detections)),
            key=lambda i: detections[i]["confidence"]
        )

        selected = detections[selected_index]

        try:
            start_time = perf_counter()

            with inference_lock:
                cam, cam_metadata = gradcam.generate(
                    original_image=image,
                    detection_box=selected["bbox"],
                    class_id=selected["class_id"]
                )

            explainability_time_ms = (
                perf_counter() - start_time
            ) * 1000

            # Convert CAM into a colored heatmap
            heatmap = cv2.applyColorMap(
                np.uint8(255 * cam),
                cv2.COLORMAP_JET
            )

            # Overlay heatmap on original X-ray
            overlay = cv2.addWeighted(
                image,
                0.6,
                heatmap,
                0.4,
                0
            )

            explanation = {
                "method": "Grad-CAM",
                "target_layer": 20,
                "detection_index": selected_index,
                "class_name": selected["class_name"],
                "confidence": selected["confidence"],
                "anchor_index": cam_metadata["anchor_index"],
                "matching_iou": cam_metadata["matching_iou"],
                "raw_class_score": cam_metadata["raw_class_score"],
                "gradcam_image": encode_image(overlay)
            }

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Grad-CAM generation failed: {str(exc)}"
            )

    elif explain:
        explanation = {
            "message": "No detections found. Grad-CAM was not generated."
        }

    # --------------------------------------------------
    # API response
    # --------------------------------------------------

    return {
        "success": True,

        "image_size": {
            "width": width,
            "height": height
        },

        "detections": detections,
        "detection_count": len(detections),

        "inference_time_ms": round(
            inference_time_ms, 2
        ),

        "annotated_image": annotated_image,

        "explanation": explanation,

        "explainability_time_ms": (
            round(explainability_time_ms, 2)
            if explainability_time_ms is not None
            else None
        )
    }
