from ultralytics import YOLO
import time

model = YOLO("models/last.pt")

start = time.perf_counter()

results = model.predict(
    source="xray_00009.png",
    imgsz=640,
    conf=0.25,
    verbose=False
)

elapsed = time.perf_counter() - start

result = results[0]

print("Inference time:", round(elapsed, 3), "seconds")
print("Detections:", len(result.boxes))

for box in result.boxes:
    cls_id = int(box.cls[0])
    confidence = float(box.conf[0])

    print(
        model.names[cls_id],
        round(confidence, 3),
        box.xyxy[0].tolist()
    )