from ultralytics import YOLO
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model = YOLO("models/last.pt")

print("\nModel loaded successfully")
print("Classes:", model.names)
print("Number of classes:", len(model.names))

print("\nArchitecture:")
for i, layer in enumerate(model.model.model):
    print(i, layer.__class__.__name__)