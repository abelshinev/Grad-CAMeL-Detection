import torch
import cv2
import numpy as np
from ultralytics import YOLO


class YOLOv8GradCAM:
    def __init__(self, target_layer):
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(forward_hook)

    def generate_cam(self):
        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)

        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.squeeze().detach().cpu().numpy()


class Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.target_layer = self.model.model.model[20]
        self.cam = YOLOv8GradCAM(self.target_layer)

    def preprocess(self, img_path):
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img_resized = cv2.resize(img_rgb, (640, 640))

        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0) / 255.0

        device = next(self.model.model.parameters()).device
        img_tensor = img_tensor.to(device)

        self.model.model.model.to(device)

        return img_rgb, img_tensor

    def run(self, img_path):
        img_rgb, img_tensor = self.preprocess(img_path)

        self.model.model.eval()
        self.model.model.zero_grad()

        img_tensor.requires_grad_(True)

        with torch.enable_grad():
            results = self.model(img_path)

            _ = self.model.model(img_tensor)

            activations = self.cam.activations
            target_score = activations.mean()

            grads = torch.autograd.grad(
                outputs=target_score,
                inputs=activations,
                retain_graph=True
            )[0]

            self.cam.gradients = grads
            cam_map = self.cam.generate_cam()

        return img_rgb, results, cam_map