
import cv2
import torch
import numpy as np


class GradCAM:
    def __init__(self, yolo_model, target_layer_index=20):
        self.yolo = yolo_model
        self.network = yolo_model.model
        self.network.eval()

        self.target_layer = self.network.model[target_layer_index]

        self.image_size = 640

    def letterbox(self, image):
        """Match fixed-size Ultralytics letterbox preprocessing."""

        height, width = image.shape[:2]
        size = self.image_size

        ratio = min(size / height, size / width)

        new_width = round(width * ratio)
        new_height = round(height * ratio)

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR
        )

        dw = (size - new_width) / 2
        dh = (size - new_height) / 2

        left = round(dw - 0.1)
        right = round(dw + 0.1)
        top = round(dh - 0.1)
        bottom = round(dh + 0.1)

        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )

        metadata = {
            "ratio": ratio,
            "left": left,
            "top": top,
            "new_width": new_width,
            "new_height": new_height
        }

        return padded, metadata

    def calculate_iou(self, boxes, target_box):
        """Calculate IoU between candidate boxes and one target box."""

        x1 = torch.maximum(boxes[:, 0], target_box[0])
        y1 = torch.maximum(boxes[:, 1], target_box[1])
        x2 = torch.minimum(boxes[:, 2], target_box[2])
        y2 = torch.minimum(boxes[:, 3], target_box[3])

        intersection = (
            (x2 - x1).clamp(min=0)
            * (y2 - y1).clamp(min=0)
        )

        area_a = (
            (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
            * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        )

        area_b = (
            (target_box[2] - target_box[0]).clamp(min=0)
            * (target_box[3] - target_box[1]).clamp(min=0)
        )

        union = area_a + area_b - intersection

        return intersection / (union + 1e-8)

    def generate(self, original_image, detection_box, class_id):
        """
        Generate detection-specific Grad-CAM.

        detection_box: [x1, y1, x2, y2] in original image coordinates.
        class_id: predicted class ID from the API detection.
        """

        padded, metadata = self.letterbox(original_image)

        # Match Ultralytics BGR -> RGB preprocessing
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(
            np.ascontiguousarray(rgb.transpose(2, 0, 1))
        ).float().unsqueeze(0) / 255.0

        device = next(self.network.parameters()).device
        tensor = tensor.to(device)
        tensor.requires_grad_(True)

        activations = {}

        def forward_hook(module, inputs, output):
            activations["feature"] = output

        handle = self.target_layer.register_forward_hook(forward_hook)

        try:
            self.network.zero_grad(set_to_none=True)

            with torch.enable_grad():
                output = self.network(tensor)

                if not isinstance(output, tuple) or len(output) < 2:
                    raise RuntimeError(
                        "Unexpected YOLO output structure."
                    )

                predictions = output[0]
                auxiliary = output[1]

                if not isinstance(auxiliary, dict):
                    raise RuntimeError(
                        "YOLO auxiliary output is not a dictionary."
                    )

                if "scores" not in auxiliary:
                    raise RuntimeError(
                        "Raw class scores are missing."
                    )

                raw_scores = auxiliary["scores"]

                # First four prediction channels are decoded xywh.
                boxes = predictions[0, :4, :].transpose(0, 1)

                center_x = boxes[:, 0]
                center_y = boxes[:, 1]
                box_width = boxes[:, 2]
                box_height = boxes[:, 3]

                decoded_boxes = torch.stack([
                    center_x - box_width / 2,
                    center_y - box_height / 2,
                    center_x + box_width / 2,
                    center_y + box_height / 2
                ], dim=1)

                # Map the API's original-image box into letterbox space.
                ratio = metadata["ratio"]
                left = metadata["left"]
                top = metadata["top"]

                x1, y1, x2, y2 = detection_box

                target_box = torch.tensor(
                    [
                        x1 * ratio + left,
                        y1 * ratio + top,
                        x2 * ratio + left,
                        y2 * ratio + top
                    ],
                    device=device,
                    dtype=decoded_boxes.dtype
                )

                # Find the raw prediction position matching this detection.
                ious = self.calculate_iou(
                    decoded_boxes,
                    target_box
                )

                anchor_index = int(ious.argmax().item())
                best_iou = float(ious[anchor_index].item())

                if best_iou < 0.1:
                    raise RuntimeError(
                        f"Could not reliably match detection to raw "
                        f"prediction (IoU={best_iou:.3f})."
                    )

                if not 0 <= class_id < raw_scores.shape[1]:
                    raise ValueError("Invalid class ID.")

                # Select the score for this detection and class.
                target_score = raw_scores[
                    0, class_id, anchor_index
                ].sigmoid()

                feature = activations["feature"]

                gradients = torch.autograd.grad(
                    outputs=target_score,
                    inputs=feature
                )[0]

                # Grad-CAM channel weights
                weights = gradients.mean(
                    dim=(2, 3),
                    keepdim=True
                )

                cam = (weights * feature).sum(dim=1)
                cam = torch.relu(cam)

                cam = cam.squeeze().detach().cpu().numpy()

            # Normalize heatmap
            cam_min = cam.min()
            cam_max = cam.max()

            cam = (
                (cam - cam_min)
                / (cam_max - cam_min + 1e-8)
            )

            # Resize to letterbox dimensions
            cam = cv2.resize(
                cam,
                (self.image_size, self.image_size)
            )

            # Remove letterbox padding
            top = metadata["top"]
            left = metadata["left"]
            new_height = metadata["new_height"]
            new_width = metadata["new_width"]

            cam = cam[
                top:top + new_height,
                left:left + new_width
            ]

            # Restore original image dimensions
            original_height, original_width = original_image.shape[:2]

            cam = cv2.resize(
                cam,
                (original_width, original_height)
            )

            return cam, {
                "anchor_index": anchor_index,
                "matching_iou": round(best_iou, 4),
                "raw_class_score": round(
                    float(target_score.detach().item()), 5
                )
            }

        finally:
            handle.remove()
