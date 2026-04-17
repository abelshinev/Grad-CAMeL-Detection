import cv2
import numpy as np

COLORS = {
    "box": (0, 180, 255),
    "text": (255, 255, 255)
}

FONT = cv2.FONT_HERSHEY_DUPLEX


def get_explanation(class_name):
    mapping = {
        "Bullet": "small dense metallic object",
        "Baton": "elongated dense structure",
        "Knife": "sharp edged dense object"
    }
    return mapping.get(class_name, "suspicious dense object")


def draw_results(img, results, cam_map, class_names):
    img_vis = img.copy()

    cam_resized = cv2.resize(cam_map, (img.shape[1], img.shape[0]))

    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    classes = results[0].boxes.cls.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cls_id = int(classes[i])
        score = scores[i]

        label = f"{class_names[cls_id]} ({score:.2f})"
        explanation = get_explanation(class_names[cls_id])

        # CAM mask
        mask = np.zeros_like(cam_resized)
        mask[y1:y2, x1:x2] = cam_resized[y1:y2, x1:x2]

        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        img_vis = (0.7 * img_vis + 0.3 * heatmap).astype(np.uint8)

        # Box
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), COLORS["box"], 2)

        # Label background
        (w, h), _ = cv2.getTextSize(label, FONT, 0.6, 2)
        cv2.rectangle(img_vis,
                      (x1, y1 - h - 10),
                      (x1 + w + 10, y1),
                      COLORS["box"],
                      -1)

        # Label text
        cv2.putText(img_vis, label,
                    (x1 + 5, y1 - 5),
                    FONT, 0.6,
                    COLORS["text"], 2)

        # Leader line
        line_end = (
            min(x2 + 120, img.shape[1] - 150),
            min(y2 + 40, img.shape[0] - 20)
        )
        cv2.line(img_vis, (x2, y2), line_end, COLORS["box"], 2)

        # Explanation text
        cv2.putText(img_vis, explanation,
                    (line_end[0] + 5, line_end[1]),
                    FONT, 0.5,
                    COLORS["text"], 2)

    return img_vis
