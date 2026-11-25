
import cv2


def draw_detections(result):
    """
    Generate an annotated image from an Ultralytics prediction result.

    Returns a BGR NumPy image.
    """
    annotated_image = result.plot(
        boxes=True,
        labels=True,
        conf=True
    )

    return annotated_image


def encode_image(image):
    """
    Encode an OpenCV image into a base64-ready JPEG string.
    """
    import base64

    success, buffer = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 85]
    )

    if not success:
        raise ValueError("Failed to encode annotated image")

    return base64.b64encode(buffer).decode("utf-8")
