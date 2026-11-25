import base64

with open("response.txt", "r") as f:
    encoded = f.read().strip()

image_bytes = base64.b64decode(encoded)

with open("annotated_result.jpg", "wb") as f:
    f.write(image_bytes)

print("Image decoded successfully!")