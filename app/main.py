import matplotlib.pyplot as plt
from inference import Detector
from visualization import draw_results

detector = Detector("models/last.pt")

img_path = "xray_00009.png"

img, results, cam_map = detector.run(img_path)

output = draw_results(img, results, cam_map, detector.model.names)

plt.imshow(output)
plt.axis("off")
plt.show()