import cv2
import numpy as np
from ultralytics import YOLO

from classifier import FruitClassifier


class AppleProcessor:
    def __init__(self,model_name) -> None:
        self.ripeness_classifier = FruitClassifier()
        self.bounding_box_detector_model = YOLO("../data/weights/"+model_name)

    def process_image(self, uploaded_file):
        count = 0
        # Convert the uploaded file to OpenCV image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(type(image_rgb))

        # Ripeness Detector Model
        ripeness_classes = list()

        # Perform YOLOv5 detection
        results = self.bounding_box_detector_model(image)

        # Draw bounding boxes on the image
        for r in results:
            for box in r.boxes:
                if float(box.conf) > 0.3:
                    cx, cy, w, h = map(float, box.xywh[0])
                    cx, cy, w, h = map(float, box.xywh[0])
                    x1 = cx - 0.5 * w
                    y1 = cy - 0.5 * h
                    x2 = cx + 0.5 * w
                    y2 = cy + 0.5 * h

                    cv2.rectangle(
                        image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 5
                    )
                    cv2.rectangle(
                        image_rgb,
                        (int(x1) + 2, int(y1) + 2),
                        (int(x2) - 2, int(y2) - 2),
                        (255, 0, 0),
                        3,
                    )

                    # Crop the image
                    single_apple_image = image_rgb[int(y1) : int(y2), int(x1) : int(x2)]
                    print(type(single_apple_image))
                    ripeness_class = self.ripeness_classifier.predict(
                        img=single_apple_image
                    )
                    ripeness_classes.append(ripeness_class)

                    count += 1

        return image_rgb, count, ripeness_classes
