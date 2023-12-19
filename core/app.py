import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model


from classifier import FruitClassifier


ripeness_classifier = FruitClassifier()

def main():
    global count
    st.title("YOLOv5 Apple Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1,col2=st.columns(2)

        col1.image(uploaded_file, caption="Uploaded Image.", width=300)

        # Run YOLOv5 detection on the uploaded image
        image_array,count,ripeness_classes = process_image(uploaded_file)

        # Display the result
        col2.image(image_array, caption="Processed Image with Bounding Boxes.", width=300)
        st.title(f"The number of apples are {count}, number of ripe : {ripeness_classes.count('Ripe')}, number of unripe : {ripeness_classes.count('Unripe')} , classes are : {ripeness_classes}")

def process_image(uploaded_file):
    global ripeness_classifier
    count = 0
    # Convert the uploaded file to OpenCV image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(type(image_rgb))

    # Load YOLOv5 model
    bounding_box_detector_model = YOLO('../data/weights/yolov8m15.pt')

    # Ripeness Detector Model
    ripeness_classes = list()

    # Perform YOLOv5 detection
    results = bounding_box_detector_model(image)

    # Draw bounding boxes on the image
    for r in results:
        for box in r.boxes:
            if float(box.conf)>0.3:
                cx,cy,w,h=map(float,box.xywh[0])
                cx, cy, w, h = map(float, box.xywh[0])
                x1 = cx - 0.5 * w
                y1 = cy - 0.5 * h
                x2 = cx + 0.5 * w
                y2 = cy + 0.5 * h

                cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 5)
                cv2.rectangle(image_rgb, (int(x1)+2, int(y1)+2), (int(x2)-2, int(y2)-2), (255, 0, 0), 3)
                
                # Crop the image
                single_apple_image = image_rgb[int(y1):int(y2), int(x1):int(x2)]
                print(type(single_apple_image))
                ripeness_class = ripeness_classifier.predict(img=single_apple_image)
                ripeness_classes.append(ripeness_class)

                count+=1

    return image_rgb,count, ripeness_classes

if __name__ == '__main__':
    main()