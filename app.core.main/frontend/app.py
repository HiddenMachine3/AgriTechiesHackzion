import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO

def main():
    st.title("YOLOv5 Apple Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1,col2=st.columns(2)

        col1.image(uploaded_file, caption="Uploaded Image.", width=300)

        # Run YOLOv5 detection on the uploaded image
        image_array,count = process_image(uploaded_file)

        # Display the result
        col2.image(image_array, caption="Processed Image with Bounding Boxes.", width=300)
    st.title(f"The number of apples are {count}")
def process_image(uploaded_file):
    # Convert the uploaded file to OpenCV image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # # TODO: Run YOLOv5 detection on image and get results
    # # For simplicity, let's assume count is 5 and bounding boxes are [(x1, y1, x2, y2), ...]
    # count = 5
    # bounding_boxes = [(10, 10, 100, 100), (120, 120, 200, 200)]  # Replace with actual results

    # # Draw bounding boxes on the image
    # for box in bounding_boxes:
    #     cv2.rectangle(image_rgb, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # return image_rgb

      # Load YOLOv5 model
    model = YOLO('C:\\Users\\25bak\\OneDrive\\Desktop\\Python\\Python Projects\\AgriTechiesHackzion\\data\\weights\\yolov8m15.pt')

    # Perform YOLOv5 detection
    results = model(image)
    count=0
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

            #     head = image[int(y1):int(y2), int(x1):int(x2)]
            #     head = cv2.cvtColor(head, cv2.COLOR_BGR2RGB)
            # x1, y1, x2, y2 = box[:4]
                cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 5)
                cv2.rectangle(image_rgb, (int(x1)+2, int(y1)+2), (int(x2)-2, int(y2)-2), (255, 0, 0), 3)
                count+=1

    return image_rgb,count

if __name__ == '__main__':
    main()