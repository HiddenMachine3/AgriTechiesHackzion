import streamlit as st
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from image_processor import AppleProcessor

processor = AppleProcessor()

def create_histogram(ripe_count, unripe_count):
    labels = ['Ripe', 'Unripe']
    counts = [ripe_count, unripe_count]

    plt.bar(labels, counts)
    plt.xlabel('Apple Ripeness')
    plt.ylabel('Count')
    plt.title('Ripe vs Unripe Apple Count')

    return plt

def main():
    global count, create_histogram
    st.title("YOLOv5 Apple Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        col1.image(uploaded_file, caption="Uploaded Image.", width=300)

        # Run YOLOv5 detection on the uploaded image
        image_array, count, ripeness_classes = processor.process_image(uploaded_file)
        ripe_count = ripeness_classes.count("Ripe")
        unripe_count = ripeness_classes.count("Unripe")
        # Display the result
        col2.image(
            image_array, caption="Processed Image with Bounding Boxes.", width=300
        )
        st.title(
            f"The number of apples are {count}, number of ripe : {ripe_count}, number of unripe : {unripe_count} , classes are : {ripeness_classes}"
        )

        st.pyplot(create_histogram(ripe_count, unripe_count))


if __name__ == "__main__":
    main()
