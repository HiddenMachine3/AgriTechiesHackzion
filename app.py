import streamlit as st
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from core.image_processor import AppleProcessor
st.set_option('deprecation.showPyplotGlobalUse', False)


processor_faraway = AppleProcessor("yolov8m15.pt")
processor_nearby = AppleProcessor("faraway.pt")

def create_histogram(ripe_count, unripe_count):
    labels = ['Ripe', 'Unripe']
    counts = [ripe_count, unripe_count]

    # plt.bar(labels, counts)
    # plt.xlabel('Apple Ripeness')
    # plt.ylabel('Count')
    # plt.title('Ripe vs Unripe Apple Count')

    # return plt
    fig = px.bar(x=labels, y=counts, labels={'x': 'Apple Ripeness', 'y': 'Count'},
                 title='Ripe vs Unripe Apple Count')
    
    st.plotly_chart(fig)

def main():

    st.set_page_config(
        page_title="Produce Estimation With Computer Vision",
        page_icon="🍎",
        layout="wide", 
        initial_sidebar_state="auto",
    )

    st.markdown(
    """
    <style>
        body {
            background-color: #e6f7ff;
            color: #343a40;
        }
        .stApp {
            max-width: 1000px;
            margin: 0 auto;
        }
        .title {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)
    
    global count, create_histogram


    st.title("Produce Estimation With Computer Vision")

    uploaded_file = st.file_uploader("Insert an image to estimate produce ", type=["jpg", "jpeg", "png"])

    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        col1.image(uploaded_file, caption="Uploaded Image.", width=300)

        # image_array1, count1, ripeness_classes1 = processor_faraway.process_image(uploaded_file)
        image_array2, count2, ripeness_classes2 = processor_nearby.process_image(uploaded_file)
        
        # TODO : remove this
        count1 = 0

        # Run YOLOv5 detection on the uploaded image
        image_array, count, ripeness_classes = image_array2, count2, ripeness_classes2 # image_array1, count1, ripeness_classes1 if count1 > count2 else image_array2, count2, ripeness_classes2
        ripe_count = ripeness_classes.count("Ripe")
        unripe_count = ripeness_classes.count("Unripe")
        # Display the result
        col2.image(
            image_array, caption="Processed Image with Bounding Boxes.", width=300
        )
        st.title(
            f"The number of apples are {count}")#\n number of ripe : {ripe_count}\n number of unripe : {unripe_count} "# classes are : {ripeness_classes}"
        st.markdown(
    f"""
    <div style="color: white; font-size: 30px; font-weight: bold;">
        Number of ripe apples are {ripe_count}
    </div>
    """,
    unsafe_allow_html=True
)

        st.markdown(
    f"""
    <div style="color: white; font-size: 30px; font-weight: bold;">
        Number of unripe apples are {unripe_count}
    </div>
    """,
    unsafe_allow_html=True
)


        # st.pyplot(create_histogram(ripe_count, unripe_count))
        create_histogram(ripe_count, unripe_count)


if __name__ == "__main__":
    main()
