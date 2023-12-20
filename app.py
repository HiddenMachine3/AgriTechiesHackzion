import streamlit as st
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from core.image_processor import AppleProcessor
import streamlit as st

st.set_option("deprecation.showPyplotGlobalUse", False)


# processor_faraway = AppleProcessor("faraway.pt")
# processor_nearby = AppleProcessor("best.pt")

# nearby = None
# faraway = None

processor = AppleProcessor("farANDnear.pt")


def create_histogram(ripe_count, unripe_count):
    labels = ["Ripe", "Unripe"]
    counts = [ripe_count, unripe_count]

    # plt.bar(labels, counts)
    # plt.xlabel('Apple Ripeness')
    # plt.ylabel('Count')
    # plt.title('Ripe vs Unripe Apple Count')

    # return plt
    fig = px.bar(
        x=labels,
        y=counts,
        labels={"x": "Apple Ripeness", "y": "Count"},
        title="Ripe vs Unripe Apple Count",
    )

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
        unsafe_allow_html=True,
    )

    global create_histogram, nearby, faraway

    total_ripe_count,total_unripe_count, total_ripeness_classes = 0,0,[]

    st.title("Produce Estimation With Computer Vision")

    
    uploaded_files = st.file_uploader(
        "Insert images to estimate produce ", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )


    def show_output_for(image_array, count, ripeness_classes,col1=None,col2=None):
        nonlocal total_ripe_count,total_unripe_count, total_ripeness_classes
        ripe_count = ripeness_classes.count("Ripe")
        unripe_count = ripeness_classes.count("Unripe")
        total_ripe_count += ripe_count
        total_unripe_count += unripe_count
        total_ripeness_classes += ripeness_classes

        if image_array is not None:
        # Display the result
            col2.image(
                image_array, caption="Processed Image with Bounding Boxes.", width=300
            )
            st.title(
                f"The number of apples are {count}"
            )  # \n number of ripe : {ripe_count}\n number of unripe : {unripe_count} "# classes are : {ripeness_classes}"
        st.markdown(
            f"""
    <div style="font-size: 30px; font-weight: bold;">
        Number of ripe apples are {ripe_count}
    </div>
    """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
    <div style="font-size: 30px; font-weight: bold;">
        Number of unripe apples are {unripe_count}
    </div>
    """,
            unsafe_allow_html=True,
        )

        # st.pyplot(create_histogram(ripe_count, unripe_count))
        create_histogram(ripe_count, unripe_count)


    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            col1, col2 = st.columns(2)

            col1.image(uploaded_file, caption="Uploaded Image.", width=300)

            show_output_for(*processor.process_image(uploaded_file),col1, col2)

        #     else:
        #         st.session_state["currentdetector"] = "faraway"
        #         faraway()

        # st.write(st.session_state["currentdetector"])
    
        st.title("Total")
        show_output_for(None,total_ripe_count+total_ripe_count,total_ripeness_classes)

# def main():
   


if __name__ == "__main__":
    main()