import streamlit as st
from singleinference_yolov7 import SingleInference_YOLOV7
from typing import List, NamedTuple
from PIL import Image
from numpy import asarray
import cv2
import numpy as np
import json
import torch

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():
    
    st.title("Proximity Detection")
    st.sidebar.markdown("Configure proximity detection, detect all persons in close proximity to a machine or vehicle.")
    CONFIG_FILE = './configs/default.json'
    model_from_config = json.load(open(CONFIG_FILE))

    # MODEL_WEIGHTS = "./weights/yolov7-construction-custom.pt"
    MODEL_WEIGHTS = model_from_config['model']
    # MODEL_WEIGHTS = "./weights/yolov7-tiny-construction-custom-v0.pt"
    IMG_SIZE = 640

    DEFAULT_CONFIDENCE_THRESHOLD = 0.25
    DEFAULT_OVERLAP_THRESHOLD = 0.45
    DEFAULT_IMG = './images/machinery-example1.jpeg'

    confidence_threshold = st.slider(
        "Confidence Threshold:", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
    )

    overlap_threshold = st.slider(
        "Overlap Threshold:", 0.0, 1.0, DEFAULT_OVERLAP_THRESHOLD, 0.05
    )

    st.image(DEFAULT_IMG,use_column_width=True)

    image = st.file_uploader("Upload an image (above image will be used as default)", type=["jpg", "jpeg"], key="machine")

    machines = st.checkbox("Machinery", True)
    vehicles = st.checkbox("Vehicle")

    policy = {
        "type" : "proximity_detection",
        "machinery" : machines,
        "vehicle" : vehicles,
    }

    predictions = st.button('Detect')
    st.download_button(
        label="Save",
        data=json.dumps(policy),
        file_name="proximity_policy.json",
        mime="text/plain"
    )

    class Detection(NamedTuple):
        name: list
        coords: list
        
    #@st.experimental_singleton
    def get_detector_model():
        multi_inputdevice = "0" if torch.cuda.is_available() else "cpu"
        model = SingleInference_YOLOV7(img_size=IMG_SIZE, path_yolov7_weights=MODEL_WEIGHTS, device_i=multi_inputdevice, conf_thres=confidence_threshold, iou_thres=overlap_threshold)
        model.load_model()
        return model

    yolov7_detector = get_detector_model() 
    # print(yolov7_detector.conf_thres)	

    if predictions: 
        if image is not None:
        # asarray() class is used to convert
        # PIL images into NumPy arrays
        # load the image and convert into
        # numpy array
            img = Image.open(image)
        else:
            img = Image.open(DEFAULT_IMG)
            
        img = asarray(img)
        # detect_proximity(img)
        color_image = yolov7_detector.detect_proximity(img, machines, vehicles)
        st.image(img, caption='Input Image', use_column_width=True)
        st.image(color_image, caption='Output Image', use_column_width=True) 	
        predictions = False