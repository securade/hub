import streamlit as st
from singleinference_yolov7 import SingleInference_YOLOV7
from typing import List, NamedTuple
from PIL import Image
from numpy import asarray
import cv2
import pandas as pd
import numpy as np
from streamlit_drawable_canvas import st_canvas
from svgpathtools import parse_path
from shapely.geometry import Polygon
from shapely.geometry import box
import functools
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
    
    st.title("Exclusion Zones")
    st.sidebar.markdown("Configure exclusion zones, detect all persons, machines or vehicles in an exclusion zone.")
    CONFIG_FILE = './configs/default.json'
    model_from_config = json.load(open(CONFIG_FILE))

    # MODEL_WEIGHTS = "./weights/yolov7-construction-custom.pt"
    MODEL_WEIGHTS = model_from_config['model']
    # MODEL_WEIGHTS = "./weights/yolov7-construction-custom.pt"
    # MODEL_WEIGHTS = "./weights/yolov7-tiny-construction-custom-v0.pt"
    IMG_SIZE = 640

    DEFAULT_CONFIDENCE_THRESHOLD = 0.25
    DEFAULT_OVERLAP_THRESHOLD = 0.45
    DEFAULT_IMG = './images/zone-example1.jpeg'

    confidence_threshold = st.slider(
        "Confidence Threshold:", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
    )

    overlap_threshold = st.slider(
        "Overlap Threshold:", 0.0, 1.0, DEFAULT_OVERLAP_THRESHOLD, 0.05
    )

    st.image(DEFAULT_IMG,use_column_width=True)

    image = st.file_uploader("Upload an image (above image will be used as default)", type=["jpg", "jpeg"], key="exclusion")

    persons = st.checkbox("Persons", True, key="zones1")
    machines = st.checkbox("Machinery", key="zones2")
    vehicles = st.checkbox("Vehicle", key="zones3")
    max_number_allowed = st.number_input("Maximum allowed:", 0, 8, 0, 1)
    inclusion = st.checkbox("Treat as inclusion zone", key="zones4")

    st.write('Draw the exclusion zone as a polygon below:')
    
    if image is not None:
    # asarray() class is used to convert
    # PIL images into NumPy arrays
    # load the image and convert into
    # numpy array
        img = Image.open(image)
    else:
        img = Image.open(DEFAULT_IMG)
    # print(img.height,img.width)
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.25)",  # Fixed fill color with some opacity
        stroke_width=2,
        stroke_color="rgba(255, 0, 0)",
        background_image=img,
        #height=img.height,
        #width=img.width,
        update_streamlit=False,
        drawing_mode="polygon",
        display_toolbar=True
    )

    predictions = st.button('Detect')

    policy = {
        "type" : "exclusion_zones",
        "persons" : persons,
        "machinery" : machines,
        "vehicle" : vehicles,
        "inclusion_zone" : inclusion, 
        "max_allowed" : max_number_allowed,
        "zones" : canvas_result.json_data
    }

    st.download_button(
        label="Save",
        data=json.dumps(policy),
        file_name="exclusion_policy.json",
        mime="text/plain"
    )

    #if canvas_result.image_data is not None:
    #    st.image(canvas_result.image_data, caption='Exclusion Zone')
    poly = []

    #print(canvas_result)

    if canvas_result.json_data is not None:
        #objects = pd.json_normalize(canvas_result.json_data["objects"])
        #for col in objects.select_dtypes(include=["object"]).columns:
        #    objects[col] = objects[col].astype("str")
        # st.dataframe(objects)
        df = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
        #print(img.width)
        #print(img.height)
        #print(df["width"][0])
        #print(df["height"][0])
        if not df.empty:
            paths = df["path"].tolist()
            # print(df["scaleX"], df["scaleY"])
            #print(paths[0])
            if isinstance(paths[0], list):
                for pt in paths[0]:
                    #print(pt)
                    if pt[0] != 'z':
                        x0, y0 = pt[1], pt[2]
                        x0 = x0/600*img.width
                        y0 = y0/400*img.height
                        poly.append([x0, y0])
                #print(poly)

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
        
    if predictions: 
        img = asarray(img)
        # detect_zone(img=img)
        st.image(img, caption='Input Image', use_column_width=True)
        color_image = yolov7_detector.detect_zone(img, poly, persons, machines, vehicles, inclusion, max_number_allowed)
        st.image(color_image, caption='Output Image', use_column_width=True) 	
        predictions = False   
