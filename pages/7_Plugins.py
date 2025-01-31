# 7_Plugins.py
import streamlit as st
from plugins.plugin_manager import PluginManager
from PIL import Image
from numpy import asarray
import cv2
import numpy as np
import torch
import json
from singleinference_yolov7 import SingleInference_YOLOV7

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
    st.title("Image Processing Plugins")
    st.sidebar.markdown("Apply various image processing plugins to uploaded images.")

    # Model parameters
    IMG_SIZE = 640
    DEFAULT_CONFIDENCE_THRESHOLD = 0.25
    DEFAULT_OVERLAP_THRESHOLD = 0.45
    MODEL_WEIGHTS = "./modelzoo/safety.pt"  # Using tiny model for faster processing

    # Initialize YOLO detector
    def get_detector_model(conf_thres, iou_thres):
        multi_inputdevice = "0" if torch.cuda.is_available() else "cpu"
        model = SingleInference_YOLOV7(
            img_size=IMG_SIZE,
            path_yolov7_weights=MODEL_WEIGHTS,
            device_i=multi_inputdevice,
            conf_thres=conf_thres,
            iou_thres=iou_thres
        )
        model.load_model()
        return model

    # Initialize plugin manager
    plugin_manager = PluginManager()
    plugin_manager.load_plugins()

    # Get list of plugin options
    plugin_options = list(plugin_manager.get_plugin_choices().keys())
    
    # Find the index of yolo_detector
    default_index = plugin_options.index("yolo_detector") if "yolo_detector" in plugin_options else 0

    # File uploader
    image = st.file_uploader("Upload an image", 
                           type=["jpg", "jpeg", "png"], 
                           key="plugin_image")

    # Plugin selector
    selected_plugin = st.selectbox(
        "Select Processing Plugin:",
        options=plugin_options,
        format_func=lambda x: plugin_manager.get_plugin_choices()[x],
        index=default_index,
        key="plugin_selector"
    )

    # Initialize or update YOLO model with default parameters first
    cache_key = "plugin_yolo_detector"
    if cache_key not in st.session_state:
        yolov7_detector = get_detector_model(DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_OVERLAP_THRESHOLD)
        st.session_state[cache_key] = yolov7_detector

    # Get any custom parameters based on selected plugin
    def get_plugin_params(selected_plugin):
        # Base parameters - YOLO detector always included
        plugin_kwargs = {'yolo_detector': st.session_state[cache_key]}
        
        # Update YOLO parameters if it's the selected plugin
        if selected_plugin == "yolo_detector":
            st.subheader("YOLO Detection Parameters")
            col1, col2 = st.columns(2)
            with col1:
                confidence_threshold = st.slider(
                    "Confidence Threshold:", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
                )
            with col2:
                overlap_threshold = st.slider(
                    "Overlap Threshold:", 0.0, 1.0, DEFAULT_OVERLAP_THRESHOLD, 0.05
                )
            
            # Update YOLO model if parameters changed
            current_model = st.session_state[cache_key]
            if confidence_threshold != current_model.conf_thres or overlap_threshold != current_model.iou_thres:
                yolov7_detector = get_detector_model(confidence_threshold, overlap_threshold)
                st.session_state[cache_key] = yolov7_detector
                plugin_kwargs['yolo_detector'] = yolov7_detector
        
        # Add plugin-specific parameters
        elif selected_plugin == "edge_detector":
            st.subheader("Edge Detection Parameters")
            col1, col2 = st.columns(2)
            with col1:
                plugin_kwargs['threshold1'] = st.slider("Threshold 1:", 0, 255, 100)
            with col2:
                plugin_kwargs['threshold2'] = st.slider("Threshold 2:", 0, 255, 200)

        elif selected_plugin == "safety_transform":
            st.subheader("Safety Gear Transform Parameters")
            col1, col2 = st.columns(2)
            with col1:
                plugin_kwargs['num_inference_steps'] = st.slider(
                    "Number of Inference Steps:", 
                    min_value=10, 
                    max_value=50, 
                    value=20, 
                    help="Higher values give better quality but take longer"
                )
            with col2:
                plugin_kwargs['guidance_scale'] = st.slider(
                    "Guidance Scale:", 
                    min_value=2.0, 
                    max_value=10.0, 
                    step = 0.1,
                    value=8.0, 
                    help="Higher values give better adherence to prompt"
                )

        return plugin_kwargs

    # Get plugin-specific parameters
    plugin_kwargs = get_plugin_params(selected_plugin)

    # Process button
    process = st.button('Process Image')

    if process:
        if image is None:
            st.error("Please upload an image first.")
        else:
            # Open and process the image
            img = Image.open(image)
            img_array = asarray(img)
            
            # Run selected plugin
            try:
                processed_image = plugin_manager.run_plugin(selected_plugin, img_array, **plugin_kwargs)
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img_array, caption='Input Image')
                with col2:
                    st.image(processed_image, caption='Processed Image')
                    
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.error("Stack trace:", stack_info=True)