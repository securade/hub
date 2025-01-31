import streamlit as st
import av
import queue
import torch
import cv2
from typing import List, NamedTuple
from singleinference_yolov7 import SingleInference_YOLOV7

import platform
if platform.machine() == 'x86_64':
    from openvino.runtime import Core

from datetime import datetime
import random

import linecache
import os
import tracemalloc, json
import gc
import pandas as pd
from io import BytesIO
import numpy as np
import requests
import socket


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # doesn't even have to be reachable
        s.connect(('10.254.254.254', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

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

    # Streamlit Components
    st.set_page_config(
        page_title="Securade.ai - HUB",
        page_icon="https://securade.ai/favicon.ico",
        layout="centered",
        # initial_sidebar_state="expanded",
        menu_items={
            "About": "# Securade HUB v0.1\n [https://securade.ai](https://securade.ai)",
            "Get Help" : "https://securade.ai",
            "Report a Bug": "mailto:hello@securade.ai"
        },
    )

    st.title("Configure Camera")
    st.sidebar.markdown("View the live feed and setup safety policy.")
    st.sidebar.markdown("[Dashboard](http://"+ get_ip()+":8888)")
    
    IMG_SIZE = 640
    CONFIG_FILE = './configs/default.json'
    CPU = False
    
    model_from_config = json.load(open(CONFIG_FILE))
    
    DEFAULT_CONFIDENCE_THRESHOLD = model_from_config['conf_thres']
    DEFAULT_OVERLAP_THRESHOLD = model_from_config['iou_thres']
    
    confidence_threshold = st.slider(
        "Confidence Threshold:", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
    )

    overlap_threshold = st.slider(
        "Overlap Threshold:", 0.0, 1.0, DEFAULT_OVERLAP_THRESHOLD, 0.05
    )

    # MODEL_WEIGHTS = "./weights/yolov7-construction-custom.pt"
    MODEL_WEIGHTS = model_from_config['model']
    # './modelzoo/safety.pt'
    # MODEL_WEIGHTS = "./weights/securade-safety-model-E001.pt"


    class Detection(NamedTuple):
        name: str
        prob: float
        
    def get_detector_model():
        multi_inputdevice = "0" if torch.cuda.is_available() else "cpu"
        if multi_inputdevice == "cpu" and CPU:
            core = Core()
            # read converted model
            open_vino_model = core.read_model('./modelzoo/safety.xml')
            # load model on CPU device
            compiled_model = core.compile_model(open_vino_model, 'CPU')
            model = SingleInference_YOLOV7(img_size=IMG_SIZE, path_yolov7_weights = None, device_i=multi_inputdevice,
                                    conf_thres=confidence_threshold,
                                    iou_thres=overlap_threshold)
            model.open_vino_model = compiled_model
            model.names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
            model.colors = [[random.randint(0, 255) for _ in range(3)] for _ in model.names]
        else:
            model = SingleInference_YOLOV7(img_size=IMG_SIZE, path_yolov7_weights=MODEL_WEIGHTS, device_i=multi_inputdevice,
                                    conf_thres=confidence_threshold,
                                    iou_thres=overlap_threshold)
            model.load_model()
        return model

    # Session-specific caching
    cache_key = "object_detection_yolo" 
    if cache_key in st.session_state:
        current_model = st.session_state[cache_key]
        if confidence_threshold != current_model.conf_thres or overlap_threshold != current_model.iou_thres:
            yolov7_detector = get_detector_model()
            st.session_state[cache_key] = yolov7_detector
        else:
            yolov7_detector = current_model
    else:
        # Initialize yolov7 object detector
        yolov7_detector = get_detector_model()
        st.session_state[cache_key] = yolov7_detector

    data = None
    policy_json = st.file_uploader("Upload the safety policy file (e.g. ppe_policy.json)", type=["json"], key="policy")
    save_frame = st.button("Save Frame")
    update_source = st.button("Update Source")
    st.write('_Model currently in use is ' + MODEL_WEIGHTS + ', you can preview it below._')
    camera_id = st.text_input('Enter the camera id or the full path to the video file', './videos/video_2.mp4')
    if camera_id.isnumeric():
        input = int(camera_id)
    else:
        input = camera_id
    run = st.checkbox('Run', key="run_box")
    labels = st.checkbox("Show the detected labels", value=False, key="show_labels")

    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(input)
    labels_placeholder = st.empty()
            
    if policy_json is not None:
        data = json.load(policy_json)
    else: data = None
    n = 0
    
    while run:
        ret, frame = camera.read()  
        if ret:
            n += 1
            if n == 4: # read every 4th frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                screen_res = 800, 600
                scale_width = screen_res[0] / frame.shape[1]
                scale_height = screen_res[1] / frame.shape[0]
                scale = min(scale_width, scale_height)
                 #resized window width and height
                window_width = int(frame.shape[1] * scale)
                window_height = int(frame.shape[0] * scale)
                image = cv2.resize(frame, (window_width, window_height))
                
                # image = frame
                if data is not None:
                    if data['type'] == 'ppe_detection':
                        hardhats, vests, masks = data['hardhats'], data['vests'], data['masks']
                        no_hardhats, no_vests, no_masks = data['no_hardhats'], data['no_vests'], data['no_masks']
                        annotated_image = yolov7_detector.detect_ppe(image, hardhats, vests, masks, no_hardhats, no_vests, no_masks)
                    elif data['type'] == 'proximity_detection':
                        machinery, vehicles = data['machinery'], data['vehicle']
                        annotated_image = yolov7_detector.detect_proximity(image, machinery, vehicles)
                    elif data['type'] == 'exclusion_zones':
                        persons, machinery, vehicles, max_number_allowed = data['persons'], data['machinery'], data['vehicle'], data['max_allowed']
                        inclusion = data['inclusion_zone']
                        json_data = data['zones']
                        poly = []
                        df = pd.json_normalize(json_data["objects"]) # need to convert obj to str because PyArrow
                        if not df.empty:
                            paths = df["path"].tolist()
                            if isinstance(paths[0], list):
                                for pt in paths[0]:
                                    if pt[0] != 'z':
                                        x0, y0 = pt[1], pt[2]
                                        x0 = x0/600*image.shape[1] 
                                        y0 = y0/400*image.shape[0]
                                        poly.append([x0, y0])
                        annotated_image = yolov7_detector.detect_zone(image, poly, persons, machinery, vehicles, inclusion, max_number_allowed)
                    elif data['type'] == 'ppe_detection_exclusion_zones':
                        hardhats, vests, masks = data['hardhats'], data['vests'], data['masks']
                        no_hardhats, no_vests, no_masks = data['no_hardhats'], data['no_vests'], data['no_masks']
                        persons, machinery, vehicles, max_number_allowed = data['persons'], data['machinery'], data['vehicle'], data['max_allowed']
                        inclusion = data['inclusion_zone']
                        json_data = data['zones']
                        poly = []
                        df = pd.json_normalize(json_data["objects"]) # need to convert obj to str because PyArrow
                        if not df.empty:
                            paths = df["path"].tolist()
                            if isinstance(paths[0], list):
                                for pt in paths[0]:
                                    if pt[0] != 'z':
                                        x0, y0 = pt[1], pt[2]
                                        x0 = x0/600*image.shape[1] 
                                        y0 = y0/400*image.shape[0]
                                        poly.append([x0, y0])
                        zone_image = yolov7_detector.detect_zone(image, poly, persons, machinery, vehicles, inclusion, max_number_allowed)
                        annotated_image = yolov7_detector.detect_ppe(zone_image, hardhats, vests, masks, no_hardhats, no_vests, no_masks) 
                else:
                    yolov7_detector.load_cv2mat(image)
                    yolov7_detector.inference()
                    annotated_image = yolov7_detector.image.copy()
                
                FRAME_WINDOW.image(annotated_image)
                
                result: List[Detection] = []
                img = None
                if save_frame:
                    img = frame
                    
                if len(yolov7_detector.predicted_bboxes_PascalVOC)>0:
                        for item in yolov7_detector.predicted_bboxes_PascalVOC:
                            name = str(item[0])
                            prob = str(round(100*item[-1],2))
                            result.append(Detection(name=name, prob=float(prob)))
                
                if labels and result != []:
                    labels_placeholder.table(result)
                
                if save_frame:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    _, encoded_image = cv2.imencode('.jpg', img_rgb)
                    byte_im = encoded_image.tobytes()
                    st.image(img, caption='Saved Frame', use_column_width=True)
                    st.download_button('Download Frame', byte_im, file_name="frame.jpg",) 
                    save_frame = False
                
                if update_source:
                    config = json.load(open(CONFIG_FILE))
                    camera_list = config['sources']
                    img_size = config['img_size']
                    img_augment = config['img_aug']
                    weights = config['model']
                    conf_thres = config['conf_thres']
                    iou_thres = config['iou_thres']
                    show_cameras = config['show_cameras']
                    mask_faces = config['mask_faces']
                    api_key = config['api_key']
                    chat_id = config['chat_id']
                    url = camera_id
                    now = datetime.now()
                    policy_file_name = 'policy_' + now.strftime("%d_%m_%Y_%H_%M_%S") + '.json' 
                    dict = {
                        'url' : url,
                        'policy_file' : policy_file_name,
                        'duration' : 10,
                        'alert_url' : ""
                    }
                    
                    with open(policy_file_name, 'w') as outfile:
                        json.dump(data, outfile, indent=4)
                    
                    not_found = True
                    for source in camera_list:
                        if camera_id == source['url'] :
                            source['policy_file'] = policy_file_name
                            not_found = False
                    if not_found:
                        camera_list.append(dict)
                    
                    config['sources'] = camera_list
                    
                    with open(CONFIG_FILE, 'w') as outfile:
                        json.dump(config, outfile, indent=4)
                        st.sidebar.text("Settings saved successfully.")
                    update_source = False

                n = 0
        else:
            camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
