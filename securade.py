import argparse
import time
from datetime import date
import subprocess
import os
import json
import safety_app
import requests
import os
import signal
import psutil
import subprocess
import time

from threading import Thread
from queue import Queue

import platform
if platform.machine() == 'x86_64':
    from openvino.runtime import Core

import cv2
import torch
import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np

import telegram
import asyncio
from io import BytesIO
from pytapo import Tapo
from PIL import Image

from requests.auth import HTTPDigestAuth
from numpy import random
from pathlib import Path
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from yolov7face import YOLOv7Face, YOLOv7Configs, FaceAnonymizer

ASCII_LOGO = '''\
--------------------------------------------------------------------------
#    #####                                                          #      
#   #     # ######  ####  #    # #####    ##   #####  ######       # #   # 
#   #       #      #    # #    # #    #  #  #  #    # #           #   #  # 
#    #####  #####  #      #    # #    # #    # #    # #####      #     # # 
#         # #      #      #    # #####  ###### #    # #      ### ####### # 
#   #     # #      #    # #    # #   #  #    # #    # #      ### #     # # 
#    #####  ######  ####   ####  #    # #    # #####  ###### ### #     # # 
--------------------------------------------------------------------------   
'''

def save_image(q):
    while True:
        (img, save_path, mask_faces, face_detector, api_key, chat_id, violation_type) = q.get()
        t = time.localtime()
        current_time = time.strftime("%H-%M-%S", t)
        if save_path is not None:
            save_path_img = str(save_path) + "/" + current_time + '.jpg'
            if mask_faces:
                img = face_detector.predict_img(img)
            cv2.imwrite(save_path_img, img)
        if chat_id != '':
            caption = '\nViolation Type: ' + violation_type + '\n\nDate: ' + date.today().strftime("%B %d, %Y") + '\nTime: ' + time.strftime("%H:%M:%S", t)
            try:
                asyncio.run(send_notification(api_key, chat_id, img, caption))
            except Exception as e:
                print("Error while sending telegram message, but processing will continue.")
            
async def send_notification(api_id, chat_id, img, caption):
    bot = telegram.Bot(api_id)
    async with bot: 
        success, encoded_image = cv2.imencode('.jpg', img)
        # Write image to buffer
        bbuffer = BytesIO()
        if success:
            # Convert the encoded image to bytes
            bytes_data = encoded_image.tobytes()
            # Write the bytes data to the buffer
            bbuffer.write(bytes_data)
            bytes_data = bbuffer.getvalue()
        await bot.send_photo(photo=bytes_data, caption='Safety Violation Detected\n' + caption, chat_id=chat_id)

def process():
    t0 = time.time()
    config = json.load(open(opt.config))
    device = select_device()
    half = device.type != 'cpu'  # half precision only supported on CUDA
    save_dir = Path(os.getcwd()+'/output/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if config is not None:
        
        camera_list = config['sources']
        img_size = config['img_size']
        img_augment = config['img_aug']
        weights = config['model']
        conf_thres = config['conf_thres']
        iou_thres = config['iou_thres']
        show_cameras = config['show_cameras']
        mask_faces =config['mask_faces']
        api_key = config['api_key']
        chat_id = config['chat_id']
         
        if opt.cpu and not half:
            
            core = Core()
            # read converted model
            model_path = weights.replace(".pt", "_int8.xml")
            open_vino_model = core.read_model(model_path)
            # load model on CPU device
            compiled_model = core.compile_model(open_vino_model, 'CPU')
            model = compiled_model
                    
        else:
            model = attempt_load(weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(img_size, s=stride)  # check img_size
        
        if half:
            model.half()  # to FP16
        
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        if mask_faces:
            face_model_configs = YOLOv7Configs(weights='modelzoo/face.pt')
            face_anonymizer = FaceAnonymizer(method='blur', blur_kernel_size=(45, 45), blur_sigma_x=45, blur_sigma_y=45)
            # face_anonymizer = FaceAnonymizer(method='block', block_intensity=255)
            face_detector = YOLOv7Face(configs=face_model_configs, anonymizer=face_anonymizer)
        else:
            face_detector = None
            
        if not weights.endswith('safety.pt'):
            map_file = weights.replace(".pt", "_map.json")
            model_object_map = json.load(open(map_file))
        else:
            model_object_map = None
            
        source = []
        policy = []
        duration = []
        timers = []
        alerts = []
        last_activity_time = []
        
        
        for camera in camera_list:
            source.append(camera['url'])
            policy.append(camera['policy_file'])
            duration.append(camera['duration'])
            if 'alert_url' in camera:
                alert_url = camera['alert_url']
                if alert_url !=  "" and "axis-cgi" not in alert_url: 
                    # These settings are for using the Tapo camera itself as an alert strobe 
                    user = 'admin' # user you set in Advanced Settings -> Camera Account
                    password = 'pass' # This seems to be using the Tapo account email address/password instead of RTSP.
                    host = alert_url # ip of the camera, example: 192.168.1.52
                    tapo = Tapo(host, user, password)  
                    alerts.append(tapo)
                else:
                    alerts.append(alert_url)
            else:
                alerts.append("")
            timers.append(time.time())
            last_activity_time.append(time.time())
        # print(source)
        policy_data = []
        # print(policy)
        for policy_file in policy:
            f = open(policy_file)
            policy_data.append(json.load(f))
            # print(policy_data)
            f.close()
        if opt.cpu and not half:
            dataset = LoadStreams(source, use_open_vino=True)
        else:
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        # Get names and colors
        if opt.cpu and not half:
            names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
        else:
            names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            old_img_w = old_img_h = imgsz
            old_img_b = 1

        # t0 = time.time()
        q = Queue()
        save_thread = Thread(target=save_image, args=[(q)], daemon=True)
        save_thread.start()
        
        alarm = False
        
        for path, img, im0s, vid_cap in dataset:
            
            t_begin = time_synchronized()

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=img_augment)[0]

            # Inference
            # t1 = time_synchronized()
            if opt.cpu and not half:
                outputs = model.output(0)
                # print(img.size())
                pred = torch.from_numpy(model(img)[outputs])
            else:
                with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                    # print(img.shape)
                    pred = model(img, augment=img_augment)[0]
            # t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
            
            #t3 = time_synchronized()
            # print(t_begin-last_detection_time)
                
            # Process detections
            for i, det in enumerate(pred):  # detections per image

                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                data = policy_data[i]
                flag_for_time = duration[i]
                start_time = timers[i]
                alert_url = alerts[i]
                # print(p)
                p = Path(p)  # to Path
                save_path = save_dir / p / Path(str(date.today()))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                
                flag = False
                
                if opt.no_activity_alert is not None:
                    if t_begin - last_activity_time[i] > opt.no_activity_alert:
                        #numpy_array = orig_img.squeeze()
                        #notify_img = np.transpose(numpy_array, (1, 2, 0)) 
                        notification_caption = 'no_activity_detection' + '\nSource: ' + source[i]   
                        q.put((im0, save_path, mask_faces, face_detector, api_key, chat_id, notification_caption))
                        last_activity_time[i] = time.time()
                
                if len(det):
                    last_activity_time[i] = time.time()
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    box_list = []
                    for *xyxy, conf, cls in reversed(det):
                        xyxy_list = (torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                        x0,y0,x1,y1 = xyxy_list[0],  xyxy_list[1],  xyxy_list[2],  xyxy_list[3]
                        label = f'{names[int(cls)]}'
                        conf_val = f'{conf:.2f}'
                        #print(val)
                        if model_object_map is not None:
                            label = model_object_map[label]
                        box_list.append([label,x0,y0,x1,y1,conf_val])
                        #print(label)
                        
                        '''
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        
                        
                        if view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                        '''
                        
                    violation_type = data['type']
                    
                    if violation_type == 'ppe_detection':
                        hardhats, vests, masks = data['hardhats'], data['vests'], data['masks']
                        no_hardhats, no_vests, no_masks = data['no_hardhats'], data['no_vests'], data['no_masks']
                        flag = safety_app.detect_ppe(im0, box_list, hardhats, vests, masks, no_hardhats, no_vests, no_masks)
                        # print(flag)
                    elif violation_type == 'proximity_detection':
                        machinery, vehicles = data['machinery'], data['vehicle']
                        flag = safety_app.detect_proximity(im0, box_list, machinery, vehicles)
                    elif violation_type == 'exclusion_zones':
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
                                        x0 = x0/600*im0.shape[1] 
                                        y0 = y0/400*im0.shape[0]
                                        poly.append([x0, y0])
                        # print(poly)
                        flag = safety_app.detect_zone(im0, box_list, poly, persons, machinery, vehicles, inclusion, max_number_allowed)
                    elif violation_type == 'ppe_detection_exclusion_zones':
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
                                        x0 = x0/600*im0.shape[1] 
                                        y0 = y0/400*im0.shape[0]
                                        poly.append([x0, y0])
                        flag_zone = safety_app.detect_zone(im0, box_list, poly, persons, machinery, vehicles, inclusion, max_number_allowed)
                        if not inclusion:
                            if flag_zone:
                                flag_ppe = safety_app.detect_ppe(im0, box_list, hardhats, vests, masks, no_hardhats, no_vests, no_masks)
                            else:
                                flag_ppe = False
                            flag = flag_ppe and flag_zone
                        else:
                            if flag_zone:
                                flag = True
                            else:
                                flag = safety_app.detect_ppe(im0, box_list, hardhats, vests, masks, no_hardhats, no_vests, no_masks)
                    else:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                        flag = True
                 
                notification_caption = data['type'] + '\nSource: ' + source[i]
                    
                if flag:
                    if time.time() - start_time > flag_for_time: 
                        q.put((im0, save_path, mask_faces, face_detector, api_key, chat_id, notification_caption))
                        timers[i] = time.time()
                        if type(alert_url) != str and alarm is False:
                            # print("Start Alarm")
                            alert_url.startManualAlarm()
                            alarm = True
                                # tapo.setAlarm(enabled=True, soundEnabled=False, lightEnabled=True)
                            # tapo.startManualAlarm()  
                        elif type(alert_url) == str and alert_url != '':
                            # alert the strobe light with profile red
                            start_profile = '{"apiVersion" : "1.0", "context" : "my context", "method": "start", "params" : {"profile" : "red"}}'
                            response = requests.post(alert_url, json=json.loads(start_profile), auth=HTTPDigestAuth('root', 'pass'))
                            # print(response.text)
                else:
                    timers[i] = time.time()
                    if type(alert_url) != str and alarm is True:
                        # print("Stop Alarm")
                        alert_url.stopManualAlarm()
                        alarm = False
                        # alert_url.stopManualAlarm() 
                        #print(f" The image with the result is saved in: {save_path}")
                # Print time (inference + NMS)
                # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
                t_end = time_synchronized()
                fps = 1/(t_end-t_begin)
                # Stream results
                if view_img and show_cameras:
                    fps_text="FPS:{:.2f}".format(fps)
                    cv2.putText(im0, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)
                    #define the screen resolution 1024, 768
                    screen_res = 800, 600
                    scale_width = screen_res[0] / im0.shape[1]
                    scale_height = screen_res[1] / im0.shape[0]
                    scale = min(scale_width, scale_height)
                    #resized window width and height
                    window_width = int(im0.shape[1] * scale)
                    window_height = int(im0.shape[0] * scale)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(str(p), window_width, window_height)
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond
                    
                # Save results (image with detections)
                '''
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)
                        '''
        
        q.join()
        #if save_txt or save_img:
        #    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            #print(f"Results saved to {save_dir}{s}")

    # print(sources)
    print(f'Done. ({time.time() - t0:.3f}s)')


def kill_process_tree(pid):
    """Kill a process and all its children."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        # First send SIGTERM to all children
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        
        # Then terminate the parent
        parent.terminate()
        
        # Wait for processes to terminate gracefully
        gone, alive = psutil.wait_procs(children + [parent], timeout=3)
        
        # If any processes are still alive, force kill them
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
                
    except psutil.NoSuchProcess:
        pass

def find_streamlit_processes(target_script_path):
    """
    Find Streamlit processes specifically running our target script.
    
    Args:
        target_script_path (str): The full path to our Configure_Camera.py script
    """
    streamlit_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline'] or []
            # Check if this is a streamlit process running our specific script
            if (len(cmdline) >= 3 and 
                'streamlit' in str(cmdline[0]).lower() and 
                'run' in str(cmdline[1]).lower() and 
                target_script_path in str(cmdline[2])):
                streamlit_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return streamlit_processes

def run_streamlit_server():
    """Run the Streamlit server with proper cleanup on shutdown."""
    # Get the full path to our script
    script_path = os.path.join(os.getcwd(), os.path.dirname(__file__), 'Configure_Camera.py')
    try:
        # Start the Streamlit process
        process = subprocess.Popen(
            ['streamlit', 'run', script_path],
            start_new_session=True  # This creates a new process group
        )
        
        print("Server started. Press Enter to shutdown...")
        input()
        
    except KeyboardInterrupt:
        print("\nReceived shutdown signal...")
    finally:
        print("Shutting down server...")
        
        # Kill the main process tree
        kill_process_tree(process.pid)
        
        # Find and kill any remaining Streamlit processes
        for proc in find_streamlit_processes(script_path):
            kill_process_tree(proc.pid)
        
        # Wait a moment to ensure all processes are cleaned up
        time.sleep(1)
        
        # Verify no Streamlit processes are left
        remaining = find_streamlit_processes(script_path)
        if remaining:
            print(f"Warning: {len(remaining)} Streamlit processes still running")
            for proc in remaining:
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
        
        print("Server shutdown complete")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Load config from file')
    parser.add_argument('--cpu', action='store_true', help='Use the OpenVINO Runtime for inference on CPU')
    parser.add_argument('--no_activity_alert', type=int, help='Time in seconds after which a no activity alert is raised')
    parser.add_argument('--server', action='store_true', help='Run the Securade web server application')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    opt = parser.parse_args()
    
    print(ASCII_LOGO)
    #print(opt)
    
    if opt.config is None and opt.server is False:
        print('No config file specified, please supply a [config.json] file via --config flag.')
    elif opt.server:
        run_streamlit_server()
    elif opt.config is not None:
        process()
