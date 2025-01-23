import streamlit as st
import json
import os
import subprocess
import datetime
import psutil
import schedule
import threading
import time

CONFIG_FILE = './configs/default.json'
LOG_FILE = 'logs.txt'
PID_FILE = 'process.pid'

@st.cache_resource
def run_continuously(interval=1):
    """Continuously run, while executing pending jobs at each
    elapsed time interval.
    @return cease_continuous_run: threading. Event which can
    be set to cease continuous run. Please note that it is
    *intended behavior that run_continuously() does not run
    missed jobs*. For example, if you've registered a job that
    should run every minute and you set a continuous run
    interval of one hour then your job won't be run 60 times
    at each interval but only once.
    """
    cease_continuous_run = threading.Event()

    class ScheduleThread(threading.Thread):
        @classmethod
        def run(cls):
            while not cease_continuous_run.is_set():
                schedule.run_pending()
                time.sleep(interval)

    continuous_thread = ScheduleThread()
    continuous_thread.start()
    return cease_continuous_run

def start_server(cpu):
    if not check_server():
        if cpu:
            process = subprocess.Popen(['python', 'securade.py', '--config', CONFIG_FILE, '--cpu'])
        else: 
            process = subprocess.Popen(['python', 'securade.py', '--config', CONFIG_FILE])
        # Write PID file
        with open(PID_FILE, 'w') as outfile:
            outfile.write(str(process.pid))
            st.sidebar.text("Server started.")
    else:
        st.sidebar.text("Server is already running.")

def stop_server():
    if check_server():
        with open(PID_FILE, 'r') as pfile:
            pid = int(pfile.read())
            if psutil.pid_exists(pid):
                p = psutil.Process(pid)
                p.terminate() 
                st.sidebar.text("Server stopped.")
    else:
        st.sidebar.text("Server is not running.")

def check_server():
    with open(PID_FILE, 'r') as pfile:
        pid = int(pfile.read())
        return True if psutil.pid_exists(pid) else False

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
    
    # Session-specific caching
    cache_key = "scheduler" 
    if cache_key in st.session_state:  
        stop_run_continuously = st.session_state[cache_key]
    else:
        # Initialize scheduler
        stop_run_continuously = run_continuously()
        st.session_state[cache_key] = stop_run_continuously
    
    st.title("Settings")
    st.sidebar.markdown("Configure the settings for the application.")
    start = st.sidebar.button('Start Server', key='start_server', disabled=check_server())
    stop = st.sidebar.button('Stop Server', key='stop_server', disabled=(not check_server()))
    
    config = json.load(open(CONFIG_FILE))
    
    if config is not None:
        camera_list = config['sources']
        img_size = config['img_size']
        img_augment = config['img_aug']
        model = config['model']
        conf_thres = config['conf_thres']
        iou_thres = config['iou_thres']
        show_cameras = config['show_cameras']
        mask_faces = config['mask_faces']
        api_key = config['api_key']
        chat_id = config['chat_id']

        
        st.divider() 
        st.subheader('Inputs')
        new_model = st.text_input('Model:', model)
        map_file = new_model.replace(".pt", "_map.json")
        object_map = json.load(open(map_file))
        new_object_map = st.experimental_data_editor(object_map)
        save_object_map = st.button("Update object map")
        if save_object_map:
            with open(map_file, 'w') as outfile:
                json.dump(new_object_map, outfile, indent=4)
                st.sidebar.text("Object map updated successfully.")
        new_img_size = st.number_input('Image size:' , img_size)
        new_img_augment = st.checkbox('Augment images', value=img_augment)
        new_conf_thres = st.slider('Confidence threshold:', 0.0, 1.0, conf_thres, 0.01)
        new_iou_thres = st.slider('Overlap threshold:', 0.0, 1.0, iou_thres, 0.01)
        st.divider() 
        
        st.subheader('Outputs') 
        new_show_cameras = st.checkbox('Show camera', value=show_cameras)
        new_mask_faces = st.checkbox('Mask faces' , value=mask_faces)
        st.divider() 
        
        st.subheader('Notifications')
        new_api_key = st.text_input('@SecuradeBot API key:' , api_key)
        new_chat_id = st.text_input('Telegram group id:' , chat_id)
        st.divider()
        
        st.subheader('Sources')
        new_camera_list = st.experimental_data_editor(camera_list)
        delete_source_url = st.text_input('Url to delete: ')
        st.divider() 
        save_settings = st.button("Save")
    
    st.divider()     
    st.subheader('Server')
    enable_cpu = st.checkbox('CPU Inference' , key='enable_cpu', value=False)
    enable_scheduler = st.checkbox('Enable Scheduler' , key='enable_scheduler', value=False)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["**Sunday**", "**Monday**", "**Tuesday**", "**Wednesday**", 
                                                        "**Thursday**", "**Friday**","**Saturday**" ])
    
    with tab1:
        sun_disable = st.checkbox('Disable', value=True, key='sun_disable')
        sun_start = st.time_input('Start:' , key='sun_start', disabled=sun_disable)
        sun_end = st.time_input('End:', key='sun_end', disabled=sun_disable)
    
    with tab2:
        mon_disable = st.checkbox('Disable', value=False, key='mon_disable')
        mon_start = st.time_input('Start:', datetime.time(9, 00), key='mon_start', disabled=mon_disable)
        mon_end = st.time_input('End:', datetime.time(17, 00), key='mon_end', disabled=mon_disable)

    with tab3:
        tue_disable = st.checkbox('Disable', value=False, key='tue_disable')
        tue_start = st.time_input('Start:', datetime.time(9, 00), key='tue_start', disabled=tue_disable)            
        tue_end = st.time_input('End:', datetime.time(17, 00), key = 'tue_end', disabled=tue_disable)
        
    with tab4:
        wed_disable = st.checkbox('Disable', value=False, key='wed_disable')
        wed_start = st.time_input('Start:', datetime.time(9, 00), key= 'wed_start', disabled=wed_disable)
        wed_end = st.time_input('End:', datetime.time(17, 00), key = 'wed_end', disabled=wed_disable)
    
    with tab5:
        thu_disable = st.checkbox('Disable', value=False, key='thu_disable')
        thu_start = st.time_input('Start:', datetime.time(9, 00), key = 'thu_start', disabled=thu_disable)
        thu_end = st.time_input('End:', datetime.time(17, 00), key = 'thu_end', disabled=thu_disable)
    
    with tab6:
        fri_disable = st.checkbox('Disable', value=False, key='fri_disable')
        fri_start = st.time_input('Start:', datetime.time(9, 00), key = 'fri_start', disabled=fri_disable)
        fri_end = st.time_input('End:', datetime.time(17, 00), key = 'fri_end', disabled=fri_disable)
    
    with tab7:
        sat_disable = st.checkbox('Disable', value=True, key='sat_disable')
        sat_start = st.time_input('Start:', key='sat_start', disabled=sat_disable)
        sat_end = st.time_input('End:', key='sat_end', disabled=sat_disable)
        
    st.divider() 
    update_schedule = st.button("Restart Server")
    
    if save_settings:
        if delete_source_url != '':
            for source in new_camera_list:
                if delete_source_url == source['url']:
                    new_camera_list.remove(source)
                    
        if new_model != config['model']:
            cache_key = "object_detection_yolo" 
            if cache_key in st.session_state:
                del st.session_state[cache_key]
        config['sources'] = new_camera_list
        config['model'] = new_model
        config['img_size'] = new_img_size
        config['img_aug'] = new_img_augment
        config['conf_thres'] = new_conf_thres
        config['iou_thres'] = new_iou_thres
        config['show_cameras'] = new_show_cameras
        config['mask_faces'] = new_mask_faces
        config['api_key'] = new_api_key
        config['chat_id'] = new_chat_id
        
        with open(CONFIG_FILE, 'w') as outfile:
            json.dump(config, outfile, indent=4)
            st.sidebar.text("Settings saved successfully.")
    
    if update_schedule:
        # print(os.getcwd())
        # stop_run_continuously.set()
        stop_server()
        if enable_scheduler:
            schedule.clear()
            if not sun_disable :
                schedule.every().sunday.at(str(sun_start)).do(start_server, cpu=enable_cpu)
                schedule.every().sunday.at(str(sun_end)).do(stop_server)
            if not mon_disable :
                schedule.every().monday.at(str(mon_start)).do(start_server, cpu=enable_cpu)
                schedule.every().monday.at(str(mon_end)).do(stop_server)
            if not tue_disable :
                schedule.every().tuesday.at(str(tue_start)).do(start_server, cpu=enable_cpu)
                schedule.every().tuesday.at(str(tue_end)).do(stop_server)
            if not wed_disable :
                schedule.every().wednesday.at(str(wed_start)).do(start_server, cpu=enable_cpu)
                schedule.every().wednesday.at(str(wed_end)).do(stop_server)
            if not thu_disable :
                schedule.every().thursday.at(str(thu_start)).do(start_server, cpu=enable_cpu)
                schedule.every().thursday.at(str(thu_end)).do(stop_server)
            if not fri_disable:
                schedule.every().friday.at(str(fri_start)).do(start_server, cpu=enable_cpu)
                schedule.every().friday.at(str(fri_end)).do(stop_server)
            if not sat_disable:
                schedule.every().saturday.at(str(sat_start)).do(start_server, cpu=enable_cpu)
                schedule.every().saturday.at(str(sat_end)).do(stop_server)
        else: 
            start_server(cpu=enable_cpu)   
                            
        st.sidebar.text("Server restarted.")
        
    if start:
        start_server(cpu=enable_cpu)
    
    if stop:
        stop_server()