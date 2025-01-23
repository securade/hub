from pathlib import Path
from st_click_detector import click_detector as did_click
import os
import streamlit as st
from PIL import Image

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

    st.title("Browse Videos")
    st.sidebar.markdown("View the recordings and download videos.")
    
    st.cache(lambda: st.session_state, allow_output_mutation=True)
    
    def get_subfolders_and_files(folder_path):
        subfolders = []
        files = []
        folder_path = os.path.normpath(folder_path).replace("\\", "/")
        try:
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)

                # Check permissions
                if not os.access(item_path, os.R_OK):
                    print(f"Permission denied for {item_path}")
                    continue

                # Check symbolic link
                if os.path.islink(item_path):
                    print(f"{item_path} is a symbolic link")
                    continue

                if os.path.isdir(item_path):
                    subfolders.append({"name": item, "path": os.path.normpath(item_path)})
                else:
                    files.append({"name": item, "path": os.path.normpath(item_path)})
            return subfolders, files
        except PermissionError as e:
            st.info(e)
            subfolders.sort(key=lambda x: x['name'])
            files.sort(key=lambda x: x['name'])
            return subfolders, files


    def get_folder_list(folder_path):
        folder_list = []
        current_path = ""
        current_path = folder_path.replace("\\", "/")
        split_drive = Path(current_path).parts
        folders = split_drive
        for folder in folders:
            current_path = os.path.join(current_path, folder)
            folder_list.append({"name": folder, "path": current_path})
        folder_list.sort(key=lambda x: x['name'])
        return folder_list


    def generate_folder_links(folder_path):
        paths = st.session_state["crumbs"]
        subfolders, files = get_subfolders_and_files(folder_path)
        crumbs = {crumb["name"]: crumb["path"] for crumb in paths}
        current_crumb = paths[-1]["name"]
        st.session_state[
            "dir_list"
        ] = f'<font size={st.session_state["font_size"]} face="tahoma" color="{st.session_state["color_2"]}"> \ </font>'.join(
            [
                f'<a href="#" id="{crumbs[crumb["name"]]}"><font size={st.session_state["font_size"]} face="tahoma" color="{st.session_state["color_1"]}">{crumb["name"]}</font></a>'
                for crumb in paths[:-1]
            ]
            + [
                f'<font size={st.session_state["font_size"]} face="tahoma" color="{st.session_state["color_2"]}">{current_crumb}</font>'
            ]
        )
        folder_links = {sub["name"]: sub["path"] for sub in subfolders}
        file_links   = {file["name"]: file["path"] for file in files}
        # print(file_links)
        # print(subfolders)
        folder_list = None
        htmlstyle = """<style>
            a:link, a:visited {
            background-color: #79797918;
            color: gray;
            padding: 0px 10px;
            text-align: left;
            text-decoration: none;
            display: column-count:5;

            }
            a:hover, a:active {
            background-color: #98989836;
            }
            </style>"""
        if len(subfolders) > 0:
            num_of_columns = 3
            folder_list = [
                f'<a href="#" id="{folder_links[subfolder["name"]]}">'
                f'{htmlstyle}<font face="tahoma" color="{st.session_state["color_2"]}">🗀</font> {subfolder["name"]}'
                f"</a>"
                for subfolder in subfolders
            ]
            # print(files)
            folder_list += [
                f'<a href="#" id="{file_links[file["name"]]}">'
                f'{htmlstyle}<font face="tahoma" color="{st.session_state["color_2"]}"> - </font> {file["name"]}'
                f"</a>"
                for file in files
            ]
        else:
            folder_list = [
                f'<a href="#" id="{file_links[file["name"]]}">'
                f'{htmlstyle}<font face="tahoma" color="{st.session_state["color_2"]}"> - </font> {file["name"]}'
                f"</a>"
                for file in files
            ]
            #print(folder_list)
        st.session_state["dirs"] = "<br>".join(folder_list or [])
        #print(st.session_state["dirs"])


    def update_paths():
        my_path = st.session_state.get("mypath", os.getcwd())
        try:
            subfolders, files = get_subfolders_and_files(my_path)
            st.session_state["subfolders"] = subfolders
            st.session_state["files"] = files
        except Exception as e:
            st.exception(e)
        try:
            crumbs = get_folder_list(my_path)
            st.session_state["crumbs"] = crumbs
        except Exception as e:
            st.exception(e)


    def update_dir_list():
        st.session_state["new_crumb"] = did_click(st.session_state["dir_list"], None)
        if st.session_state["new_crumb"]:
            update_paths()
            st.session_state["run_again"] = True


    def update_dirs():
        st.session_state["new_subfolder"] = did_click(st.session_state["dirs"], None)
        new_sub = st.session_state.get("new_subfolder")
        if new_sub:
            if os.path.isfile(new_sub):
                with open(new_sub, "rb") as f:
                    if new_sub.endswith(".jpg"):
                        image = Image.open(new_sub)
                        st.sidebar.image(image)
                        # st.sidebar.download_button('Download '+ new_sub, data=f, file_name=os.path.basename(new_sub), mime="image/jpeg")
                    elif new_sub.endswith(".mp4"):
                        video_bytes = f.read()
                        st.sidebar.video(video_bytes)
                    st.sidebar.download_button('Download '+ new_sub, f, file_name=os.path.basename(new_sub))  
                    # Defaults to 'application/octet-stream'
                st.session_state["new_subfolder"] = None
            else:
                update_paths()
                st.session_state["run_again"] = True


    def new_path():
        current_path = st.session_state.get("mypath", os.getcwd())
        new_crumb = st.session_state.get("new_crumb")
        new_subfolder = st.session_state.get("new_subfolder")
        if new_crumb:
            st.session_state["new_crumb"] = None
            st.session_state["new_path"] = new_crumb
        elif new_subfolder:
            st.session_state["new_path"] = new_subfolder
        else:
            st.session_state["new_path"] = current_path


    def update_new_path():
        new_path()
        update_paths()
        generate_folder_links(st.session_state["new_path"])
        update_dir_list()
        update_dirs()
        new_path()
        return st.session_state["new_path"]


    st.session_state["font_size"] = "16"
    st.session_state["color_1"] = "#0088ff"
    st.session_state["color_2"] = "#ff8800"

    if "new_path" not in st.session_state:
        update_paths()
        st.session_state["new_path"] = st.sidebar.text_input(
            "mypath", os.getcwd(), key="mpath"
        )
        generate_folder_links(st.session_state["new_path"])
    else:
        st.session_state["mypath"] = st.session_state.get("new_path", os.getcwd())

    st.session_state["mypath"] = update_new_path()

    if st.session_state.get("run_again"):
        st.session_state["run_again"] = False
        update_paths()
        st.experimental_rerun()

    st.session_state["mypath"]