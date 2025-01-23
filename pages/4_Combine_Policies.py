import streamlit as st
import json

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
    st.title("Combine Policies")

    st.sidebar.markdown("Combine two different safety policies to create a combo.")
    
    policy_json_1 = st.file_uploader("Upload the first safety policy file (e.g. ppe_policy.json)", type=["json"], key="policy1")
    policy_json_2 = st.file_uploader("Upload the second safety policy file (e.g. exclusion_policy.json)", type=["json"], key="policy2")
    
    if policy_json_1 is not None and policy_json_2 is not None:
        data_1 = json.load(policy_json_1)
        data_2 = json.load(policy_json_2)
        if data_1 is not None and data_2 is not None:
            if data_1['type'] == 'ppe_detection' and data_2['type'] == 'exclusion_zones':
                policy = {
                    "type" : "ppe_detection_exclusion_zones",
                    "hardhats" : data_1['hardhats'],
                    "vests" : data_1['vests'],
                    "masks" : data_1['masks'],
                    "no_hardhats" : data_1['no_hardhats'],
                    "no_vests" : data_1['no_vests'],
                    "no_masks" : data_1['no_masks'],                    
                    "persons" : data_2['persons'],
                    "machinery" : data_2['machinery'],
                    "vehicle" : data_2['vehicle'],
                    "inclusion_zone" : data_2['inclusion_zone'], 
                    "max_allowed" : data_2['max_allowed'],
                    "zones" : data_2['zones']
                }
                st.download_button(label="Save", data=json.dumps(policy),file_name="ppe_detection_exclusion_zones.json", mime="text/plain")
            else:
                st.write("Only ppe_detection.json and exclusion_policy.json can be combined at the moment.")