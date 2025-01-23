#!/bin/bash

# F=/Users/user/Documents/GitHub/edge-app
cd $1
activate () {
  source .venv/bin/activate
}

activate
streamlit run Configure_Camera.py &
python dashboard.py &
python securade.py --config config.json --cpu &
python ./pynvr/pynvrd.py &
