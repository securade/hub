#!/bin/bash

# Define the directory
F=$1
# F=/Users/user/Documents/GitHub/edge-app
cd $F

# Function to activate the virtual environment
activate () {
  source .venv/bin/activate
}

# Function to retry a command with exponential backoff and stop after max delay
retry_command() {
  local delay=1
  local max_delay=600  # Maximum delay in seconds
  local attempt=1

  while true; do
    "$@" && break || {
      echo "Command failed: $@"
      echo "Retrying in ${delay}s... (attempt: $attempt)"
      sleep $delay
      if [ $delay -ge $max_delay ]; then
        echo "Max delay reached. Stopping attempts."
        break
      fi
      delay=$(( delay * 2 ))
      attempt=$(( attempt + 1 ))
    }
  done
}

# Activate the virtual environment
activate

# Start the applications with exponential backoff
streamlit run Configure_Camera.py &
python dashboard.py &
python ./pynvr/pynvrd.py &
retry_command python securade.py --config config.json --cpu &