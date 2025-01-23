#!/bin/bash

# Define the service name
SERVICE_NAME="openvpn3-session@CloudConnexa.service"

# Function to check the status of the service
check_service_status() {
    # Get the status of the service
    status=$(systemctl status $SERVICE_NAME | grep "Status:")

    # Check if the status is not the desired one
    if [[ $status != *"StatusMajor.CONNECTION:StatusMinor.CONN_CONNECTED"* ]]; then
        echo "Service status is not as expected. Restarting the service..."
        sudo systemctl stop $SERVICE_NAME
        sudo systemctl start $SERVICE_NAME
        echo "Service restarted."
    else
        echo "Service status is as expected. No action needed."
    fi
}

# Check if the service is running
if systemctl is-active --quiet $SERVICE_NAME; then
	echo "$SERVICE_NAME is already running."
	check_service_status
else
	echo "$SERVICE_NAME is stopped. Starting it now..."
	# Start the service
	sudo systemctl start $SERVICE_NAME
	echo "$SERVICE_NAME started."
fi
