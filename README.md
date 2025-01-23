# HUB

Securade.ai HUB  is a generative AI based edge platform for computer vision that connects to existing CCTV cameras and makes them smart.
It uses natural language text and generative AI to automatically train and fine-tune state-of-the-art computer vision models on the edge. This eliminates costly data labelling and annotations work typically required in training new models. Thus, enabling us to deploy a custom accurate model per camera feed.

## Installation

```bash
git clone https://github.com/securade/hub.git
cd hub
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You can check if the installation was successful by running:

```bash
python securade.py --help
usage: securade.py [-h] [--config CONFIG] [--cpu] [--no_activity_alert NO_ACTIVITY_ALERT] [--server] [--version]

options:
  -h, --help            show this help message and exit
  --config CONFIG       Load config from file
  --cpu                 Use the OpenVINO Runtime for inference on CPU
  --no_activity_alert NO_ACTIVITY_ALERT
                        Time in seconds after which a no activity alert is raised
  --server              Run the Securade web server application
  --version             show program's version number and exit
```

Once the hub is installed you will need to configure the streamlit web server.

```bash
mkdir .streamlit
cp config.toml .streamlit
cp secrets.toml .streamlit
```

You can then run the web server to configure the CCTV cameras and policies:

```bash
python securade.py --server
--------------------------------------------------------------------------
#    #####                                                          #      
#   #     # ######  ####  #    # #####    ##   #####  ######       # #   # 
#   #       #      #    # #    # #    #  #  #  #    # #           #   #  # 
#    #####  #####  #      #    # #    # #    # #    # #####      #     # # 
#         # #      #      #    # #####  ###### #    # #      ### ####### # 
#   #     # #      #    # #    # #   #  #    # #    # #      ### #     # # 
#    #####  ######  ####   ####  #    # #    # #####  ###### ### #     # # 
--------------------------------------------------------------------------   

Press Enter to exit ...

  You can now view your Streamlit app in your browser.

  Network URL: http://192.168.10.147:8080
  External URL: http://58.182.134.244:8080
```

The default password is `pass`, you can change it in the `secrets.toml` file that you have in the `.streamlit` folder.

You can watch the detailed demo and turorial on how to use the HUB [here](https://www.youtube.com/playlist?list=PLphF_n2JfD10TEjFfKwQPBCdA47lyv7ae).

### On a Jetson device
Installing the HUB on an edge device with GPU is significantly more involved as we need to ensure that correct versions of Jetpack, CUDA, cuDNN, Torch and Torchvision
are installed. You can read the detailed instructions on the wiki. 

The HUB should work with any Jetson device with atleast 8 GB of memory. It has been tested to work on Lenovo ThinkEdge SE70 Edge Client and the Nvidia Jetson AGX Orin Developer Kit.

## License

Securade.ai HUB is open-source and available under the GNU AGPL license. You can freely use it on your own edge devices or servers.
If you are looking to bundle it and distribute to others, you will need a commercial license. 

You can get the [Securade.ai Commercial License](https://securade.ai/subscribe) for a nominal fees of 99 SGD per month. 
Securade.is is a tech4good venture and the commercial license allows you to deploy the HUB on an unlimited number of devices and servers.

## Customers & Partners
