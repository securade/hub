# HUB

Securade.ai HUB  is a generative AI based edge platform for computer vision that connects to existing CCTV cameras and makes them smart.
It uses natural language text and generative AI to automatically train and fine-tune state-of-the-art computer vision models on the edge. This eliminates costly data labelling and annotations work typically required in training new models. Thus, enabling us to deploy a custom accurate model per camera feed.

<div align="center">
  <img src="https://securade.ai/assets/images/blog/securade.ai-edge-app-screenshot.jpeg" alt="Securade.ai HUB" width="600"/>
  <br />
  <br />
  <a href="https://securade.streamlit.app/"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App"/></a>
</div>

## Features

🤖 **Zero-Shot Learning** - No manual data labeling needed - train models using natural language descriptions

🎯 **Real-Time Detection** - Live monitoring of CCTV feeds for safety and security incidents

👷 **PPE Detection** - Automated detection of hardhat, vest, and mask compliance

⚡ **Proximity Alerts** - Monitor safe distances between workers and machinery/vehicles

🚫 **Exclusion Zones** - Configurable restricted area monitoring and access control

📊 **Analytics Dashboard** - Real-time safety metrics and violation tracking interface

🎥 **Multi-Camera Support** - Compatible with D-link, Tapo, TP-Link, Axis and HikVision cameras

🔔 **Instant Notifications** - Real-time Telegram alerts for detected violations

🎭 **Privacy Protection** - Automatic face masking to maintain worker privacy

💻 **Edge Processing** - Local processing without cloud dependency

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

Once the HUB is installed you will need to configure the streamlit web server.

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

To use Telegram for notifications, you will need to get an `api_id` and the `chat_id` for the channel or group you want to send the notifications.
You can get the `api_id` by creating a new application [here](https://core.telegram.org/api/obtaining_api_id). The `chat_id` can be obtained
by following the instructions [here](https://stackoverflow.com/q/32423837). Make sure you add your bot to the channel or group and set group admin rights for it.

You can watch a detailed demo and turorial on how to use the HUB [here](https://www.youtube.com/playlist?list=PLphF_n2JfD10TEjFfKwQPBCdA47lyv7ae).

### On a Jetson device

Installing the HUB on an edge device with GPU is significantly more involved as we need to ensure that correct versions of Jetpack, CUDA, cuDNN, Torch and Torchvision
are installed. You can read the detailed instructions on the [wiki](https://github.com/securade/hub/wiki/How-to-install-on-Jetson). 

The HUB should work with any Jetson device with atleast 8 GB of memory. It has been tested to work on Lenovo ThinkEdge SE70 Edge Client and the NVIDIA Jetson AGX Orin Developer Kit.

## Plugin System

The HUB includes a flexible plugin system that allows you to extend its image processing capabilities. You can create custom plugins to add new computer vision features, image processing algorithms, or integrate with other AI models.

### Available Plugins

The HUB comes with several built-in plugins:
- **YOLO Object Detector**: Default object detection using YOLOv7
- **Edge Detector**: Basic edge detection using OpenCV

### Creating Custom Plugins

To create a new plugin, add a Python file to the `plugins` directory that implements the BasePlugin interface:

```python
from plugins.base_plugin import BasePlugin

class MyPlugin(BasePlugin):
    SLUG = "my_plugin"
    NAME = "My Custom Plugin"
    
    def run(self, image, **kwargs):
        # Your image processing code here
        return processed_image
```

The plugin will automatically be discovered and appear in the HUB interface's plugin selection dropdown.

For detailed documentation on creating plugins, including examples and best practices, see the [plugins/README.md](plugins/README.md).

## License

Securade.ai HUB is open-source and available under the GNU AGPL license. You can freely use it on your own edge devices or servers.
If you are looking to bundle it and distribute to others, you will need a commercial license. 

You can get the [Securade.ai Commercial License](https://securade.ai/subscribe) for a nominal fees of 99 SGD per month. 
Securade.ai is a tech4good venture and the commercial license allows you to deploy the HUB on an unlimited number of devices and servers.

## Partners and Customers

<div align="center">
  <table border="0" cellspacing="10" cellpadding="20">
    <tr>
      <td align="center" width="200">
        <img src="https://imgur.com/SJIyr7P.png" width="150" alt="Panasonic"/>
      </td>
      <td align="center" width="200">
        <img src="https://imgur.com/RpNEomG.png" width="150" alt="Omron"/>
      </td>
      <td align="center" width="200">
        <img src="https://imgur.com/Bd3VnU4.png" width="150" alt="Lenovo"/>
      </td>
    </tr>
    <tr>
      <td align="center" width="200">
        <img src="https://imgur.com/AnHTgT0.png" width="150" alt="Axis Communications"/>
      </td>
      <td align="center" width="200">
        <img src="https://imgur.com/OruGLi2.png" width="150" alt="NVIDIA"/>
      </td>
      <td align="center" width="200">
        <img src="https://imgur.com/qTI4QWi.png" width="150" alt="King Island"/>
      </td>
    </tr>
        <tr>
      <td align="center" width="200">
        <img src="https://imgur.com/kDcJjnd.png" width="150" alt="Woh Hup"/>
      </td>
      <td align="center" width="200">
        <img src="https://imgur.com/WkboPy9.png" width="150" alt="Vestar Iron Works"/>
      </td>
      <td align="center" width="200">
        <img src="https://imgur.com/IDeo2xX.png" width="150" alt="SRPOST"/>
      </td>
    </tr>
  </table>
</div>

## References

- [HUB Wiki](https://github.com/securade/hub/wiki)
- [Deep Dive Demos](https://www.youtube.com/playlist?list=PLphF_n2JfD10TEjFfKwQPBCdA47lyv7ae)
- [White Paper on Generative AI-Based Video Analytics](https://securade.ai/assets/pdfs/Securade.ai-Generative-AI-Video-Analytics-Whitepaper.pdf)
- [Solution Deck](https://securade.ai/assets/pdfs/Securade.ai-Solution-Overview.pdf)
- [Customer Case Study](https://securade.ai/assets/pdfs/Vestar-Iron-Works-Pte-Ltd-Case-Study.pdf)
- [Safety Copilot for Worker Safety](https://securade.ai/safety-copilot.html)
- [White Paper on Safey Copilot](https://securade.ai/assets/pdfs/Securade.ai-Safety-Copilot-Whitepaper.pdf)
- [More Resources](https://securade.ai/resources.html)
