# HUB

Securade.ai HUB  is a generative AI based edge platform for computer vision that connects to existing CCTV cameras and makes them smart.
It uses natural language text and generative AI to automatically train and fine-tune state-of-the-art computer vision models on the edge. This eliminates costly data labelling and annotations work typically required in training new models. Thus, enabling us to deploy a custom accurate model per camera feed.

<div align="center">
  <img src="https://securade.ai/assets/images/blog/securade.ai-edge-app-screenshot.jpeg" alt="Securade.ai HUB" width="600"/>
</div>

## Features

🤖 **Zero-Shot Learning** - Train computer vision models using natural language descriptions without manual data labeling

🎯 **Real-Time Detection** - Process live CCTV feeds to detect safety violations and security incidents in real-time

👷 **PPE Detection** - Automatically identify workers not wearing required safety equipment like hardhats, vests, and masks

⚡ **Proximity Alerts** - Detect unsafe proximity between workers and heavy machinery or vehicles

🚫 **Exclusion Zones** - Monitor restricted areas and detect unauthorized access with configurable zone policies

📊 **Analytics Dashboard** - Track safety metrics, violation trends, and generate reports through an intuitive web interface

🎥 **Multi-Camera Support** - Connect to multiple CCTV cameras simultaneously with support for major brands like D-link, Tapo, TP-Link, Axis and HikVision

🔔 **Instant Notifications** - Receive real-time alerts via Telegram when safety violations are detected

🎭 **Privacy Protection** - Optional face masking feature to protect worker privacy in captured images

💻 **Edge Processing** - Run entirely on local hardware with no cloud dependency, ideal for sites with limited connectivity

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

You can watch the detailed demo and turorial on how to use the HUB [here](https://www.youtube.com/playlist?list=PLphF_n2JfD10TEjFfKwQPBCdA47lyv7ae).

The HUB can connect with any CCTV camera with RTSP streams. It has been tested to work with cameras from various vendors like D-link, Tapo, TP-Link, Axis Communications
and HikVision.

### On a Jetson device
Installing the HUB on an edge device with GPU is significantly more involved as we need to ensure that correct versions of Jetpack, CUDA, cuDNN, Torch and Torchvision
are installed. You can read the detailed instructions on the [wiki](https://github.com/securade/hub/wiki/How-to-install-on-Jetson). 

The HUB should work with any Jetson device with atleast 8 GB of memory. It has been tested to work on Lenovo ThinkEdge SE70 Edge Client and the NVIDIA Jetson AGX Orin Developer Kit.

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

- [Deep Dive Demos](https://www.youtube.com/playlist?list=PLphF_n2JfD10TEjFfKwQPBCdA47lyv7ae)
- [White Paper on Generative AI-Based Video Analytics](https://securade.ai/assets/pdfs/Securade.ai-Generative-AI-Video-Analytics-Whitepaper.pdf)
- [Solution Deck](https://securade.ai/assets/pdfs/Securade.ai-Solution-Overview.pdf)
- [Customer Case Study](https://securade.ai/assets/pdfs/Vestar-Iron-Works-Pte-Ltd-Case-Study.pdf)
- [Safety Copilot for Worker Safety](https://securade.ai/safety-copilot.html)
- [More Resources](https://securade.ai/resources.html)
