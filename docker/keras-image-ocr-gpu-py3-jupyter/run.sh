#!/bin/bash

# This is required after installing the Nvidia drivers.  Once system rebooted, not required.
#     sudo systemctl daemon-reload

# This is required to start docker. On my system Docker does not start boot time.
#     sudo systemctl restart docker

# This is required to run this bash file. 
#     /bin/bash run.sh
sudo docker run --runtime=nvidia -it --rm -v ~/Documents/GitHub:/home/john/Documents/GitHub -v /tmp/tf_logs:/tmp/logs -v ~/Documents/GitHub/keras-image-ocr:/tf/notebooks -p 8888:8888 -p 5678:5678 keras-image-ocr-gpu-py3-jupyter:1.13.1
