#!/bin/bash

# do i need ?
# sudo systemctl daemon-reload
# sudo systemctl restart docker

# run with /bin/bash run.sh
#sudo docker run --runtime=nvidia -it --rm -v ~/Documents/GitHub/ancient-german/docker/tensorboard/tf_logs:/tmp/logs -v ~/Documents/GitHub/ancient-german:/tf/notebooks -p 8888:8888 -p 5678:5678 jjg-gpu-py3-jupyter-keras-pillow:1.13.1
sudo docker run --runtime=nvidia -it --rm -v ~/Documents/GitHub:/home/john/Documents/GitHub -v /tmp/tf_logs:/tmp/logs -v ~/Documents/GitHub/keras-image-ocr:/tf/notebooks -p 8888:8888 -p 5678:5678 keras-image-ocr-gpu-py3-jupyter:1.13.1
