#!/bin/bash

# do i need ?
# sudo systemctl daemon-reload
# sudo systemctl restart docker

# run with /bin/bash run.sh
sudo docker run --runtime=nvidia -it --rm -v /tmp/tf_logs:/tmp/logs -p 6006:6006 jjg-tensorboard:1.13.1
