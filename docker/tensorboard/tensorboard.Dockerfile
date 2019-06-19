FROM tensorflow/tensorflow:1.13.1-gpu-py3-jupyter

RUN pip3 install keras
RUN pip3 install Pillow

RUN mkdir /tmp/logs

CMD ["sh", "-c", "tensorboard --logdir=/tmp/logs"]