FROM tensorflow/tensorflow:1.13.1-gpu-py3-jupyter

RUN pip3 install keras
RUN pip3 install Pillow
RUN pip3 install ptvsd

RUN pip3 install matplotlib
RUN pip3 install graphviz
RUN pip3 install pydot
RUN pip3 install cairocffi
RUN pip3 install editdistance

RUN apt-get install -y libcairo2-dev
RUN apt install -q -y graphviz

# A place for tensorboard logs. Not using presently.
RUN mkdir /tmp/logs

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]
