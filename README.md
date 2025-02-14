# References

- https://keras.io/examples/image_ocr/   
- https://github.com/mbhenry/keras/blob/master/examples/image_ocr.py

The root source. This example uses a convolutional stack followed by a recurrent stack and a CTC logloss function to perform optical character recognition of generated text images.

- https://github.com/Tony607/keras-image-ocr

Chengwei Zhang adds prediction to the original.  Also moves code to a jupyter notebook and gives instructions to run in Google Colab and installation on your local dev machine.

- https://www.dlology.com/blog/how-to-train-a-keras-model-to-recognize-variable-length-text/

This article by Chengwei Zhang explains the operation of this code.

- https://towardsdatascience.com/faq-build-a-handwritten-text-recognition-system-using-tensorflow-27648fb18519
- https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5
- https://github.com/githubharald/SimpleHTR

Need to read / study these articles from Harald Scheidl 

# My Version

The goal of my modifications is mainly to aid the learning of this code, Keras CTC loss, and explore a deep learning development environment using Nvidia Docker, remote debugging from Visual Studio Code.

# Tasks
- [x] Docker Image to run Keras, Jupyter, Remote Debugging
- [x] Separate the generator so that different parts of the code are clearly delineated.
- [ ] Move bulk of code into Python modules so that I can use Visual Studio Code
- [ ] A second source for training data
- [ ] Experiment with curriculum training
- [ ] Add tensorboard for visualizing differences in runs

# Docker Image

Installing Tensorflow, Cuda, and all the dependencies can be messy.  You may need to modify or add to your development environment as you bounce between projects.  So at any time, your environment is the merger of many project needs.  Incompatible needs create confusion.  Reconstructing a project environment efficiently may difficult given the muddled path followed.   

Tensorflow does get into funky states.  Rebooting your PC is time-consuming and forces you to close the may windows you may have open.

Docker solves the above.  Create a  build script for your specific project environment needs. Quickly run.

 On your PC only installed the nvidia drivers and nvidia's docker. Keras and Tensorflow are run in a container. See https://www.tensorflow.org/install/docker for details on installation.

to build and run the docker image

```
cd /Documents/GitHub/keras-image-ocr/docker/keras-image-ocr-gpu-py3-jupyter
/bin/bash build.sh 

/bin/bash run.sh to run the docker image. This is found in the docker directory of this project.
```