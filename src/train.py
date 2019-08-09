#!/usr/bin/env python
# coding: utf-8

# version from exporting the jupyter notebook to python.
# Now, i can see delta/diff in git
#
# ## References
# 
# see README.md

# ## Baseline with generator_choice = "GSQL"
# 200 interation got to 10.xxx something

# ## Basedline with generator_choice =  "German" on July 31, 2019
# - Epoch 1/200  - loss: 22.5422 - val_loss: 44.3061
# - Epoch 2/200  - loss: 20.7822 - val_loss: 73.2133
# - Epoch 3/200  - loss: 20.5434 - val_loss: 20.1864
# - Epoch 4/200  - loss: 19.6281 - val_loss: 18.1584
# - Epoch 5/200  - loss: 18.3635 - val_loss: 17.9036
# - Epoch 6/200  - loss: 17.1934 - val_loss: 16.0855
# - Epoch 7/200  - loss: 16.2462 - val_loss: 14.7814
# - Epoch 8/200  - loss: 15.4579 - val_loss: 14.7433
# - Epoch 9/200  - loss: 14.7134 - val_loss: 13.2510
# - Epoch 10/200 - loss: 14.1979 - val_loss: 12.8388
# - Epoch 11/200 - loss: 13.7155 - val_loss: 12.4681
# - Epoch 12/200 - loss: 13.1744 - val_loss: 12.4697
# - Epoch 13/200 - loss: 12.8387 - val_loss: 12.2364
# - Epoch 14/200 - loss: 12.4855 - val_loss: 12.0318
# - Epoch 15/200 - loss: 12.1907 - val_loss: 11.1764
# - Epoch 16/200 - loss: 11.7321 - val_loss: 11.1777
# - Epoch 17/200 - loss: 11.5132 - val_loss: 11.0034
# - Epoch 18/200 - loss: 11.1854 - val_loss: 11.1696
# - Epoch 19/200 - loss: 11.0407 - val_loss: 10.7113
# - Epoch 20/200 - loss: 10.6401 - val_loss: 10.6150
# - Epoch 21/200 - loss: 10.4692 - val_loss: 10.4183
# - Epoch 22/200 - loss: 10.2791 - val_loss: 11.1796
# - Epoch 23/200 - loss: 9.9911 - val_loss: 10.4712
# - Epoch 24/200 - loss: 9.7312 - val_loss: 10.4887
# - Epoch 25/200 - loss: 9.4863 - val_loss: 10.3883
# - Epoch 26/200 - loss: 9.3771 - val_loss: 10.2179
# - Epoch 27/200 - loss: 9.0670 - val_loss: 10.1179
# - Epoch 28/200 - loss: 8.7698 - val_loss: 10.2423
# - Epoch 29/200 - loss: 8.6642 - val_loss: 10.3849
# - Epoch 30/200 - loss: 8.4662 - val_loss: 10.1517
# - Epoch 31/200 - loss: 8.2157 - val_loss: 10.0927
# - Epoch 32/200 - loss: 8.0640 - val_loss: 10.0808
# - Epoch 33/200 - loss: 7.8819 - val_loss: 10.0854
# - Epoch 34/200 - loss: 7.8030 - val_loss: 9.7388
# - Epoch 35/200 - loss: 7.5315 - val_loss: 10.1922
# - Epoch 36/200 - loss: 7.3847 - val_loss: 10.1470
# - Epoch 37/200 - loss: 7.3319 - val_loss: 10.0684
# - Epoch 38/200 - loss: 7.1130 - val_loss: 9.6827
# - Epoch 39/200 - loss: 7.0022 - val_loss: 10.3757
# - Epoch 40/200 - loss: 6.9908 - val_loss: 10.1155
# - Epoch 41/200 - loss: 6.6640 - val_loss: 10.3377
# - Epoch 42/200 - loss: 6.5059 - val_loss: 10.1471
# - Epoch 43/200 - loss: 6.6160 - val_loss: 9.5882
# - Epoch 44/200 - loss: 6.3216 - val_loss: 10.1955
# - Epoch 45/200 - loss: 6.2142 - val_loss: 10.1830
# - Epoch 46/200 - loss: 6.1089 - val_loss: 10.4291
# - Epoch 47/200 - loss: 6.0326 - val_loss: 10.4027
# - Epoch 48/200 - loss: 5.8245 - val_loss: 10.6347
# - Epoch 49/200 - loss: 5.9137 - val_loss: 10.6149
# - Epoch 50/200 - loss: 5.6554 - val_loss: 10.1823
# - Epoch 51/200 - loss: 5.6850 - val_loss: 9.9264
# - Epoch 52/200 - loss: 5.5150 - val_loss: 10.0987
# - Epoch 53/200 - loss: 5.4917 - val_loss: 9.8134
# 

# In[1]:


import os
import itertools


import datetime

import editdistance
import numpy as np

import pylab
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks


# In[2]:


OUTPUT_DIR = 'image_ocr'



np.random.seed(55)


# In[3]:


# Text_Image is the original generator. It creates images programaticaly. 
# Script_Image takes handwritten words from the IAM database.
generator_choice =  "GSQL" #"German" # "Script_Image" # "German" # "Text_Image" #
import generator_text_image as GTI
import generator_iam_words as IAM
import generator_german_words as GER
import generator_SQL_words as GSQL


# In[4]:


import ctc_drop_first_2
import cnn_rnn_model


# In[5]:





# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        if generator_choice ==  "Script_Image":
            outstr = IAM.labels_to_text(out_best)
        elif generator_choice == "German":
            outstr = GER.labels_to_text(out_best)
        elif generator_choice == "GSQL":
            outstr = GSQL.labels_to_text(out_best)
        elif generator_choice == "Text_Image":
            outstr = GTI.labels_to_text(out_best)
        else:
            assert(False)
        ret.append(outstr)
    return ret


# In[6]:


class VizCallback(keras.callbacks.Callback):

    def __init__(self, run_name, test_func, text_img_gen, num_display_words=6):
        self.test_func = test_func
        self.output_dir = os.path.join(
            OUTPUT_DIR, run_name)
        self.text_img_gen = text_img_gen
        self.num_display_words = num_display_words
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def show_edit_distance(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        while num_left > 0:
            word_batch = next(self.text_img_gen)[0]
            num_proc = min(word_batch['the_input'].shape[0], num_left)
            decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:num_proc])
            for j in range(num_proc):
                edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        print('\nOut of %d samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f'
              % (num, mean_ed, mean_norm_ed))

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(os.path.join(self.output_dir, 'weights%02d.h5' % (epoch)))
        self.show_edit_distance(256)
        word_batch = next(self.text_img_gen)[0]
        res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words])
        if word_batch['the_input'][0].shape[0] < 256:
            cols = 2
        else:
            cols = 1
        for i in range(self.num_display_words):
            plt.subplot(self.num_display_words // cols, cols, i + 1)
            if K.image_data_format() == 'channels_first':
                the_input = word_batch['the_input'][i, 0, :, :]
            else:
                the_input = word_batch['the_input'][i, :, :, 0]
            plt.imshow(the_input.T, cmap='Greys_r')
            plt.xlabel('Truth = \'%s\'\nDecoded = \'%s\'' % (word_batch['source_str'][i], res[i]))
        fig = pylab.gcf()
        fig.set_size_inches(10, 13)
        plt.savefig(os.path.join(self.output_dir, 'e%02d.png' % (epoch)))
        plt.close()


# In[7]:


img_gen = None
def train(run_name, start_epoch, stop_epoch, img_w):
    global img_gen
    img_h = 64
    
    # Input Parameters

    words_per_epoch = 16000
    val_split = 0.2
    
    # 16000 * 0.2 = 3200
    val_words = int(words_per_epoch * (val_split))

    # Network parameters
    minibatch_size = 32
    
    
    # (16000 - 3200) // 32 = 400
    steps_per_epoch = (words_per_epoch - val_words) // minibatch_size


    if generator_choice == "Text_Image":
        fdir = os.path.dirname(get_file('wordlists.tgz',
                                        origin='http://www.mythic-ai.com/datasets/wordlists.tgz', untar=True))

        img_gen = GTI.TextImageGenerator(monogram_file=os.path.join(fdir, 'wordlist_mono_clean.txt'),
                                     bigram_file=os.path.join(fdir, 'wordlist_bi_clean.txt'),
                                     minibatch_size=minibatch_size,
                                     img_w=img_w,
                                     img_h=img_h,
                                     downsample_factor=(pool_size ** 2),
                                     val_split=words_per_epoch - val_words
                                     )
    elif generator_choice == "Script_Image":
        img_gen = IAM.IAM_Word_Generator(minibatch_size = 32, img_w = img_w, img_h = img_h, downsample_factor=4, absolute_max_string_len=16)
    elif generator_choice == "German":
        img_gen = GER.German_Word_Generator(minibatch_size = 32, img_w = img_w, img_h = img_h, downsample_factor=4, absolute_max_string_len=16)
    elif generator_choice == "GSQL":
        img_gen = GSQL.German_Word_Generator(minibatch_size = 32, img_w = img_w, img_h = img_h, downsample_factor=4, absolute_max_string_len=16)
    else:
        assert False

    model, model_p, input_data, y_pred = cnn_rnn_model.make_model(img_w, img_h, img_gen.get_output_size(), img_gen.absolute_max_string_len)
    
    model_p.summary() # print summary of model before added ctc
    model.summary() # print summary of model
    
    from keras.utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True)
    from IPython.display import Image
    Image(filename='model.png')

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    if start_epoch > 0:
        weight_file = os.path.join(OUTPUT_DIR, os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
        model.load_weights(weight_file)
    # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], [y_pred])

    viz_cb = VizCallback(run_name, test_func, img_gen.next_val())

    model.fit_generator(generator=img_gen.next_train(),
                        steps_per_epoch=steps_per_epoch,
                        epochs=stop_epoch,
                        validation_data=img_gen.next_val(),
                        validation_steps=val_words // minibatch_size,
                        callbacks=[viz_cb, img_gen],
                        initial_epoch=start_epoch)


# In[ ]:





# ### Start the training
# 
# 1080ti run times: 
# - 12 minutes for 20/20 epoch
# - 25 minutes for 25/25 epoch
# 

# In[8]:


## run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
train(run_name, 0, 200, 128)
# increase to wider images and start at epoch 20. The learned weights are reloaded
img_gen.set_img_w(512)
train(run_name, 20, 250, 512)


# In[ ]:





# In[ ]:





# In[ ]:




