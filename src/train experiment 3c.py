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





OUTPUT_DIR = 'image_ocr'
image_name = 'experiment_3c_dropout.png'


np.random.seed(55)




# Text_Image is the original generator. It creates images programaticaly. 
# Script_Image takes handwritten words from the IAM database.
generator_choice =  "GSQL" #"German" # "Script_Image" # "German" # "Text_Image" #
#import generator_text_image as GENERATOR
#import generator_iam_words as GENERATOR
#import generator_german_words as GENERATOR
#import generator_SQL_words as GENERATOR

import generator_SQL_words as GENERATOR


import ctc_drop_first_2
import model_experiment_3c as cnn_rnn_model








# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

# place in generator maybe ??? or store generator selected and pull xxx.labels_to_text
def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        
        outstr = GENERATOR.labels_to_text(out_best)
       
        ret.append(outstr)
    return ret



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

        img_gen = GENERATOR.TextImageGenerator(monogram_file=os.path.join(fdir, 'wordlist_mono_clean.txt'),
                                     bigram_file=os.path.join(fdir, 'wordlist_bi_clean.txt'),
                                     minibatch_size=minibatch_size,
                                     img_w=img_w,
                                     img_h=img_h,
                                     downsample_factor=(pool_size ** 2),
                                     val_split=words_per_epoch - val_words
                                     )
    else 
        img_gen = GENERATOR.IAM_Word_Generator(      minibatch_size = 32, img_w = img_w, img_h = img_h, downsample_factor=4, absolute_max_string_len=16)
    

    model, model_p, input_data, y_pred = cnn_rnn_model.make_model(img_w, img_h, img_gen.get_output_size(), img_gen.absolute_max_string_len)
    
    model_p.summary() # print summary of model before added ctc
    model.summary() # print summary of model
    

    from keras.utils import plot_model
    plot_model(model, to_file=image_name, show_shapes=True)
    from IPython.display import Image
    Image(filename=image_name)

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






# ### Start the training
# 
# 1080ti run times: 
# - 12 minutes for 20/20 epoch
# - 25 minutes for 25/25 epoch
# 




## run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
train(run_name, 0, 200, 128)
# increase to wider images and start at epoch 20. The learned weights are reloaded
#img_gen.set_img_w(512)
#train(run_name, 20, 250, 512)


















