
import os
from os import walk
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
import SQL_get_np_for_image as SQL_image

import ptvsd
ptvsd.enable_attach()
print("WAITING FOR DEBUGGER")
ptvsd.wait_for_attach()




np.random.seed(55)



generator_choice =  "GSQL" 
mypath_images = r'/crops_set/crop_images/'
mypath_ai_guess = r'/crops_set/ai_guess/'

import generator_SQL_words as GSQL




import ctc_drop_first_2
import cnn_rnn_model







# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        if generator_choice == "GSQL":
            outstr = GSQL.labels_to_text(out_best)
        else:
            assert(False)
        ret.append(outstr)
    return ret





    
def decode_predict_ctc(out, top_paths = 1):
    results = []
    beam_width = 5
    if beam_width < top_paths:
      beam_width = top_paths
    for i in range(top_paths):
      lables = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0])*out.shape[1],
                           greedy=False, beam_width=beam_width, top_paths=top_paths)[0][i])[0]
      text = GSQL.labels_to_text(lables)
      results.append(text)
    return results    

def predit_a_image(a, model_p, top_paths = 1):
  c = np.expand_dims(a, axis=0)
  net_out_value = model_p.predict(c)
  top_pred_texts = decode_predict_ctc(net_out_value, top_paths)
  return top_pred_texts


img_gen = None
def predict(weight_file, img_w, img_h):
    global img_gen
    
    # Input Parameters

    words_per_epoch = 16000
    val_split = 0.2
    
    # 16000 * 0.2 = 3200
    val_words = int(words_per_epoch * (val_split))

    # Network parameters
    minibatch_size = 32
    
    
    # (16000 - 3200) // 32 = 400
    steps_per_epoch = (words_per_epoch - val_words) // minibatch_size


    if generator_choice == "GSQL":
        img_gen = GSQL.German_Word_Generator(minibatch_size = 32, img_w = img_w, img_h = img_h, downsample_factor=4, absolute_max_string_len=16)
    else:
        assert False

    model, model_p, input_data, y_pred = cnn_rnn_model.make_model(img_w, img_h, img_gen.get_output_size(), img_gen.absolute_max_string_len)
    
    
    model.summary() # print summary of model

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    
    
    model.load_weights(weight_file)

    # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], [y_pred])

    # read all files 
    for (dirpath, dirnames, filenames) in walk(mypath_images):
        for name in filenames:
            if name.endswith(".png"):
                id = name[:-4]

                im = SQL_image.get_np_for_image(id, img_w, img_h, train=False)
    
                #im2 = np.expand_dims(im, axis=0)
                #net_out_value = model_p.predict(im2)
                #pred_texts = decode_predict_ctc(net_out_value)
                pred_texts = predit_a_image(im, model_p, top_paths = 3)
                #print("id", id, "text", pred_texts)

                ai_guess_file_path = os.path.join(mypath_ai_guess, id) + '.txt'
                f=open(ai_guess_file_path,"w+")
                f.write("{}\t{}\t{}".format(pred_texts[0], pred_texts[1], pred_texts[2]))
                f.close()



# path on ubuntu = xxxxx
# mapped in docker image with command = -v xxxx
# path in docker image = xxxxxx
WEIGHTS_DIR = r'image_ocr'
run_name = r'2019:07:31:20:12:17'
weight_file = os.path.join(WEIGHTS_DIR, os.path.join(run_name, 'weights%02d.h5' % (199)))


predict(weight_file, 128, 64)



















