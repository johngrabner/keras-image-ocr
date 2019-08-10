
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
import time

import ptvsd
#ptvsd.enable_attach()
#print("WAITING FOR DEBUGGER")
#ptvsd.wait_for_attach()




np.random.seed(55)



generator_choice =  "GSQL" 
mypath_images = r'/crops_set/crop_images/'
mypath_ai_guess = r'/crops_set/ai_guess/'

import generator_SQL_words as GSQL




import ctc_drop_first_2
import model_experiment_3c as cnn_rnn_model







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





# todo CRAZY slow    
def decode_predict_ctc(out, top_paths = 1):
    results = []
    beam_width = 5
    if beam_width < top_paths:
      beam_width = top_paths
    for i in range(top_paths):
        t0 = time.time()
        v1 = np.ones(out.shape[0])
        t1 = time.time()
        v2 = v1*out.shape[1]
        t2 = time.time()
        v3 = K.ctc_decode(out, input_length=v2, greedy=False, beam_width=beam_width, top_paths=top_paths)[0][i]
        t3 = time.time()
        lables = K.get_value(v3)[0]
        t4 = time.time()
        #lables = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0])*out.shape[1],
        #                   greedy=False, beam_width=beam_width, top_paths=top_paths)[0][i])[0]
        
        text = GSQL.labels_to_text(lables)
        
        results.append(text)
        
        print("why slow down" t4-t3)
    return results    

def predit_a_image(a, model_p, top_paths = 1):
    t0 = time.time()  
    c = np.expand_dims(a, axis=0)
    t1 = time.time()
    net_out_value = model_p.predict(c)
    t2 = time.time()
    top_pred_texts = decode_predict_ctc(net_out_value, top_paths)
    t3 = time.time()
    #print(t1-t0, t2-t1, t3-t2)
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

                id_int = int(id)
                if id_int>253987:
                    t0 = time.time()
                    im = SQL_image.get_np_for_image(id, img_w, img_h, train=False)
                    t1 = time.time()
                    #im2 = np.expand_dims(im, axis=0)
                    #net_out_value = model_p.predict(im2)
                    #pred_texts = decode_predict_ctc(net_out_value)
                    if im is not None:
                        
                        pred_texts = predit_a_image(im, model_p, top_paths = 1)
                        t2 = time.time()
                        

                        ai_guess_file_path = os.path.join(mypath_ai_guess, id) + '.txt'
                        f=open(ai_guess_file_path,"w+")
                        #f.write("{}\t{}\t{}".format(pred_texts[0], pred_texts[1], pred_texts[2]))
                        f.write("{}".format(pred_texts[0]))
                        f.close()
                        t3 = time.time()
                        #print("get image", t1-t0, "predict",t2-t1, "write", t3-t2, "id", id, "text", pred_texts)
                        print("id", id, "text", pred_texts)



# path on ubuntu = xxxxx
# mapped in docker image with command = -v xxxx
# path in docker image = xxxxxx
WEIGHTS_DIR = r'image_ocr'
run_name = r'2019:08:09:19:49:02'
weight_file = os.path.join(WEIGHTS_DIR, os.path.join(run_name, 'weights%02d.h5' % (141)))


predict(weight_file, 128, 64)



















