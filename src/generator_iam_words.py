import os
from os import walk
import xml.etree.ElementTree as ET
import random
import keras
from keras import backend as K
import numpy as np

print("code under development ...")
print("click debug in visual studio code")
# Setting up remote debugging:
#     https://code.visualstudio.com/docs/python/debugging

# Allow other computers to attach to ptvsd at this IP address and port.
#     ptvsd.enable_attach(address=('1.2.3.4', 3000), redirect_output=True)
#import ptvsd
#ptvsd.enable_attach()

# Pause the program until a remote debugger is attached
#ptvsd.wait_for_attach()

# This generator will select words from the following database:
# -------------------------------------------------------------
# http://www.fki.inf.unibe.ch/databases/iam-handwriting-database/download-the-iam-handwriting-database
#
# The IAM Handwriting Database is publicly accessible and freely available for non-commercial research purposes. 
#
# U. Marti and H. Bunke. The IAM-database: An English Sentence Database for Off-line Handwriting Recognition. 
# Int'l Journal on Document Analysis and Recognition, Volume 5, pages 39 - 46, 2002.

alphabet = u'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '

def read_words_from_xml_file(file_to_open, words_id_text_list, max_string_len):
    root = ET.parse(file_to_open).getroot()

    # Top level contains 2 elements. we want the "handwritten-part" element.
    for c in root.getchildren():
        if c.tag == 'handwritten-part':

            # handwritten-part contains 'lines'
            for c2 in c.getchildren():
                if c2.tag == 'line':

                    # "lines" contains "words"
                    for c3 in c2.getchildren():
                        if c3.tag == 'word':

                            if len(c3.attrib["text"]) < max_string_len:
                                words_id_text_list.append( {"id": c3.attrib["id"], "text": c3.attrib["text"]})




def read_all_words_in_directory(mypath, words_id_text_list, max_string_len):

    for (dirpath, dirnames, filenames) in walk(mypath):
        for name in filenames:
            file_to_open = os.path.join(dirpath, name)

            read_words_from_xml_file(file_to_open, words_id_text_list, max_string_len)




# # alphabet is required for 1 hot encoding
# def make_alphabet(words_id_text_list):
#     alphabet = ""

#     for w in words_id_text_list:
#         for c in w["text"]:
#             if c not in alphabet:
#                 alphabet = alphabet + c

#     # sort the alphabet used in these words so that it's easier to see if any are missing
#     alphabet = ''.join(sorted(alphabet))
#     #alphabet = alphabet + " "
#     return alphabet

# Translation of characters to unique integer values
def text_to_labels(text, absolute_max_string_len, alphabet):
    ret = []
    for i in range(absolute_max_string_len):
        if i<len(text):
            ret.append(alphabet.find(text[i]))
        else:
            ret.append(-1)
    return ret

# Reverse translation of numerical classes back to characters
def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)

# original only had lowercase letters
def is_valid_str(in_str):

    # todo, for now, restrict to small words ... lets see if this helps
    #if len(in_str)>4:
    #    return False

    for c in in_str:
        i = alphabet.find(c)
        if i == -1:
            return False
    return True

import numpy as np
import PIL
from PIL import ImageOps

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_np_for_image(id, w, h): # example id = m04-012-08-02
    try:
        mypath = '/home/john/Documents/GitHub/iam/unzipped/words/'
        split_id = id.split("-")
        mypath = mypath + split_id[0] + "/" + split_id[0] + "-" + split_id[1] + "/" + id + ".png"

        im = PIL.Image.open(mypath)
        
        old_size = im.size # old_size[0] = width, old_size[1] = height

        # see https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
        height_ratio = float(h) / float(old_size[1])
        new_size = tuple([int(round(x*height_ratio)) for x in old_size])
        im = im.resize(new_size, PIL.Image.ANTIALIAS)

        if new_size[0]>w:
            return None
        else:
            im = ImageOps.expand(im, (0, 0, w-new_size[0],0), fill=255)
            im_np_h_w = np.array(im)
            im_np_w_h = np.moveaxis(im_np_h_w, -1, 0) # numpy.moveaxis(a, source, destination)
            im_np_w_h_c = np.expand_dims(im_np_w_h, axis=2) # add dimension for channel
            return im_np_w_h_c / 255

        
        #print(split_id)
    except:
       return None 

mypath = '/home/john/Documents/GitHub/iam/unzipped/xml/'
words_id_text_list = [] 

# todo, alphabet is input to generator and restricts words based on alphabet

class IAM_Word_Generator(keras.callbacks.Callback):

    def __init__(self, minibatch_size,
                 img_w, img_h, downsample_factor, 
                 absolute_max_string_len=16):

        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        self.downsample_factor = downsample_factor
        #self.blank_label = self.get_output_size() - 1 # last entry of alphabet needs to be blank
        self.absolute_max_string_len = absolute_max_string_len
        self.words_id_text_list = [] 
        self.validation_words_id_text_list = []
        self.index_into_word_id_text_list = 0
        self.build_word_list(16)

    def set_img_w(self, w):
        self.img_w = w

    def set_absolute_max_string_len(self, m):
        self.absolute_max_string_len = m

    def get_output_size(self):
        return len(alphabet) + 1

    def build_word_list(self, max_string_len): # tbd, why max_string_length here
        read_all_words_in_directory(mypath, self.words_id_text_list, max_string_len) # tbd, why max_string_length here
        random.shuffle(self.words_id_text_list)
        # 25% for validation
        validation_0_to_n = int(len(self.words_id_text_list)/4)
        self.validation_words_id_text_list = self.words_id_text_list[:validation_0_to_n]
        self.words_id_text_list = self.words_id_text_list[validation_0_to_n:]

    def get_batch(self, words_id_text_list, minibatch_size, train):
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        if K.image_data_format() == 'channels_first':
            ## not implemented
            assert False
        else:
            # TensorFlow
            image_batch = []
            source_str_batch = [] # abc
            lables_batch = [] # [1, 2, 3, -1, -1 ] where 1=a, 2=b, 3=c
            lables_length_batch = [] # how long each text
            ctc_input_length = [] 

            while len(image_batch)<minibatch_size:
                i = random.randint(0, len(words_id_text_list)-1)
                if is_valid_str(words_id_text_list[i]['text']):
                    im = get_np_for_image(words_id_text_list[i]['id'], self.img_w, self.img_h)

                    if im is not None:
                        image_batch.append(im)
                        source_str_batch.append(words_id_text_list[i]['text'])
                        lables_batch.append(text_to_labels(words_id_text_list[i]['text'], self.absolute_max_string_len, alphabet))
                        lables_length_batch.append(float(len(words_id_text_list[i]['text'])))
                        ctc_input_length.append(float(self.img_w // self.downsample_factor - 2)) # magic number from perspective of generator
                #else:
                    #print("rejecting word ", words_id_text_list[i]['text'])

            inputs = {'the_input': np.array(image_batch),          # this corresponds to cnn_rnn_model Input(name='the_input'
                  'the_labels': np.array(lables_batch),         # this corresponds to cnn_rnn_model Input(name='the_labels'
                  'input_length': np.expand_dims(np.array(ctc_input_length),axis=2), # this corresponds to cnn_rnn_model Input(name='input_length',
                  'label_length': np.expand_dims(np.array(lables_length_batch),axis=2), # this corresponds to cnn_rnn_model Input(name='label_length'
                  'source_str': np.array(source_str_batch)      # used for visualization only
                  }
            outputs = {'ctc': np.zeros([minibatch_size])}  # dummy data for dummy loss function
            return (inputs, outputs)
    
    def on_train_begin(self, logs={}):
        # word list built on creation of the generator object
        print("on_train_begin")

    # model.fit_generator(generator=img_gen.next_train()
    def next_train(self):
        while 1:
            ret = self.get_batch(self.words_id_text_list, self.minibatch_size, train=True)
            yield ret

    # model.fit_generator(validation_data=img_gen.next_val()
    def next_val(self):
        while 1:
            ret = self.get_batch(self.validation_words_id_text_list, self.minibatch_size, train=False)
            yield ret

    def on_epoch_begin(self, epoch, logs={}):
        print("on_epoch_begin")

