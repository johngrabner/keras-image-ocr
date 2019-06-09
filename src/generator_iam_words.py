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
import ptvsd
ptvsd.enable_attach()

# Pause the program until a remote debugger is attached
ptvsd.wait_for_attach()

# This generator will select words from the following database:
# -------------------------------------------------------------
# http://www.fki.inf.unibe.ch/databases/iam-handwriting-database/download-the-iam-handwriting-database
#
# The IAM Handwriting Database is publicly accessible and freely available for non-commercial research purposes. 
#
# U. Marti and H. Bunke. The IAM-database: An English Sentence Database for Off-line Handwriting Recognition. 
# Int'l Journal on Document Analysis and Recognition, Volume 5, pages 39 - 46, 2002.


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




# alphabet is required for 1 hot encoding
def make_alphabet(words_id_text_list):
    alphabet = ""

    for w in words_id_text_list:
        for c in w["text"]:
            if c not in alphabet:
                alphabet = alphabet + c

    # sort the alphabet used in these words so that it's easier to see if any are missing
    alphabet = ''.join(sorted(alphabet))
    alphabet = alphabet.join(" ")
    return alphabet

# Translation of characters to unique integer values
def text_to_labels(text):
    ret = []
    for char in text:
        ret.append(alphabet.find(char))
    return ret

# original only had lowercase letters
def is_valid_str(in_str):
    return True

import numpy as np
import PIL
from PIL import ImageOps

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_np_for_image(id, w, h): # example id = m04-012-08-02
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
        im = ImageOps.expand(im, (0, 0, w-new_size[0],0))
        im_np_h_w = np.array(im)
        im_np_w_h = np.moveaxis(im_np_h_w, -1, 0) # numpy.moveaxis(a, source, destination)
        im_np_w_h_c = np.expand_dims(im_np_w_h, axis=2) # add dimension for channel
        return im_np_w_h_c

    
    #print(split_id)


mypath = '/home/john/Documents/GitHub/iam/unzipped/xml/'
words_id_text_list = [] 

# read_all_words_in_directory(mypath, words_id_text_list, max_string_len=4)
# random.shuffle(words_id_text_list)

# alphabet = make_alphabet(words_id_text_list)

# get_np_for_image(words_id_text_list[0]["id"])

# print("alphabet is ", alphabet)            
# print("words generator will produce", len(words_id_text_list))

#im = get_np_for_image("m04-012-08-02", 128, 64) # expect it is w 101, h 76
#print(im.shape)

class IAM_Word_Generator(keras.callbacks.Callback):

    def __init__(self, minibatch_size,
                 img_w, img_h, downsample_factor, val_split,
                 absolute_max_string_len=16):

        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        self.downsample_factor = downsample_factor
        self.val_split = val_split
        #self.blank_label = self.get_output_size() - 1 # last entry of alphabet needs to be blank
        self.absolute_max_string_len = absolute_max_string_len
        self.words_id_text_list = [] 
        self.index_into_word_id_text_list = 0
        self.alphabet = ""


    def get_output_size(self):
        return len(self.alphabet) + 1

    def build_word_list(self, max_string_len): # tbd, why max_string_length here
        read_all_words_in_directory(mypath, self.words_id_text_list, max_string_len) # tbd, why max_string_length here
        random.shuffle(self.words_id_text_list)
        self.alphabet = make_alphabet(self.words_id_text_list) # tbd, is alphabet needed for model before first run

    def get_batch(self, index, minibatch_size, train):
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        if K.image_data_format() == 'channels_first':
            ## not implemented
            assert False
        else:
            # TensorFlow
            batch = []
            while len(batch)<minibatch_size:
                i = random.randint(0, len(self.words_id_text_list)-1)
                im = get_np_for_image(self.words_id_text_list[i]['id'], self.img_w, self.img_h)

                if im is not None:
                    batch.append(im)
            
            return np.array(batch)
    
    def on_train_begin(self, logs={}):
        self.build_word_list(16) #tbd, why string length here


# development test    
# ---------------- 
g = IAM_Word_Generator(minibatch_size = 32, img_w = 128, img_h = 64, downsample_factor=-1, val_split=-1, absolute_max_string_len=16)
#im = IAM.get_np_for_image("m04-012-08-02", 128, 64)
g.on_train_begin()
im = g.get_batch(-1, 32, True)
print(im.shape)