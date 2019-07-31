import os
from os import walk
import random
import keras
from keras import backend as K
import numpy as np
import json
import numpy as np
import PIL
from PIL import ImageOps
#from PIL import Image
from PIL import ImageFilter

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# The "help me translate" program has a SQL backend.
# This SQL dumps all the images in file located at
#    /media/john/dropbox/transcription_db/crop_sets/2019_07_24/crop_images
#    /dropbox/transcription_db/crop_sets/2019_07_24/crop_images
#    with one file per image and file name is id.png
#
#    Inside Docker is is at /crops_set/crop_images
#
# Also created by the SQL is a ground truth translates for some of these images.
# These are located at 
#    /media/john/dropbox/transcription_db/crop_set/2019_07_24/transcribed_words
#    /dropbox/transcription_db/crop_set/2019_07_24/transcribed_words. Same as before
#    file names are id.txt
#
#    Inside Docker is is at /crops_set/transcribed_words
#
# this reads the json, them crops all images and places them in
# a temporary directory
#import make_tmp_german_png

print("code under development ...")
print("click debug in visual studio code")
# Setting up remote debugging:
#     https://code.visualstudio.com/docs/python/debugging

# Allow other computers to attach to ptvsd at this IP address and port.
#     ptvsd.enable_attach(address=('1.2.3.4', 3000), redirect_output=True)
#import ptvsd
#ptvsd.enable_attach()

# Pause the program until a remote debugger is attached
#print("WAITING FOR DEBUGGER")
#ptvsd.wait_for_attach()

# This generator will select words from the following database:
# -------------------------------------------------------------
# hand crafted db by jjg

alphabet = u'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÖÜßáäõöüăČčŠšẞ,-. '
alphabet = u'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÖÜßáäõöüăČčŠšẞ '
#mypath = '/home/john/Documents/GitHub/ancient-german/db-of-words-crops/known_words.json'
mypath_images = r'/crops_set/crop_images/'
mypath_words = r'/crops_set/transcribed_words'
words_id_text_list = [] 


def read_word_from_txt_file(id, file_to_open, words_id_text_list, max_string_len):

    word = open(file_to_open).read()

    if len(word)>0:
        #if len(words_id_text_list)<32:  #for debug only, do not submit
        words_id_text_list.append( {"id": id, "word": word})





def read_all_words_in_directory(mypath, words_id_text_list, max_string_len):

    for (dirpath, dirnames, filenames) in walk(mypath):
        for name in filenames:
            if name.endswith(".txt"):
                file_to_open = os.path.join(dirpath, name)

                read_word_from_txt_file(name[:-4], file_to_open, words_id_text_list, max_string_len)



# # alphabet is required for 1 hot encoding
def make_alphabet(words_id_text_list):
    alphabet = ""

    for w in words_id_text_list:
        for c in w["text"]:
            if c not in alphabet:
                alphabet = alphabet + c

    # sort the alphabet used in these words so that it's easier to see if any are missing
    alphabet = ''.join(sorted(alphabet))
    #alphabet = alphabet + " "
    return alphabet

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


def strip_invalid_characters(in_str):
    # remove non matching alphabet 
    look_to_remove_invalid = True
    while look_to_remove_invalid:
        look_to_remove_invalid = False
        for index_in_str, c in enumerate(in_str):
            i = alphabet.find(c)
            if i == -1:
                look_to_remove_invalid = True # loop until remove non
                in_str = in_str[0:index_in_str]+in_str[index_in_str+1:]
                break
    return in_str

# original only had lowercase letters
def is_valid_str(in_str):

    # remove non matching alphabet, since images were stripped too
    in_str = strip_invalid_characters(in_str)

    if len(in_str)==0:
        return False

    if len(in_str)>15:
        return False
    # todo, for now, restrict to small words ... lets see if this helps
    #if len(in_str)>4:
    #    return False

    # check for invalid chars, only usefull when not striping invalids 
    for c in in_str:
        i = alphabet.find(c)
        if i == -1:
            return False

    return True



def get_np_for_image(record_fn_text, w, h, train): # example id = m04-012-08-02
    try:
    
        im = PIL.Image.open(mypath_images+record_fn_text["id"]+".png")

        # experiment 1 - no equalize 
        # experiment 2
        #im = ImageOps.equalize(im) # did not help, stuck at 12 vs virgin at 11
        #im = im.filter(ImageFilter.MedianFilter).filter(ImageFilter.CONTOUR) # did not help, stuck at 12 vs virgin at 11

        # min_y = record_fn_text["record"]["coordinates"]["min_y"]
        # max_y = record_fn_text["record"]["coordinates"]["max_y"]
        # min_x = record_fn_text["record"]["coordinates"]["min_x"]
        # max_x = record_fn_text["record"]["coordinates"]["max_x"]
        # im = im.crop((min_x, min_y, max_x, max_y))
        # im.load() # lazy ops, so load will force eval
        
        # im = ImageOps.autocontrast(im, 0.1)
        # im = im.convert('L') # black and white

        old_size = im.size # old_size[0] = width, old_size[1] = height

        # try rotation to help learning
        angle = 0

        if train:
            angle = np.random.normal(scale=5)

        im = im.rotate(angle=angle, resample=PIL.Image.BILINEAR, fillcolor=255)
        if old_size[0] != im.size[0]:
            raise Exception("size changed" )
        if old_size[1] != im.size[1]:
            raise Exception("size changed" )

        # see https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
        height_ratio_to_fill = float(h) / float(old_size[1])
        width_ratio_to_fill = float(w) / float(old_size[0])
        ratio_to_fill = min(height_ratio_to_fill, width_ratio_to_fill)

        height_ratio_to_be_half = 0.5 * float(h) / float(old_size[1])
        width_ratio_to_be_half = 0.5 * float(w) / float(old_size[0])
        ratio_to_half = min(height_ratio_to_be_half, width_ratio_to_be_half)



        if train:
            # so target size
            if ratio_to_fill > ratio_to_half:
                # normal case
                ratio = random.uniform(ratio_to_half,ratio_to_fill)
            else:
                # scale up to full size, but not down
                ratio = random.uniform(1, ratio_to_fill)
        else:
            ratio = ratio_to_fill

        new_size = tuple([int(round(x*ratio)) for x in old_size])
        if new_size[0]>w:
            new_size = (w, int(new_size[1]/new_size[0]*w))
        if new_size[1]>h:
            new_size = (int(new_size[0]/new_size[1]*h), h)
        im = im.resize(new_size, PIL.Image.ANTIALIAS)

        if new_size[0]>w:
            print("should not happen")
            return None
        else:
            missing_height = h-new_size[1]
            missing_width = w-new_size[0]

            if missing_width>0:
                in_front = random.randrange(0,missing_width)
            else:
                in_front = 0    
            in_back = missing_width - in_front

            if missing_height>0:
                on_top = random.randrange(0,missing_height)
            else:
                on_top = 0
            on_bottom = missing_height - on_top

            im = ImageOps.expand(im, (in_front, on_top, in_back, on_bottom), fill=255)

            if im.mode == "RGBA":
                new_image = PIL.Image.new("RGBA", im.size, "WHITE") # Create a white rgba background
                new_image.paste(im, (0, 0), im)
                #new_image.convert('RGB')
                im = new_image.convert('L') #makes it greyscale

            im_np_h_w = np.array(im)
            #print("id", record_fn_text["id"], "shape", im_np_h_w.shape)

            add_noise = train # add noise only if training
            # for debug to see
            #add_noise = False
            if add_noise:
                mean = 0.0   # some constant
                std = 40.0    # some constant (standard deviation)
                std = abs(np.random.normal(scale=20)) # amount of noise varies
                im_np_h_w = im_np_h_w + np.random.normal(mean, std, im_np_h_w.shape)
                im_np_h_w = np.clip(im_np_h_w, 0, 255)  # we might get out of bounds due to noise

            im_np_w_h = np.moveaxis(im_np_h_w, -1, 0) # numpy.moveaxis(a, source, destination)
            im_np_w_h_c = np.expand_dims(im_np_w_h, axis=2) # add dimension for channel

            #print("shape", im_np_w_h_c.shape)
            assert(im_np_w_h_c.shape[0]==w)
            assert(im_np_w_h_c.shape[1]==h)
            assert(im_np_w_h_c.shape[2]==1)

            #print("good", "id", record_fn_text["id"], "shape", im_np_h_w.shape)

            return im_np_w_h_c / 255

        
        #print(split_id)
    # except Exception as ex:
    #     print("failed", "id", record_fn_text["id"], "shape", im_np_h_w.shape)
    #     template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    #     message = template.format(type(ex).__name__, ex.args)
    #     print (message)
    #     return None 
    except:
       print("failed", "id", record_fn_text["id"], "shape", im_np_h_w.shape)
    
       return None 


#read_all_words_in_directory(mypath, words_id_text_list, 16)
#print(len(words_id_text_list))
#alphabet = make_alphabet(words_id_text_list)
#print(alphabet)
#for i in range(300): #len(words_id_text_list)):
#    get_np_for_image(words_id_text_list[i], 128, 64)
#print("done")
# todo, alphabet is input to generator and restricts words based on alphabet

class German_Word_Generator(keras.callbacks.Callback):

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
        read_all_words_in_directory(mypath_words, self.words_id_text_list, max_string_len)
        #random.shuffle(self.words_id_text_list)

        # todo, rename this variable
        
        #self.words_id_text_list = json.loads(open(mypath).read())
        ##self.words_id_text_list = make_tmp_german_png.create_temp_png_files_and_return_records()
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

                candidate_word = strip_invalid_characters(words_id_text_list[i]['word'])
                if is_valid_str(candidate_word):
                    im = get_np_for_image(words_id_text_list[i], self.img_w, self.img_h, train)
                    

                    if im is not None:
                        #print("shape", im.shape)
                        image_batch.append(im)
                        source_str_batch.append(candidate_word)
                        lables_batch.append(text_to_labels(candidate_word, self.absolute_max_string_len, alphabet))
                        lables_length_batch.append(float(len(candidate_word)))
                        ctc_input_length.append(float(self.img_w // self.downsample_factor - 2)) # magic number from perspective of generator
                #else:
                    #print("rejecting word ", words_id_text_list[i]['german_text'])

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
        #print("on_train_begin")
        ii = 0

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
        #print("on_epoch_begin")
        iii = 0



#ger = German_Word_Generator(32,512, 64, 2, 16)
#b = ger.get_batch(ger.words_id_text_list, 32, True)
#print(len(b))