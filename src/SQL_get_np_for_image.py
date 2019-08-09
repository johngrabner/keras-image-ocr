import os

import random
import numpy as np
import PIL
from PIL import ImageOps

mypath_images = r'/crops_set/crop_images/'

def get_np_for_image(id, w, h, train): 
    try:
        #train = True # experiment 1, force  rotation and noise when train=false
        im = PIL.Image.open(mypath_images + id + ".png")

        old_size = im.size # old_size[0] = width, old_size[1] = height

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

            assert(im_np_w_h_c.shape[0]==w)
            assert(im_np_w_h_c.shape[1]==h)
            assert(im_np_w_h_c.shape[2]==1)

            return im_np_w_h_c / 255

    except:
       print("failed", "id", id, "shape", im_np_h_w.shape)
    
       return None 


