
import PIL
from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter
import matplotlib.pyplot as plt
import os
#from skimage import feature
#import numpy

#import ptvsd
#ptvsd.enable_attach()



# Pause the program until a remote debugger is attached
#ptvsd.wait_for_attach()

path = "C:/Users/john_/Documents/GitHub/ancient-german/db-of-words-crops/db-of-crops-of-known-words/"
fn0 = "D:/Dropbox (Grabner)/German Source Script Transcription/wetransfer-b606cb/Scans and transcriptions/Postcards part 1/Postcard 47.jpg"
fn5 = "D:/Dropbox (Grabner)/German Source Script Transcription/wetransfer-b606cb/Scans and transcriptions/Postcards part 1/Postcard 1.jpg"
fn1 = "C:/Users/john_/Documents/GitHub/ancient-german/db-of-words-crops/db-of-crops-of-known-words/17032.png" # gray background
fn2 = "C:/Users/john_/Documents/GitHub/ancient-german/db-of-words-crops/db-of-crops-of-known-words/16991.png" # gray background
fn3 = "C:/Users/john_/Documents/GitHub/ancient-german/db-of-words-crops/db-of-crops-of-known-words/10102.png" # clean
fn4 = "C:/Users/john_/Documents/GitHub/ancient-german/db-of-words-crops/db-of-crops-of-known-words/10340.png" # clean

def threshold_grayscale(fn):

    # open image
    im1 = PIL.Image.open(fn)
    #im1.show()

    # make gray
    #im1 = im1.convert('L')
    #im1.show()

    #im1.show()
    #im1_equalized.show()

    #imageWithEdges = im1.filter(ImageFilter.FIND_EDGES)
    #imageWithEdges.show()

    # nice, but next is better
    #imageWithCountour = im1.filter(ImageFilter.CONTOUR) 
    #imageWithCountour.show()

    countour_of_median = im1.filter(ImageFilter.MedianFilter).filter(ImageFilter.CONTOUR)
    countour_of_median.show(title=fn)

    # inefective on faded post card
    # countour_of_equialized = ImageOps.equalize(im1)#.filter(ImageFilter.CONTOUR)
    # countour_of_equialized.show()

    # ugly
    # now_contour = countour_of_equialized.filter(ImageFilter.CONTOUR)
    # now_contour.show()

    # ugly
    # c2 = imageWithCountour.filter(ImageFilter.CONTOUR)
    # c2.show()

    # ugly
    #imageWithEdgeEqualized = ImageOps.equalize(imageWithEdges)
    #imageWithEdgeEqualized.show()

    # ugly
    # imageWithCountourEqualized = ImageOps.equalize(imageWithCountour)
    # imageWithCountourEqualized.show()

    # # historgram, needed for threshold to make white
    # im1_hist = im1.histogram()  
    
    # # calculate number of white pixels
    # pixel_count = im1.height * im1.width
    # percent_to_make_white = 0.8
    # pixel_to_be_white = int(pixel_count * percent_to_make_white)

    # # calculate what value will be masked to white
    # sum = 0
    # value_where_white_pixels_met = 0
    # for i in range(255, 0, -1):
    #     sum = sum + im1_hist[i]
    #     if sum > pixel_to_be_white:
    #         value_where_white_pixels_met = i
    #         break
    
    
    # image_with_whites_percent_met = Image.eval(im1, lambda a: a if a <= value_where_white_pixels_met else 255)

    # image_with_whites_percent_met.show()

    # im1_equalized_and_thresholded = ImageOps.equalize(image_with_whites_percent_met)
    # im1_equalized_and_thresholded.show()

    #equalize_hist = im1_equalized.histogram()


    # calculate slope
    # alpha = 0.1
    # slope = 0
    # slope_histogram = []

    # count = 0
    # for i in range(len(im1_hist)):
    #     slope = (1-alpha)*slope + alpha* im1_hist[i]
    #     slope_histogram.append(slope) 

    # bins = list(range(256))
    # plt.plot(bins, im1_hist, 'r')
    # plt.plot(bins, equalize_hist, "g")
    
    # #plt.plot(bins, slope_histogram, 'g')
    # plt.xlabel('Pixel value')
    # plt.ylabel('slope_histogram')
    # plt.title("histogram")
    # # plt.legend(
    # #     ('histogram', 'slope'),
    # #     shadow=True, loc=(0.01, 0.75))
    # plt.grid(True)
    # plt.show()


#threshold_grayscale(fn0)
#threshold_grayscale(fn5)
#threshold_grayscale(fn1)
#threshold_grayscale(fn2)
#threshold_grayscale(fn3)
#threshold_grayscale(fn4)

for root, dirs, files in os.walk(path):
    #print root
    #print dirs
    #print files
    for fn in files:
        threshold_grayscale(os.path.join(root, fn))
        print(fn)
