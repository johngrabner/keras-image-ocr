import os
from os import walk
import xml.etree.ElementTree as ET
import random

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


def read_words_from_xml_file(file_to_open, words_id_text_list):
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

                            #print(c3.tag, c3.attrib)
                            words_id_text_list.append( {"id": c3.attrib["id"], "text": c3.attrib["text"]})




def read_all_words_in_directory(mypath, words_id_text_list):

    for (dirpath, dirnames, filenames) in walk(mypath):
        for name in filenames:
            file_to_open = os.path.join(dirpath, name)

            read_words_from_xml_file(file_to_open, words_id_text_list)




# alphabet is required for 1 hot encoding
def make_alphabet(words_id_text_list):
    alphabet = ""

    for w in words_id_text_list:
        for c in w["text"]:
            if c not in alphabet:
                alphabet = alphabet + c

    # sort the alphabet used in these words so that it's easier to see if any are missing
    alphabet = ''.join(sorted(alphabet))
    return alphabet

mypath = '/home/john/Documents/GitHub/iam/unzipped/xml/'
words_id_text_list = [] 

read_all_words_in_directory(mypath, words_id_text_list)
random.shuffle(words_id_text_list)

alphabet = make_alphabet(words_id_text_list)

print("alphabet is ", alphabet)            
print("words generator will produce", len(words_id_text_list))