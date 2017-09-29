"""
Aman Arya

This file pickles all the current database of papers in /Lantern_v2/Data. Shows how to load data as well.
"""

import os
import pickle
from io import open

labels = ['Sexual Harrassment', 'Bullying', 'Drug', 'Vandalism']
labels = sorted(labels)

def save(dir):

    label_list = os.listdir(dir)
    if 'B_data.pickle' in label_list:
        label_list.remove('B_data.pickle')
        label_list.remove('fight_assault_data')
    obj = []
    for label in label_list:
        fpath = os.path.join(dir, label)
        list = os.listdir(fpath)
        for file in list:
            print(file)
            f = open(fpath+'/'+file, 'rb')
            obj.append(f.read())

    pickle.dump([obj, labels], open(dir+'B_data.pickle', 'wb'))

save(os.path.dirname(os.getcwd())+'/Data/')

# Loading example

data_txt, labels = pickle.load(open(os.path.dirname(os.getcwd())+'/Data/B_data.pickle', 'rb'))
