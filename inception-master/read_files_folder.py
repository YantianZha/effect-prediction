#!/usr/bin/env python
#http://stackoverflow.com/questions/954504/how-to-get-files-in-a-directory-including-all-subdirectories

import os

#directory = '/Users/yantian/Documents/DL_data'
directory = '/Users/yantian/Google Drive/SML575/Project/data_preprocessing/UCF11_updated_mpg_small'
#os.walk(directory)

#print next(os.walk('.'))[1]

import os
import os.path

for dirpath, dirnames, filenames in os.walk(directory):
#    for filename in [f for f in filenames]:
    print dirnames
    for filename in [f for f in filenames if f.endswith(".mpg")]:
        print os.path.join(dirpath, filename)









