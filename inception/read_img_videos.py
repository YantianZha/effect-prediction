#!/usr/bin/env python
# Written by Yantian Zha at Yochan lab, ASU, on April 2016


import os
import os.path

import read_img_video 

class FrameNotMatchException(Exception):
    pass

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

videoDirectory = '/home/yochan/DL/action-recognition-visual-attention-master/dataset/small_UCF11_updated_mpg_test'
imgDirectory = '/home/yochan/DL/action-recognition-visual-attention-master/dataset/genFramesTestSmall'
name_file = '/home/yochan/DL/action-recognition-visual-attention-master/dataset/small_test_filename.txt'
frame_file = '/home/yochan/DL/action-recognition-visual-attention-master/dataset/small_test_framenum.txt'

"""
videoPaths = []
for dirpath, dirnames, filenames in os.walk(directory):
#    for flename in [f for f in filenames]:
    for filename in [f for f in filenames if f.endswith(".mpg")]:
        print os.path.join(dirpath, filename)
	videoPaths.append(os.path.join(dirpath, filename))
"""

fnames = []
subnames = []
frame_num = []
for line in open(name_file):
    fnames.append(videoDirectory+'/'+line.strip())
    subnames.append(line.strip())

for line in open(frame_file):
    frame_num.append(line.strip())
#print fnames
it = 0
for video in fnames:
    it += 1
    tgtPath = os.path.splitext((imgDirectory+'/'+subnames[it-1]))[0]
    _ = create_dir(tgtPath)
    t = read_img_video.read_video(video, tgtPath)
    if t != int(frame_num[it-1]):
	raise FrameNotMatchException('The number of frames generated does not match framenum.txt')
    else:
        print 'Have processed %dth video =^_^=' %it


