#!/usr/bin/env python
# Written by Yantian Zha at Yochan lab, ASU, on April 2016


import os
import os.path

frame_file = '/home/yochan/DL/effect-recognition/dataset/train_framenum.txt'

frame_num = []
for line in open(frame_file):
    frame_num.append(int(line.strip()))

print sum(frame_num)
