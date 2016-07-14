# -*- coding: utf-8 -*-
'''
Project: CSE 575 Statistical Machine Learning
Python code for converting images to video
'''
__author__ = "Jianmei Ye"

import os
import ImageSpitter
import cv2.cv as cv


imgTypes=[".png",".jpg",".bmp"]
fps =15
CODEC = cv.CV_FOURCC('m', 'p', '4', 'v')


# recursively convert the images under ~/images/ into video
def images2Video(root_path,is_rgb = 1):
    for root, dirs, files in os.walk(root_path):
        # print root
        for dir in dirs:
            if dir == 'images':
                video_name =  os.path.basename(root)
                images_path = root + '/' + dir
                video_path = root + '/video'
                ImageSpitter.makeDir(video_path)
                video = cv.CreateVideoWriter(video_path+ '/'+video_name+'.mp4', CODEC, fps, (640, 520), is_rgb)
                frames_list = []
                jpg_list = []
                for jpg in os.listdir(images_path):
                    if '.jpg' in jpg:
                        jpg_list.append(jpg)
                jpg_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
                for jpg in jpg_list:
                    jpg_path = images_path + '/' + jpg
                    frames_list.append(cv.LoadImage(jpg_path))
                for jpg in frames_list:
                    cv.WriteFrame(video, jpg)



def images2Video_single(folder_path,is_rgb = 1):
    for root, dirs, files in os.walk(folder_path):
        video_name = os.path.basename(root)
        images_path = root + '/'
        print "root", root
        video_path = root + '/'
        ImageSpitter.makeDir(video_path)
        print (video_path + '/' + video_name + '.mp4')
        video = cv.CreateVideoWriter(video_path + '/' + video_name + '.mp4', CODEC, fps, (640, 520), is_rgb)
        frames_list = []
        jpg_list = []
        for jpg in os.listdir(images_path):
            if '.jpg' in jpg:
                jpg_list.append(jpg)
        jpg_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        for jpg in jpg_list:
            jpg_path = images_path + '/' + jpg
            frames_list.append(cv.LoadImage(jpg_path))
        for jpg in frames_list:
            cv.WriteFrame(video, jpg)


if __name__ == '__main__':
    # rgb_path = '/Users/JMYE/Desktop/effect-recog/rgb_frames_all'
    # depth_path = '/Users/JMYE/Desktop/effect-recog/depth_frames_all'
    # images2Video(rgb_path,1)
    # images2Video(depth_path, 0)
    single_video = '/Volumes/YANTIAN/EP2/test_small/put_down/put_down_v_05/put_down_v_05_11'
    images2Video_single(single_video)
