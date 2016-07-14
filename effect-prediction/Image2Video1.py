'''
Project: CSE 575 Statistical Machine Learning
Python code for converting images to video
'''
__author__ = "Jianmei Ye"

import os
import ImageSpitter
import cv2.cv as cv


imgTypes=[".png",".jpg",".bmp"]
fps = 60
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
                for jpg in os.listdir(images_path):
                    if '.jpg' in jpg:
                        jpg_path = images_path + '/'+jpg
                        print "CC", jpg_path
                        frames_list.append(cv.LoadImage(jpg_path))
                print frames_list
                for jpg in frames_list:
                    cv.WriteFrame(video,jpg)





if __name__ == '__main__':
    rgb_path = '/home/yochan/DL/effect-recognition/dataset/test'
   # depth_path = '/Users/JMYE/Desktop/effect-recog/depth_frames_all'
    images2Video(rgb_path,1)
   # images2Video(depth_path, 0)
