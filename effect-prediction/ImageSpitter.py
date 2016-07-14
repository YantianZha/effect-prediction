'''
Project: CSE 575 Statistical Machine Learning
Python code for processing Kinect raw data
'''
__author__ = "Jianmei Ye"

import cv2.cv as cv
import os

imgTypes=[".png",".jpg",".bmp"]

############# change your path here ###########################

# path for splited Images and video for each action with effect
rgb_path_all = "/Users/JMYE/Desktop/effect-recog/rgb_frames_all/"
depth_path_all = "/Users/JMYE/Desktop/effect-recog/depth_frames_all/"

# path for splited Images and video for each action without effect
rgb_path = "/Users/JMYE/Desktop/effect-recog/rgb_frames/"
depth_path = "/Users/JMYE/Desktop/effect-recog/depth_frames/"


# path for raw frames from Kinect
input_frame_path = "/Users/JMYE/Downloads/records/"

##############################################################

# create target dir if not exists
def makeDir(action_dir):
    if not os.path.exists(action_dir):
        # os.removedirs(action_dir)
        os.makedirs(action_dir)
        # print 'new dir is made:', action_dir




# split the  raw frames from Kinect into rgb and depth
def splitImages(idx,rgb_frames,depth_frames):
    d = input_frame_path + str(idx)
    pngCount = 0
    for root, dirs, files in os.walk(d):
        for afile in files:
            ffile = root + "/" + afile
            if ffile[ffile.rindex("."):].lower() in imgTypes:
                img = cv.LoadImage(ffile)
                # split rgb frames
                if pngCount < 10:
                    count_name = pngCount%10
                else:
                    count_name = pngCount
                cv.SetImageROI(img, (0, 0, img.width / 2, img.height))
                cv.SaveImage(rgb_frames+ '/'+ str(count_name)+'.jpg', img)
                # split depth frames
                cv.SetImageROI(img, (img.width / 2, 0, img.width / 2, img.height))
                cv.SaveImage(depth_frames+'/' + str(count_name)+'.jpg', img)
                # remove temp files
                os.remove(ffile)
                pngCount += 1






if __name__ == "__main__":

    ############ change the action name here #########
    # names for action and serials
    action_name = 'pour_water'
    action_serial = '_v_'
    serial_num1 = '03'
    ##################################################
    rgb_path_all += action_name+'/'
    depth_path_all += action_name+'/'
    for i in range(1,41):
        if i < 10:
            serial_num2 = str(0) + str(i)
        else:
            serial_num2 = str(i)
        action_serial_num = action_name + action_serial + serial_num1 + '_' + serial_num2
        # path for final output of images and videos with effect
        rgb_action_records_path_all_1 = rgb_path_all + action_name+ action_serial + serial_num1
        rgb_action_records_path_all_2 = rgb_action_records_path_all_1+ '/' + action_serial_num
        depth_action_records_path_all_1 = depth_path_all + action_name+ action_serial + serial_num1
        depth_action_records_path_all_2 = depth_action_records_path_all_1+'/'+action_serial_num


        rgb_images = rgb_action_records_path_all_2
        depth_images = depth_action_records_path_all_2
        makeDir(rgb_action_records_path_all_1)
        makeDir(rgb_images)
        makeDir(depth_action_records_path_all_1)
        makeDir(depth_images)
        splitImages(i,rgb_images,depth_images)

