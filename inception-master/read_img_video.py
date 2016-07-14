"""
Used for reading each frame from a video file, which is then saved as an image
Ref: http://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
Modified by Yantian Zha at Yochan lab, ASU, on April 2016
"""

import cv2
def read_video(video_name, directory):
    vidcap = cv2.VideoCapture(video_name)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        if success == True:
            #print 'Read a new frame: ', success
    	    cv2.imwrite(directory + '/%d.jpg' % count, image)     # save frame as JPEG file
    	    count += 1
    print "The number of frames:\n", count
    return count

if __name__ == '__main__':
    directory = '/home/yochan/DL/inception-master/dataset/test/seq1'
    video_name = '/home/yochan/DL/action-recognition-visual-attention-master/dataset/UCF11_updated_mpg_train/horse_riding/v_riding_15/v_riding_15_07.mpg'
    read_video(video_name, directory)


