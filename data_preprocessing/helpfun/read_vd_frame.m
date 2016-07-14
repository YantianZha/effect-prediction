function numVideoFrames = read_vd_frame(videoFile)
% find the total number of frames in a video
% written by Yantian Zha at Yochan lab, ASU

xyloObj = VideoReader(videoFile);
xylDat = read(xyloObj);
t = size(xylDat);
numVideoFrames = t(4);