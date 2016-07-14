% train_filename.txt
% train_labels.txt
% train_framenum.txt
% first remove those files if you have in the target folder
% removed a bad video: UCF11_updated_mpg/basketball/v_shooting_24/v_shooting_24_01.mpg
% unzip/delete everything outsiding your mac
% written by Yantian Zha at Yochan lab, ASU, on March 2016

clear;

% add paths
addpath('./helpfun/');
% addpath('./UCF11_updated_mpg/');
%addpath_recurse('/home/yochan/Downloads/UCF11_updated_mpg');
%addpath('/home/yochan/Downloads/UCF11_updated_mpg/basketball/v_shooting_01')

%addpath(genpath('./UCF11_updated_mpg/')); 
% if isempty(strfind(path,'/home/yochan/Downloads/UCF11_updated_mpg;'))
%     addpath('/home/yochan/Downloads/UCF11_updated_mpg')
% end


videoDir = '/Users/yantian/Documents/DL_data/effect-recog/rgb_dataset';
%videoDir = '/Users/yantian/Documents/DL_data/action-recognition-visual-attention/UCF11_updated_mpg_small';
[labels, videoPath, rl_videoPath, framenum] = get_im_label(videoDir);

t = rl_videoPath(1,:);
v = framenum(1,:);

fid = fopen('/Users/yantian/Documents/DL_data/effect-recognition/train_filename.txt','w');
fid1 = fopen('/Users/yantian/Documents/DL_data/effect-recognition/train_labels.txt','w');
fid2 = fopen('/Users/yantian/Documents/DL_data/effect-recognition/train_framenum.txt','w');
formatSpec = '%s\n';
formatSpec1 = '%d\n';
n = size(t);
for i=1:n(2)
  fprintf(fid, formatSpec, t{i});
  fprintf(fid1, formatSpec1, labels(i));
  fprintf(fid2, formatSpec1, v(i));
end
fclose(fid);
fclose(fid1);
fclose(fid2);