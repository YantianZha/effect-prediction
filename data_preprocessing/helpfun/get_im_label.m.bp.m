function [videolabel, videopath, rl_videopath, framenum] = get_im_label(imdir)
% generate the image paths and the corresponding image labels
% written by Liefeng Bo at University of Washington on January 2011
% editted by Yantian Zha at Yochan lab, ASU

if nargin < 2
   subdir = dir_bo(imdir);
   %subdir = imdir;
   it = 0;
   kt = 0;
   for i = 1:length(subdir)
       datasubdir{i} = [imdir '/' subdir(i).name];
       dataname = dir_bo_rm3(datasubdir{i});
       for j = 1:size(dataname,1)
 	       % generate image paths
           impath = [datasubdir{i} '/' dataname(j).name];
           subvideodir = dir_bo(impath);
           for k = 1:length(subvideodir)
               kt = kt+1;
               videopath{1,kt} = [impath '/' subvideodir(k).name];
               rl_videopath{1,kt} = [dataname(j).name '/' subvideodir(k).name];
               videolabel(1,kt) = it;
               
               % get frame number
               v = sprintf('%s\n', videopath{:,kt});
               framenum(1,kt) = read_vd_frame(v);
           end
       end
       it = it+1;
   end
     

else
   subdir = dir_bo(imdir);
   it = 0;
   for i = 1:length(subdir)
       datasubdir{i} = [imdir subdir(i).name];
       dataname = dir([datasubdir{i} '/*' subname]);
       for j = 1:size(dataname,1)
           it = it + 1;
           % generate image paths
           impath{1,it} = [datasubdir{i} '/' dataname(j).name];
           imlabel(1,it) = i;
       end
   end
end



