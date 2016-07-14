function [videolabel, videopath, rl_videopath, framenum] = get_im_label(imdir)
% generate the image paths and the corresponding image labels
% written by Liefeng Bo at University of Washington on January 2011
% Modified by Yantian Zha at Yochan lab, ASU, on April 2016

if nargin < 2
   subdir = dir_bo(imdir);
   %subdir = imdir;
   it = 0;
   kt = 0;
   for i = 1:length(subdir)
       datasubdir{i} = [imdir '/' subdir(i).name];
       dataname = dir_bo(datasubdir{i});
       for j = 1:size(dataname,1)
           kt = kt+1;
           videopath{1,kt} = [datasubdir{i} '/' dataname(j).name];
           parts = strsplit(datasubdir{i}, '/');
           DirPart = parts{end-1};
           rl_videopath{1,kt} = [DirPart '/' subdir(i).name '/' dataname(j).name];            
           videolabel(1,kt) = it;
               
           % get frame number
           v = sprintf('%s', videopath{:,kt});
           u = dir_bo(v);
           %http://www.mathworks.com/matlabcentral/answers/106642-how-do-i-count-the-number-of-csv-files-in-a-folder
           framenum(1,kt) = length(u);
           %framenum(1,kt) = read_vd_frame(v);
           %framenum(1,kt) = read_vd_frame(v)-1;% The last frame is empty image
           fprintf('Have processed %ith video, which has %i frames =^_^=\n',kt,framenum(1,kt));           
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



