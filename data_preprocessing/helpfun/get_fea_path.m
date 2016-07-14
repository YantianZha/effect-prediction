function feapath = get_fea_dir(feadir)
% generate the image paths and the corresponding image labels
% subdirectory
% written by Liefeng Bo at University of Washington on January 2011

feaname = bodir(feadir);
if length(feaname)
   for i = 1:length(feaname)
       % generate image paths
       feapath{1,i} = [feadir '/' feaname(i).name];
   end
else
   feapath = [];
end

