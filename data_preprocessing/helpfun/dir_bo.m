function imname = dir_bo(datadir)
% remove rootdir
% written by Liefeng Bo at University of Washington on January 2011

imname = dir(datadir);
imname(1:2) = [];

