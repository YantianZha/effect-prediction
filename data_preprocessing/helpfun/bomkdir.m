function bomkdir(datadir)
% make a directory with checking whether there is an existing directory
% written by Liefeng Bo at University of Washington on January 2011

if exist(datadir,'dir')
   ;
else
   mkdir(datadir);
end

