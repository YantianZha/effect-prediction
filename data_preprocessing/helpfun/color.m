function im = color(input)
% im = color(input)
% convert input image to color.
% written by Liefeng Bo at University of Washington on January 2011

if size(input, 3) == 1
  im(:,:,1) = input;
  im(:,:,2) = input;
  im(:,:,3) = input;
else
  im = input;
end
