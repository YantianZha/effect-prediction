# Inception

This repository contains a reference pre-trained network for the Inception
model, complementing the Google publication.

Going Deeper with Convolutions, CVPR 2015.
Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.

You can view "inception.ipynb" directly on GitHub, or clone the
repository, install dependencies listed in the notebook and play with code
locally.

You may also be interested in the [Multibox](https://github.com/google/multibox)
approach that uses the Inception architecture for object detection, also
available on GitHub.

Disclaimer: this is not an official Google product (experimental or otherwise).


# From Yantian
1. visualize inception network
   runipy -o inception.ipynb
   ipython nbconvert inception.ipynb --to html

   sample output: inception_yantian.html

2. get 7*7*1024 from inception
   ipython 771024.py

   Then find the .h5 file in the current folder

3. convert videos to frames (images)
   python read_img_videos.py

