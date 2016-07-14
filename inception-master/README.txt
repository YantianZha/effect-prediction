1. visualize inception network
   runipy -o inception.ipynb
   ipython nbconvert inception.ipynb --to html

   sample output: inception_yantian.html

2. get 7*7*1024 from inception
   ipython 771024.py

   Then find the .h5 file in the current folder

3. convert videos to frames (images)
   python read_img_videos.py

