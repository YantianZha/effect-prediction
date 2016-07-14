from IPython import display
from matplotlib import pyplot
import numpy as np
import os
import sys

# Make sure that you set this to the location your caffe2 library lies.
caffe2_root = '/home/user/caffe2/'
sys.path.insert(0, os.path.join(caffe2_root, 'gen'))

# After setting the caffe2 root path, we will import all the caffe2 libraries needed.
from caffe2.proto import caffe2_pb2
from pycaffe2 import core, net_drawer, workspace, visualize