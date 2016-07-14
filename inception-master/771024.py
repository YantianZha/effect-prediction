# coding: utf-8
# reference: https://github.com/google/inception/blob/master/inception.ipynb
# Modified by Yantian Zha at Yochan lab, ASU, on April 2016
# Usage: ipython 771024.py

from IPython import display
from matplotlib import pyplot
import numpy as np
import os
import sys
get_ipython().magic(u'matplotlib auto')	# Yantian
# Another possible solution: http://stackoverflow.com/questions/21366672/run-python-script-in-ipython-with-inline-embedded-plots

# Make sure that you set this to the location your caffe2 library lies.
caffe2_root = '/home/yochan/DL/caffe2/'
sys.path.insert(0, os.path.join(caffe2_root, 'gen'))

# After setting the caffe2 root path, we will import all the caffe2 libraries needed.
from caffe2.proto import caffe2_pb2
#from pycaffe2 import core, net_drawer, workspace, visualize	# Yantian
from caffe2.python import core, net_drawer, workspace, visualize


# (note: if you see warning messages above, it might be coming from some dependency libraries like pydot, and you do not need to worry about it.)
# 
# Loading the model
# -------------------
# First, let's load the inception model. It is composed of two parts: one defines the network architecture, and one provides the parameters that the network uses.

# In[2]:

# net is the network definition.
net = caffe2_pb2.NetDef()
net.ParseFromString(open('/home/yochan/DL/inception-master/inception_net.pb').read())
# tensors contain the parameter tensors.
tensors = caffe2_pb2.TensorProtos()
tensors.ParseFromString(open('/home/yochan/DL/inception-master/inception_tensors.pb').read())

# Yantian, illustrated by multibox.py
for op in net.op:
    if op.type == 'MaxPool':
        del op.output[1:]

# Let's take a look at the model architecture using Caffe2's net_drawer. Warning: this network is very deep, so you might need to scroll down to show the whole network. To assist your scroll, we are showing the flow direction from the top to the bottom, so the input image is filled into the top of the graph. Each node in this graph is an operator (loosely equivalence of a layer). We are showing activations like ReLU as layers as well. The type of the operators are shown in the name of each node.
# 
# The graph is quite tall, so you may need to scroll down.

# In[3]:

# Note that the following line hides the intermediate blobs and only shows the operators.
# If you want to show all the blobs as well, use the commented GetPydotGraph line.
graph = net_drawer.GetPydotGraphMinimal(net.op, name="inception", rankdir='TB')
#graph = net_drawer.GetPydotGraph(net.op, name="inception", rankdir='TB')

print 'Visualizing network:', net.name
display.Image(graph.create_png(), width=200)


# If the image shown above is too small, try right-click it and select "Open image in new tab". This will show a larger version. You may notice the two "side towers" branching out, and ending with names "softmax0" and "softmax1": as described in our paper, such side towers exist to help making models converging faster. In practice, when you are doing inference (as in this example, you do not need to use these side towers. We are leaving them here just for completeness, since we do not care too much about speed for this example.

# Instantiate the Model in Caffe2
# ------------------------------------
# If you are familiar with Caffe, you may notice that Caffe2 instantiates a model slightly differently - it deals with the network and the parameters separately, and has a specific device_option field that specifies where to run the model. This allows one to have more fine-grained control over things.

# In[4]:

DEVICE_OPTION = caffe2_pb2.DeviceOption()
# Let's use CPU in our example.
DEVICE_OPTION.device_type = caffe2_pb2.CPU

# If you have a GPU and want to run things there, uncomment the below two lines.
# If you have multiple GPUs, you also might want to specify a gpu id.
#DEVICE_OPTION.device_type = caffe2_pb2.CUDA
#DEVICE_OPTION.cuda_gpu_id = 0

# Caffe2 has a concept of "workspace", which is similar to that of Matlab. Each workspace
# is a self-contained set of tensors and networks. In this case, we will just use the default
# workspace so we won't dive too deep into it.
workspace.SwitchWorkspace('default')

# First, we feed all the parameters to the workspace.
for param in tensors.protos:
    workspace.FeedBlob(param.name, param, DEVICE_OPTION)
# The network expects an input blob called "input", which we create here.
# The content of the input blob is going to be fed when we actually do
# classification.
workspace.CreateBlob("input")
# Specify the device option of the network, and then create it.
net.device_option.CopyFrom(DEVICE_OPTION)
workspace.CreateNet(net)


# Now let's do some classification on images. There are a lot of tricks that one can employ in image pre-processing, and one often employs multiple cropping strategies in order to increase testing performance. Here we just do a very simple example: we resize the center square region of the image to 224x224, which is the input size expected by the network, and subtract the mean value 117 (which is a hyperparameter we used in training), and directly feeds it to the network. For simplicity, we will write a function that wraps all these.

# In[5]:

def ClassifyImageWithInception(image_file, show_image=True, output_name="softmax2"):
    from skimage import io, transform
    img = io.imread(image_file)
    # Crop the center
    shorter_edge = min(img.shape[:2])
    crop_height = (img.shape[0] - shorter_edge) / 2
    crop_width = (img.shape[1] - shorter_edge) / 2
    cropped_img = img[crop_height:crop_height + shorter_edge, crop_width:crop_width + shorter_edge]
    # Resize the image to 224 * 224
    resized_img = transform.resize(cropped_img, (224, 224))
    if show_image:
        pyplot.imshow(resized_img)
    # normalize the image and feed it into the network. The network expects
    # a four-dimensional tensor, since it can process images in batches. In our
    # case, we will basically make the image as a batch of size one.
    normalized_img = resized_img.reshape((1, 224, 224, 3)).astype(np.float32) * 256 - 117
    workspace.FeedBlob("input", normalized_img, DEVICE_OPTION)
    workspace.RunNet("inception")
    return workspace.FetchBlob(output_name)

# We will also load the synsets file where we can look up the actual words for each of our prediction.
synsets = [l.strip() for l in open('synsets.txt').readlines()]


# We will use a nice Dalmatian image to show how we run a simple prediction.

# In[6]:

predictions = ClassifyImageWithInception("dog.jpg").flatten()
idx = np.argmax(predictions)
print 'Prediction: %d, synset %s' % (idx, synsets[idx])


# Let's also show the top five predictions that we had, and the distribution of scores. A little bit more information for the keen readers: Our model is trained with a dummy class at index 0, which never gets used. This is in order to be consistent with the Matlab indexing convention, which starts with 1. The total number of predictions is actually 1008 classes. In case you are wondering, class 1001-1007 are also dummy, we used 1008 instead of 1001 just to make numerical optimization better for historical reasons. In practice, only the indices 1 to 1000 are going to have nontrivial predictions, since class 0 and 1001-1007 never have positive examples during training.

# In[7]:

indices = np.argsort(predictions)
print 'Top five predictions:'
for idx in indices[:-6:-1]:
    print '%6d (prob %.4f) synset %s' % (idx, predictions[idx], synsets[idx])
pyplot.plot(predictions)

#visualize.PatchVisualizer vis	# Yantian
# We can also inspect the intermediate results and parameters of the network, similar to what one can do in Caffe. Let's show the first layer filters, for example.

# In[8]:

filters = workspace.FetchBlob('conv2d0_w')
# We normalize the filters for visualization.
filters = (filters - filters.min()) / (filters.max() - filters.min())
_ = visualize.PatchVisualizer(gap=2).ShowMultiple(filters, bg_func=np.max)


# And let's show the first layer outputs. Since there are 64 output channels, we will only show the first 16 channels. The output are real-valued, but we will use pyplot's "hot" colormap so we can see the values better than grayscale.

# In[9]:

conv2d0 = workspace.FetchBlob('conv2d0')
print 'First layer output shape:', conv2d0.shape
_ = visualize.PatchVisualizer(gap=10).ShowChannels(conv2d0[0, :, :, :16], bg_func=np.max, cmap=pyplot.cm.hot)


# Such activations go on and on, and for simplicity, let's just show the output of the last convolution, right before the average pooling layer. The layer has 1024 channels, so we will be only showing the first 144 channels.

# In[10]:

mixed5b = workspace.FetchBlob('mixed5b')
print 'First layer output shape:', mixed5b.shape
#visualize.ShowChannels(mixed5b[0, :, :, :144], bg_func=np.max, cmap=pyplot.cm.hot)
#vis.ShowChannels(mixed5b[0, :, :, :144], bg_func=np.max, cmap=pyplot.cm.hot)

""" 
# Test
video_dir = '/home/yochan/DL/inception-master/dataset/test/sequences'
#dataset = []
dataset = {}
for dirpath, dirnames, filenames in os.walk(video_dir):
    for filename in [f for f in filenames if f.endswith(".jpg")]:
        #print "Processing file:\n", filename
        s_fname = os.path.splitext(filename)[0]
        print "Processing file:\n", s_fname
        predictions = ClassifyImageWithInception(os.path.join(dirpath, filename)).flatten()
        #idx = np.argmax(predictions)
        #print 'Prediction: %d, synset %s' % (idx, synsets[idx])
        mixed5b = workspace.FetchBlob('mixed5b')
        #print 'First layer output shape:', mixed5b.shape
        #dataset.append((int(s_fname), mixed5b))
        dataset[int(s_fname)] = mixed5b
        #print type(mixed5b)

#dataset = sorted(dataset)  # sorted with 1st item
import collections
dataset = collections.OrderedDict(sorted(dataset.items()))
"""

import os
import collections

imgDirectory = '/home/yochan/DL/effect-recognition/dataset/EP2/test_effect/pour'
name_file = '/home/yochan/DL/effect-recognition/dataset/EP2/test_effect/test_ef_pour_filename.txt'
frame_file = '/home/yochan/DL/effect-recognition/dataset/EP2/test_effect/test_ef_pour_framenum.txt'

subnames = []
frame_num = []
fullnames = []

class FrameNotMatchException(Exception):
    pass

for line in open(name_file):
    subnames.append(os.path.splitext(line.strip())[0])
    fullnames.append(imgDirectory+'/'+os.path.splitext(line.strip())[0])
#print fullnames
for line in open(frame_file):
    frame_num.append(int(line.strip()))

totalFrames = sum(frame_num)
#print totalFrames

def cnn_processing(fileNm):
    predictions = ClassifyImageWithInception(fileNm).flatten()
    return workspace.FetchBlob('mixed5b')

it = 0
vt = 0

def getDataset(it, vt, old_dataset):
    """
    Input: it (num_frames), vt (num_videos), old_dataset (dataset generated from last iteration)
    Output: dict[index] = numpy.array, which is the output from inception, 7*7*1024
    """
    dataset = {}

    if it >= totalFrames:
        if old_dataset.keys()[-1] != it-1:
            raise FrameNotMatchException('There are %d frames but the max id of processed img is %d.' %(it, old_dataset.keys()[-1]))
        else:
            return old_dataset

    else:
        print 'Processing %dth video #^_^#' %vt
        for dirpath, dirnames, filenames in os.walk(fullnames[vt]):
            for filename in [f for f in filenames if f.endswith(".jpg")]:
                s_fname = os.path.splitext(filename)[0]
                print "Processing %d file:\n" %it, (fullnames[vt], s_fname)
                if vt == 0:
                    dataset[int(s_fname)] = cnn_processing(os.path.join(dirpath, filename))
                elif vt > 0:
                    dataset[int(s_fname)+sum(frame_num[:vt])] = cnn_processing(os.path.join(dirpath, filename))
                it += 1

        dataset = collections.OrderedDict(sorted(dataset.items()))
        vt += 1

        # merge two dictionaries together
        old_dataset.update(dataset)
        return getDataset(it, vt, old_dataset)

dataset = getDataset(it, vt, {})
print "There're totally %d images" %len(dataset)
import h5py

import datetime

f = h5py.File('/home/yochan/DL/effect-recognition/dataset/EP2/XX.h5', 'w')
dset = f.create_dataset("features", (len(dataset.keys()), 7, 7, 1024), maxshape=(None, None, None, None))
#dset = f.create_dataset("features", (len(dataset.keys()), 7*7*1024), maxshape=(None, None))

for ky, val in dataset.iteritems():
#    print val
    print ky
    dset[ky,...] = val
#    f['ky'] = val

f.close()

print 'done!'



