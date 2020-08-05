#!/usr/bin/env python
# coding: utf-8

# # Jupyter Notebook Object Detection Sample Tutorial

# # Preface: How to Use this Jupyter Notebook Tutorial
# Below is a sequence of boxes referred to as "cells".  Each cell will contain text, like this one, or Python code that may be executed as part of this tutorial.  As you go through this turoial, please note the following:
# 
# ### Running the Tutorial
# You can always use either the "Run" button at the top or **Shift+Enter** to execute a selected cell, starting with this one, and then automatically move to the next cell.
# 
#    **Note**: If you happen to press just **Enter**, you will enter the editing mode for the cell.  To exit and continue, use **Shift+Enter**.  
#    
# Unless stated otherwise, the cells containing code within this tutorial **MUST** be executed in sequence.
# 
# You may save the tutorial at any time which will save the output, but not state.  Saved Jupyter Notebooks will save sequence numbers which may make a cell appear executed when it has not been for the new session.  Because state is not saved, re-opening or restarting a Jupyter Notebook will required re-executing all the executable steps starting from the beginning.
# 
# If for any reason you need to restart the tutorial from the beginning, you may reset the state of the Jupyter Notebook and clear all output by using the menu at the top by selecting **Kernel->"Restart and Clear Output"**
# 
# ### Cells Containing Executable Code
# Executable cells will have "In [n]:" to the left of the cell:
# - If 'n' is blank (no number), it means that the cell has not yet been executed.  
# - If 'n' is '*', it means that the cell is currently executing.  
# - Once a cell is done executing, 'n' will appear as a number incrementing with each cell execution to indicate where in the sequence the cell has been executed.  Any output (e.g. print()'s) from the code will appear below the cell.
#     - Note: If you need to stop a cell during execution, for example during a long video, you can use the "Stop" button at the top (square to the right of the "Run" button).  After stopping the cell, you may re-execute it if needed.

# # Prerequisites
# Before going through this tutorial, please be sure that:
# - All files from the .zip file containing the tutorial are present and in the same directory.  The required files are:
#     - tutorial_object_detection_ssd.ipynb - This Jupyter Notebook
#     - mobilenet-ssd/mobilenet-ssd.bin and mobilenet-ssd.xml - The IR files for the inference model created using Model Optimizer
#     - mobilenet-ssd/labels.txt - mapping of numerical labels to text strings
#     - face.jpg - test image
#     - car.bmp - test image
#     - libcpu_extension.so - pre-compiled CPU extension library
#     - doc_*.png - images used in the documentation
# - Optional: URL to image or user's video to run inference on
# 
# **Note:** It is assumed that the server this tutorial is being run on has Jupyter Notebook, OpenVINO toolkit, and other required libraries already installed.  If you download or copy to a new server, the tutorial may not run.  

# # Introduction
# The purpose of this tutorial is to examine a sample application that was created using the [Intel® Distribution of Open Visual Inference & Neural Network Optimization (OpenVINO™) toolkit](https://software.intel.com/openvino-toolkit).  This tutorial will go step-by-step through the necessary steps to demonstrate object detection on images and video.  Object detection is performed using a pre-trained network and running it using the OpenVINO™ toolkit Inference Engine.  Inference will be executed using the same CPU(s) running this Jupyter Notebook.
# 
# The pre-trained model to be used for object detection is the ["mobilenet-ssd"](https://github.com/chuanqi305/MobileNet-SSD) which has already been converted to the necessary Intermediate Representation (IR) files needed by the Inference Engine (Conversion is not covered here, please see the [Intel® Distribution of OpenVINO™ toolkit](https://software.intel.com/en-us/openvino-toolkit) documentation for more details).  The model is capable of detecting 
# different objects including: airplane, bicycle, bird, boat, bus, car, cat, dog, horse, person and more (see mobilenet-ssd/labels.txt file for complete list).  

# # Key Concepts
# Before going into the samples in the tutorial steps, first we will go over some key concepts that will be covered in this tutorial.

# ## OpenVINO™ Toolkit Overview and Terminology 
# 
# Let us begin with a brief overview of the OpenVINO™ toolkit and what this tutorial will be covering.  The OpenVINO™ toolkit enables the quick deployment of convolutional neural networks (CNN) for heterogeneous execution on Intel® hardware while maximizing performance. This is done using the Intel® Deep Learning Deployment Toolkit (Intel® DL Deployment Toolkit) included within the OpenVINO™ toolkit with its main components shown below.
# 
# ![image alt text](./doc_openvino_overview_image.png)
# 
# The basic flow is:
# 
# 1. Use a tool, such as Caffe, to create and train a CNN inference model
# 
# 2. Run the created model through Model Optimizer to produce an optimized Intermediate Representation (IR) stored in files (.bin and .xml) for use with the Inference Engine
# 
# 3. The User Application then loads and runs models on devices using the Inference Engine and the IR files  
# 
# This tutorial will focus on the last step, the User Application and using the Inference Engine to run a model on CPU.
# 
# ### Using the Inference Engine
# 
# Below is a more detailed view of the User Application and Inference Engine:
# 
# ![image alt text](./doc_inference_engine_image.png)
# 
# The Inference Engine includes a plugin library for each supported device that has been optimized for the Intel® hardware device CPU, GPU, and Myriad.  From here, we will use the terms "device" and “plugin” with the assumption that one infers the other (e.g. CPU device infers the CPU plugin and vice versa).  As part of loading the model, the User Application tells the Inference Engine which device to target which in turn loads the associated plugin library to later run on the associated device. The Inference Engine uses “blobs” for all data exchanges, basically arrays in memory arranged according the input and output data of the model.
# 
# #### Inference Engine API Integration Flow
# 
# Using the Inference Engine API follows the basic steps outlined briefly below.  The API objects and functions will be seen later in the sample code.
# 
# 1. Load the plugin
# 
# 2. Read the model IR
# 
# 3. Load the model into the plugin
# 
# 6. Prepare the input
# 
# 7. Run Inference
# 
# 8. Process the output
# 
# More details on the Inference Engine can be found in the [Inference Engine Development Guide](https://software.intel.com/inference-engine-devguide)

# ## Input Preprocessing
# 
# Often, the dimensions of the input data does not match the required dimensions of the input data for the inference model.  A common example is an input video frame.  Before the image may be input to the inference model, the input must be preprocessed to match the required dimensions for the inference model as well as channels (i.e. colors) and batch size (number of images present).  The basic step performed is to resize the frame from the source dimensions to match the required dimensions of the inference model’s input, reorganizing any dimensions as needed.
# 
# This tutorial and the many samples in the OpenVINO™ toolkit use OpenCV to perform resizing of input data.  The basic steps performed using OpenCV are:
# 
# 1.  Resize image dimensions form image to model's input W x H:
#     frame = cv2.resize(image, (w, h))
#    
# 2. Change data layout from (H x W x C) to (C x H x W)
#     frame = frame.transpose((2, 0, 1))  
# 
# 3. Reshape to match input dimensions
#     frame = frame.reshape((n, c, h, w))

# # Sample Application
# Now we will begin going through the sample application.

# ## Importing Python Modules
# Here we begin by importing all the Python modules that will be used by the sample code:
# - [os](https://docs.python.org/3/library/os.html#module-os) - Operating system specific module (used for file name parsing)
# - [cv2](https://docs.opencv.org/trunk/) - OpenCV module
# - [time](https://docs.python.org/3/library/time.html#module-time) - time tracking module (used for measuring execution time)
# - [openvino.inference_engine](https://software.intel.com/en-us/articles/OpenVINO-InferEngine) - import the IENetwork and IEPlugin objects
# - [matplotlib](https://matplotlib.org/) - import pyplot used for displaying output images

# In[1]:


import os
import cv2
import time
from openvino.inference_engine import IENetwork, IEPlugin
#get_ipython().run_line_magic('matplotlib', 'inline')
#from matplotlib import pyplot as plt
print("Imported Python modules.")


# ## Configuration Parameters
# Here we will create and set the following configuration parameters used by the sample:  
# * *model_xml* - Path to the .xml IR file of the trained model to use for inference
# * *model_bin* - Path to the .bin IR file of the trained model to use for inference (derived from *model_xml*)
# * *input_path* - Path to input image
# * *cpu_extension_path* - Path to a shared library with CPU extension kernels for custom layers not already included in plugin
# * *device* - Specify the target device to infer on,  CPU, GPU, FPGA, or MYRIAD is acceptable, however the device must be present.  For this tutorial we use "CPU" which is known to be present.
# * *labels_path* - Path to labels mapping file used to map outputted integers to strings (e.g. 7="car")
# * *prob_threshold* - Probability threshold for filtering detection results
# 
# We will set all parameters here only once except for *input_path* which we will change later to point to different images and video.

# In[2]:


# model IR files
model_xml = "./mobilenet-ssd/mobilenet-ssd.xml"
model_bin = os.path.splitext(model_xml)[0] + ".bin" # create IR .bin filename from path to IR .xml file

# input image file
input_path = "car.bmp"

# CPU extension library to use
cpu_extension_path = "libcpu_extension.so"

#cpu_extension_path = "/home/ojuan/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.so"

# device to use
device = "CPU"

# output labels 
labels_path = "./mobilenet-ssd/labels.txt"

# minimum probabilty threshold to detect an object
prob_threshold = 0.5

print("Configuration parameters settings:"
     "\n\tmodel_xml=", model_xml,
      "\n\tmodel_bin=", model_bin,
      "\n\tinput_path=", input_path,
      "\n\tcpu_extension_path=", cpu_extension_path, 
      "\n\tdevice=", device, 
      "\n\tlabels_path=", labels_path, 
      "\n\tprob_threshold=", prob_threshold)


# ## Create Plugin for Device
# Here we create a plugin object for the specified device using IEPlugin().  
# If the plugin is for a CPU device, and the *cpu_extensions_path* variable has been set, we load the extensions library. 

# In[3]:


# create plugin for device
plugin = IEPlugin(device=device)
print("A plugin object has been created for device [", plugin.device, "]\n")

# if the device is CPU and a path to an extension library is set, load the extension library 
if cpu_extension_path and 'CPU' in device:
    plugin.add_cpu_extension(cpu_extension_path)
    print("CPU extension [", cpu_extension_path, "] has been loaded")


# ## Create Network from Model IR files
# Here we create a *IENetwork* object and load the model's IR files into it.  After loading the model, we check to make sure that all the model's layers are supported by the plugin we will use.  We also check to make sure that the model's input and output are as expected for later when we run inference.

# In[4]:


# load network from IR files
net = IENetwork(model=model_xml, weights=model_bin)
print("Loaded model IR files [",model_bin,"] and [", model_xml, "]\n")

# check to make sure that the plugin has support for all layers in the loaded model
supported_layers = plugin.get_supported_layers(net)
not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
if len(not_supported_layers) != 0:
    print("ERROR: Following layers are not supported by the plugin for specified device {}:\n {}".
             format(plugin.device, ', '.join(not_supported_layers)))
    if not cpu_extension_path:
        print("       Please try specifying the cpu extensions library path by setting the 'cpu_extension_path' variable")
    assert 0 == 1, "ERROR: Missing support for all layers in th emodel, cannot continue."

# check to make sue that the model's input and output are what is expected
assert len(net.inputs.keys()) == 1, "ERROR: This sample supports only single input topologies"
assert len(net.outputs) == 1, "ERROR: This sample supports only single output topologies"
print("SUCCESS: Model IR files have been loaded and verified")
print(net.inputs.keys())
print(net.outputs)


# ## Load the Model into the Device Plugin
# Here we load the model network into the plugin so that we may run inference.  *exec_net* will be used later to actually run inference.  After loading, we store the names of the input (*input_blob*) and output (*output_blob*) blobs to use when accessing the input and output blobs of the model.  Lastly, we store the model's input dimensions into the following variables:
# - *n* = input batch size
# - *c* = number of input channels (here 1 channel per color R,G, and B)
# - *h* = input height
# - *w* = input width

# In[ ]:


# load the model into the plugin
exec_net = plugin.load(network=net, num_requests=2)

# store name of input and output blobs
input_blob = next(iter(net.inputs))
output_blob = next(iter(net.outputs))

# read the input's dimensions: n=batch size, c=number of channels, h=height, w=width
n, c, h, w = net.inputs[input_blob].shape
print("Loaded model into plugin.  Model input dimensions: n=",n,", c=",c,", h=",h,", w=",w)


# ## Load Label Map 
# For each detected object, the output from the model will include an integer to indicate which type (e.g. car, person, etc.) of trained object has been detected.  To translate the integer into a more readable text string, a label mapping file may be used.  The label mapping file is simply a text file of the format "n: string" (e.g. "7: car" for 7="car") that is loaded into a lookup table to be used later while labeling detected objects.
# 
# Here, if the *labels_path* variable has been set to point to a label mapping file, we open the file and load the labels into the variable *labels_map*.

# In[1]:


labels_map = None
# if labels points to a label mapping file, then load the file into labels_map
print(labels_path)
if os.path.isfile(labels_path):
    with open(labels_path, 'r') as f:
        labels_map = [x.strip() for x in f]
    print("Loaded label mapping file [",labels_path,"]")
else:
    print("No label mapping file has been loaded, only numbers will be used for detected object labels")


# ## Prepare Input Image
# Here we read and then prepare the input image by resizing and re-arranging its dimensions according to the model's input dimensions.  We define functions the *loadInputImage()* and *resizeInputImage()* for the operations so that we may reuse them again later in the tutorial.

# In[ ]:


# define function to load an input image
def loadInputImage(input_path):
    # globals to store input width and height
    global input_w, input_h
    
    # use OpenCV to load the input image
    cap = cv2.VideoCapture(input_path) 
    
    # store input width and height
    input_w = cap.get(3)
    input_h = cap.get(4)
    print("Loaded input image [",input_path,"], resolution=", input_w, "w x ",input_h,"h")
    return cap

# define function for resizing input image
def resizeInputImage(image):
    # resize image dimensions form image to model's input w x h
    in_frame = cv2.resize(image, (w, h))
    # Change data layout from HWC to CHW
    in_frame = in_frame.transpose((2, 0, 1))  
    # reshape to input dimensions
    in_frame = in_frame.reshape((n, c, h, w))
    return in_frame

# load the input image
cap = loadInputImage(input_path)
ret, image = cap.read()

# resize the input image
in_frame = resizeInputImage(image)
print("Resized input image from {} to {}".format(image.shape[:-1], (h, w)))

# display input image
print("Input image:")
plt.axis("off")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# ## Run Inference
# Now that we have the input image in the correct format for the model, we now run inference on the input image.

# In[ ]:


# save start time
inf_start = time.time()

# run inference
res = exec_net.infer(inputs={input_blob: in_frame})   

# calculate time from start until now
inf_time = time.time() - inf_start
print("Inference complete, run time: {:.3f} ms".format(inf_time * 1000))


# ## Process Results
# Now we parse the inference results and for each object detected draw boxes with text annotations on image.  We define the function *processResults()* so that we may use it again later in the tutorial to process results.
# 
# *res* is set to the output of the inference model which is an array of results, with one element for each detected object.  We loop through *res* setting *obj* to hold the results for each detected object which appear in *obj* as:
# - *obj[1]* = Class ID (type of object detected)
# - *obj[2]* = Probability of detected object
# - *obj[3]* = lower x coordinate of detected object 
# - *obj[4]* = lower y coordinate of detected object
# - *obj[5]* = upper x coordinate of detected object
# - *obj[6]* = upper y coordinate of detected object

# In[ ]:


# create function to process inference results
def processResults(result):
    # get output results
    res = result[output_blob]
    
    # loop through all possible results
    for obj in res[0][0]:
        # If probability is more than specified threshold, draw and label box 
        if obj[2] > prob_threshold:
            # get coordinates of box containing detected object
            xmin = int(obj[3] * input_w)
            ymin = int(obj[4] * input_h)
            xmax = int(obj[5] * input_w)
            ymax = int(obj[6] * input_h)
            
            # get type of object detected
            class_id = int(obj[1])
            
            # Draw box and label for detected object
            color = (min(class_id * 12.5, 255), 255, 255)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 4)
            det_label = labels_map[class_id] if labels_map else str(class_id)
            cv2.putText(image, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
                        cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

processResults(res)
print("Processed inference output results.")


# ## Display Results
# Now that the results from inference have been processed, we display the image to see what has been detected.  

# In[ ]:


# convert colors BGR -> RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# disable axis display, then display image
plt.axis("off")
plt.imshow(image)


# # Exercise #1: Run a Different Image
# Now that we have seen all the steps, let us run them again on a different image.  We also define *inferImage()* to combine the input processing, inference, and results processing so that we may use it again later in the tutorial.

# In[ ]:


# define function to prepare input, run inference, and process inference results
def inferImage(image):
    # prepare input
    in_frame = resizeInputImage(image)

    # run inference
    res = exec_net.infer(inputs={input_blob: in_frame})   

    # process inference results 
    processResults(res)

# set path to differnt input image
input_path="face.jpg"

# load input image
cap = loadInputImage(input_path)

# read image
ret, image = cap.read()

# display input image
plt.axis("off")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# infer image
inferImage(image)

# display image with inference results
# convert colors BGR -> RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# create new figure, disable axis display, then display image
plt.figure()
plt.axis("off")
plt.imshow(image)


# # Exercise #2: (Optional) Run Your Own Image
# Here you may run any image you would like by setting the *input_path* variable which may be set to a local file or URL.  A sample URL is provided as an example.

# In[ ]:


# input_path may be set to a local file or URL
input_path="https://github.com/chuanqi305/MobileNet-SSD/raw/master/images/004545.jpg"

# load input image
cap = loadInputImage(input_path)

# read image
ret, image = cap.read()

# display input image
plt.axis("off")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# infer image
inferImage(image)

# display image with inference results
# convert colors BGR -> RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# create figure, disable axis display, then display image
plt.figure()
plt.axis("off")
plt.imshow(image)


# # Exercise #3: Running Inference on Video
# We have seen how to run individual images, now how do we do video?  To run inference on video is much the same as for a single image except that a loop is necessary to process all the frames in the video.  In the code below, we use the same method of loading a video as we had for an image, but now include the while-loop to keep reading images until *cap.isOpened()* returns false or *cap.read()* sets *ret* to false:
# 
# while cap.isOpened():
#     # read video frame
#     ret, im = cap.read()
#    
#     # break if no more video frames
#     if not ret:
#         break  
#     ...

# In[ ]:


# close and then re-import matplotlib to be able to update output images for video
plt.close()
get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib import pyplot as plt
# disable axis display
plt.axis("off")

# input_path may be set to local file or URL 
input_path="/opt/intel/openvino/deployment_tools/inference_engine/samples/end2end_video_analytics/test_content/video/cars_768x768.h264"

print("Loading video [",input_path,"]")
cap = loadInputImage(input_path)

# track frame count and set maximum
frame_num = 0
max_frame_num = 60
skip_num_frames = 100
last_frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
if last_frame_num < 1: last_frame_num = "unknown"

while cap.isOpened():
    # read video frame
    ret, image = cap.read()
   
    # break if no more video frames
    if not ret:
        break  
    
    frame_num += 1

    # skip skip_num_frames of frames, then infer max_frame_num frames from there
    if frame_num > skip_num_frames: 
        # infer image
        inferImage(image)

        # display results
        # convert colors BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # show image then force re-draw to show new image
        plt.imshow(image)
        plt.gcf().canvas.draw()
    
    # display current frame number
    print("Frame #:", frame_num, "/", last_frame_num, end="\r")
    
    # limit number of frames, video can be slow and gets slower the more frames that are processed
    if frame_num >= (max_frame_num + skip_num_frames): 
        print("\nStopping at frame #", frame_num)
        break

print("\nDone.")


# # Exercise #4: (Optional) Run Your Own Video
# If you would like to see inference run on your own video, you may do so by first setting *input_path* to a local file or URL and then re-executing the cell above.  For example, you could use this video: https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4 by replacing the *input_path="..."* line above with the line:
# 
# input_path="https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4"
# 
# You can control which frame to start from by setting *skip_num_frames* which will skip that many frames.
# YOu can also control how many frames to show by setting *max_frame_num*.
# 
# **Note:** There are more videos available to choose from at: https://github.com/intel-iot-devkit/sample-videos/

# # Exit: Free Resources
# Now that we are done running the sample, we clean up by deleting objects before exiting.

# In[ ]:


del exec_net
del plugin
del net


# # End of Tutorial - Next Steps
# 
# ### [More Jupyter Notebook Tutorials](https://access.colfaxresearch.com/?p=experience)
# ### [Intel® Distribution of OpenVINO™ toolkit Main Page](https://software.intel.com/openvino-toolkit)

#  
