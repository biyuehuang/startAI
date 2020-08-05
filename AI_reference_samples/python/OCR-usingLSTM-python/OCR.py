#!/usr/bin/env python
# coding: utf-8

# # OCR (Optical Character Recognition)
# 
# This is an example for OCR (Optical Character Recognition) using the Intel® Distribution of OpenVINO™ toolkit.  
# We will use the Convolutional Recurrent Neural Networks (CRNN) for Scene Text Recognition from the following github page : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# 
# To obtain the frozen model necessary to start with the Intel® Distribution of OpenVINO™ toolkit from the github repository, please look at our [documentation](https://docs.openvinotoolkit.org/R5/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_CRNN_From_Tensorflow.html) 
# 
# In this tutorial, we will show you first how to convert the TF (TensorFlow) frozen model through the Model Optimizer, then we will perform inference on the CPU (first Intel® Xeon® CPU and then Intel® Core™ CPUs).
# As the CRNN includes a LSTM (Long Short-term Memory) cell, the inference can only be performed on CPU (the only hardware plugin to support this layer yet)

# ## 0. Setup the Python environement 
# 

# In[1]:


#from IPython.display import HTML
#import matplotlib.pyplot as plt
import os
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent.parent))
#from demoTools.demoutils import *


# ## 1. Model Optimizer
# 
# 
# Model Optimizer creates Intermediate Representation (IR) models that are optimized for inference. 
# 

# In[2]:


#get_ipython().system('/opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model model/crnn.pb --data_type FP32 -o model/FP32')


# **Note** the above line is a single command line input, which spans 4 lines thanks to the backslash '\\', which is a line continuation character in Bash.
# 
# Here, the arguments are:
# * --input-model : the original model
# * --data_type : Data type to use. One of {FP32, FP16, half, float}
# * -o : output directory
# 
# This script also supports `-h` that will you can get the full list of arguments.
# 
# With the `-o` option set as above, this command will write the output to the directory `model/FP32`
# 
# There are two files produced:
# ```
# models/FP32/crnn.xml
# models/FP32/crnn.bin
# ```
# These will be used later in the exercise.

# ## 2. Inference Engine
# 
# Now, we will run the inference on this model by building progressively the Python sample required to perform inference. 
# This part of exercise feaures our Python API, similar functionalities can be found in our C++ API too.

# We will do OCR on the following input image, which obviously reads as **Industries**.
# ![Image](board4.jpg)

# #### Call Python Packages

# In[3]:


#get_ipython().system('pip3 install --user easydict')


# In[4]:


#from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork, IEPlugin
from local_utils import log_utils, data_utils
from local_utils.config_utils import load_config
import os.path as ops
from easydict import EasyDict


# #### Define variables  like model path, target device, the codec for letter conversion

# In[5]:


model_xml='model/FP32/crnn.xml'
device_arg='CPU'
input_arg=['board4.jpg']
iterations=1
perf_counts=False

codec = data_utils.TextFeatureIO(char_dict_path='Config/char_dict.json',ord_map_dict_path=r'Config/ord_map.json')
log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)


# #### Plugin initialization for specified device and load extensions library if specified
# Now we must select the device used for inferencing. This is done by loading the appropriate plugin to initialize the specified device and load the extensions library (if specified) provided in the extension/ folder for the device.
# 
# The following cell constructs **`IEPlugin`**:

# In[6]:


plugin = IEPlugin(device=device_arg, plugin_dirs='')


# #### Read IR
# We can import optimized models (weights) from step 1 into our neural network using **`IENetwork`**. 

# In[7]:


model_bin = os.path.splitext(model_xml)[0] + ".bin"
net = IENetwork(model=model_xml, weights=model_bin)


# #### Preparing input blobs

# In[8]:


input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
net.batch_size = len(input_arg)


# #### Read and pre-process input images
# First let's load the image using OpenCV.
# We will also have to do some shape manipulation to convert the image to a format that is compatible with our network

# In[9]:


n, c, h, w = net.inputs[input_blob].shape
images = np.ndarray(shape=(n, c, h, w))
for i in range(n):
    image = cv2.imread(input_arg[i])
    if image.shape[:-1] != (h, w):
        log.warning("Image {} is resized from {} to {}".format(input_arg[i], image.shape[:-1], (h, w)))
        image = cv2.resize(image, (w, h))
    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    images[i] = image
log.info("Batch size is {}".format(n))


# #### Loading model to the plugin
# Once we have the plugin and the network, we can load the network into the plugin using **`plugin.load`**.

# In[10]:


exec_net = plugin.load(network=net)


# #### Start Inference
# We can now run the inference on the object  **`exec_net`** using the function infer.

# In[ ]:


infer_time = []
for i in range(iterations):
    t0 = time()
    res = exec_net.infer(inputs={input_blob: images})
    infer_time.append((time()-t0)*1000)

res = res[out_blob]
    
log.info("Average running time of one iteration: {} ms".format(np.average(np.asarray(infer_time))))


# #### Processing output blob
# The network outputs a tensor of dimension 25 (string length) * 37 (dimension of character space).
# First, we will go through the 25 characters and extracts the highest probability in the character space and its index in this space. 
# We use the encoding files from the Github page to recover the mapping from index to character. (0&rarr;"a",36&rarr;" ")
# In the github page, they also remove the consecutive duplicates and the space char, therefore we also perform this postprocessing. 
# 

# In[ ]:


preds = res.argmax(2) ## extract highest probability in the second dimension
preds = preds.transpose(1, 0)
preds = np.ascontiguousarray(preds, dtype=np.int8).view(dtype=np.int8) # reformat to an array 
values=codec.writer.ordtochar( preds[0].tolist()) # map from index to character
values=[v for i, v in enumerate(values) if i == 0 or v != values[i-1]] # remove duplicates
values = [x for x in values if x != ' '] # remove space char (was character from index 36)
res=''.join(values)
print("The result is : " + res)


# # 3. Job submission
# 
# All the code up to this point has been run within the Jupyter Notebook instance running on a development node based on an Intel Xeon Scalable processor, where the Notebook is allocated a single core. 
# We will run the workload on other edge compute nodes represented in the IoT DevCloud. We will send work to the edge compute nodes by submitting the corresponding non-interactive jobs into a queue. For each job, we will specify the type of the edge compute server that must be allocated for the job.
# 
# The job file is written in Bash, and will be executed directly on the edge compute node.
# For this example, we have written the job file for you in the notebook.
# Run the following cell to write this in to the file "ocr_job.sh"

# In[ ]:


get_ipython().run_cell_magic('writefile', 'ocr_job.sh', '\n# The default path for the job is your home directory, so we change directory to where the files are.\ncd $PBS_O_WORKDIR\nmkdir -p $1\n# Running the object detection code\n# -l /opt/intel/openvino/deployment_tools/inference_engine/samples/build/intel64/Release/lib/libcpu_extension.so \\\nSAMPLEPATH=$PBS_O_WORKDIR\npython3 classification_sample.py  -m model/$3/crnn.xml  \\\n                                           -i board4.jpg \\\n                                           -o $1 \\\n                                           -d $2\n                                           ')


# ### 3.1 Understand how jobs are submitted into the queue
# 
# Now that we have the job script, we can submit the jobs to edge compute nodes. In the IoT DevCloud, you can do this using the `qsub` command.
# We can submit ocr_job to 5 different types of edge compute nodes simultaneously or just one node at at time.
# 
# There are three options of `qsub` command that we use for this:
# - `-l` : this option lets us select the number and the type of nodes using `nodes={node_count}:{property}`. 
# - `-F` : this option lets us send arguments to the bash script. 
# - `-N` : this option lets use name the job so that it is easier to distinguish between them.
# 
# The `-F` flag is used to pass in arguments to the job script.
# The [ocr.sh](ocr_job.sh) takes in 4 arguments:
# 1. the path to the directory for the output video and performance stats
# 2. targeted device (e.g. CPU,GPU,MYRIAD)
# 3. the floating precision to use for inference
# 4. the path to the input video
# 
# The job scheduler will use the contents of `-F` flag as the argument to the job script.
# 
# If you are curious to see the available types of nodes on the IoT DevCloud, run the following optional cell.

# In[ ]:


get_ipython().system('pbsnodes | grep compnode | sort | uniq -c')


# Here, the properties describe the node, and number on the left is the number of available nodes of that architecture.
# 
# ### 3.2 Job queue submission
# 
# The output of the cell is the `JobID` of your job, which you can use to track progress of a job.
# 
# **Note** You can submit all 5 jobs at once or follow one at a time. 
# 
# After submission, they will go into a queue and run as soon as the requested compute resources become available. 
# (tip: **shift+enter** will run the cell and automatically move you to the next cell. So you can hit **shift+enter** multiple times to quickly run multiple cells).
# 

# #### Run on Intel® Xeon® E3 CPU

# In[ ]:


print("Submitting a job to an edge compute node with an Intel Xeon CPU...")
#Submit job to the queue
job_id_xeon = get_ipython().getoutput('qsub ocr_job.sh -l nodes=1:tank-870:e3-1268l-v5 -F "results/xeon CPU FP32" $VIDEO -N obj_det_xeon')
print(job_id_xeon[0]) 
#Progress indicators
if not job_id_xeon:
    print("Error in job submission.")


# #### Run on Kabylake Intel® Core™ CPU

# In[ ]:


print("Submitting a job to an edge compute node with an Intel Core CPU...")
#Submit job to the queue
job_id_kabylake_core = get_ipython().getoutput('qsub ocr_job.sh -l nodes=1:tank-870:i5-7500t -F "results/kabylake_core CPU FP32" $VIDEO -N obj_det_core')
print(job_id_kabylake_core[0]) 
#Progress indicators
if not job_id_kabylake_core:
    print("Error in job submission.")


# #### Run on (Skylake) Intel® Core™ CPU

# In[ ]:


print("Submitting a job to an edge compute node with an Intel Core CPU...")
#Submit job to the queue
job_id_skylake_core = get_ipython().getoutput('qsub ocr_job.sh -l nodes=1:tank-870:i5-6500te -F "results/skylake_core CPU FP32" $VIDEO -N obj_det_core')
print(job_id_skylake_core[0]) 
#Progress indicators
if not job_id_skylake_core:
    print("Error in job submission.")


# ### 3.3 Check if the jobs are done
# 
# To check on the jobs that were submitted, use the `qstat` command.
# 
# We have created a custom Jupyter widget  to get live qstat update.
# Run the following cell to bring it up. 

# In[ ]:


liveQstat()


# You should see the jobs you have submitted (referenced by `Job ID` that gets displayed right after you submit the job in step 2.3).
# There should also be an extra job in the queue "jupyterhub": this job runs your current Jupyter Notebook session.
# 
# The 'S' column shows the current status. 
# - If it is in Q state, it is in the queue waiting for available resources. 
# - If it is in R state, it is running. 
# - If the job is no longer listed, it means it is completed.
# 
# **Note**: Time spent in the queue depends on the number of users accessing the edge nodes. Once these jobs begin to run, they should take from 1 to 5 minutes to complete. 

# ### 3.4 View Results
# 
# Once the jobs are completed, the queue system outputs the stdout and stderr streams of each job into files with names of the form
# 
# `obj_det_{type}.o{JobID}`
# 
# `obj_det_{type}.e{JobID}`
# 
# (here, obj_det_{type} corresponds to the `-N` option of qsub).
# 
# However, for this case, we may be more interested in the output result, which can be found inside the results/core/result.txt file.
# Run the cells below to display them.

# In[ ]:


with open("results/xeon/result.txt") as f: # The with keyword automatically closes the file when you are done
    print(f.read())


# In[ ]:


with open("results/kabylake_core/result.txt") as f: # The with keyword automatically closes the file when you are done
    print(f.read())


# In[ ]:


with open("results/skylake_core/result.txt") as f: # The with keyword automatically closes the file when you are done
    print(f.read())


# At this stage, the output result you should have got is: **industries**, matching the input image.

# In[ ]:


arch_list = [('kabylake_core', 'Intel Core\ni5-6500TE\nCPU (Kabylake)'),
             ('skylake_core', 'Intel Core\ni5-6500TE\nCPU (Skylake)'),
             ('xeon', 'Intel Xeon\nE3-1268L v5\nCPU'),
             ]

stats_list = []
for arch, a_name in arch_list:
    if 'job_id_'+arch in vars():
        stats_list.append(('results/{dir}/stats.txt'.format(dir=arch), a_name))
    else:
        stats_list.append(('placeholder'+arch, a_name))

summaryPlot(stats_list, 'Architecture', 'Time, mseconds', 'Inference Engine Processing Time', 'time' )

