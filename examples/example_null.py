#Example of a function under test which returns an empty segmentation for all inputs
#
#Can be used to view true segmentations without any overlay of a calculated segmentation
#
#Function nullMesh can be used in mesh mode
#Function nullVoxel can be used in voxel mode

import numpy as np

#Example function for use in voxel mode
#Always returns a numpy array of zeros with shape matching the input array
#To run, use command "py testbench.py examples/example_null.py nullVoxel(filename) voxel -r"
def nullVoxel(filename):
    return np.zeros(voxel_array.shape)
    

#Example function for use in mesh mode
#Always returns an empty numpy array
#To run, use command "py testbench.py examples/example_null.py nullMesh(filename) mesh -r"
def nullMesh(filename):
    return np.array([])
