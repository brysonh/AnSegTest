# Installation:
The test script requires a valid installation of Python version 3.9 or greater. The test script also requires the following additional modules: pyglet version>=1.5.26 and <2.0.0, numpy, pydicom, scipy, and networkx. No additional installation is required. The requirements.txt file can be used with pip to install these modules.

# Usage:
The program can be run using the command `py testbench.py [PATH] [PROTOTYPE] [MODE] -i [INCLUDE] -e [EXCLUDE] -o [OUTPUT] -r`

**PATH**, **PROTOTYPE**, and **MODE** are required positional arguments. All other arguments are optional.

- **PATH:** file location which contains the function to be tested. Can be a relative or absolute path. Must include the name of the file.   
  *Example: for file "test.py" located in directory "temp" within the test script's directory, PATH would be "temp/test.py"*

- **PROTOTYPE:** name and arguments of the test function. Arguments should be contained within parentheses and separated py commas.  
  *Example: for function "testFn" with arguments "filename", "point", and "vertices", PROTOTYPE would be "testFn(filename,point,vertices)"*
  #### Available arguments:
  ##### Mesh mode:
  The following arguments can be used when MODE is set to "mesh": 
  - **filename** *(string)*: contains the absolute path and filename of the .obj file which stores the current test data.  
  - **point** *(1D numpy array of 3 floats)*: contains the x, y, and z coordinates of a vertex which is part of the aneurysm.  
  - **vertices** *(Nx3 numpy array of floats)*: contains the vertices which make up the mesh.  
  - **faces** *(Mx3 numpy array of ints)*: indexed vertex list which contains faces which make up the mesh. Each int is an index in "vertices" which identifies a vertex which forms a corner of the given face. The first dimension with length M contains each face, and the second dimension with length 3 contains the indices of each vertex of a given face.  
  
  ##### Voxel mode:
  The following arguments can be used when MODE is set to "voxel": 
  - **filename** *(string)*: contains the absolute path and filename of the .obj file which stores the current test data.  
  - **point** *(1D numpy array of 3 ints)*: contains the x, y, and z indices of a voxel which is part of the aneurysm.  
  - **voxel_array** *(3D numpy array of ints)*: voxel array where "0" represents a voxel which is not part of the image and "1" represents a voxel which is part of the image.

- **MODE:** data format to be used in the test. Supported options are "mesh" and "voxel".  
  *Example: if the function under test expects a voxel array as input, MODE would be "voxel"*

- **-i/--include [INCLUDE]:** files which should be used as input data. By default, all files located in the test data directories (by default "mesh" and "voxel" in the test script's directory) which are of the correct file type will be used. Specifying specific input files will ignore all files other than the files listed in INCLUDE. Filenames should be separated by spaces.  
  *Example: to use only AN1, AN2, and AN3 as input, specify "-i AN1 AN2 AN3"*

- **-e/--exclude [EXCLUDE]:** files which should not be used as input data. All files in the test data directory other than files listed in EXCLUDE will be used for testing. Cannot be used together with -i. If -i and -e are both specified, -e is ignored and only files listed in INCLUDE will be used. Filenames should be separated by spaces.  
  *Example: to use all files except AN1, AN2, and AN3, specify "-e AN1 AN2 AN3"*

- **-o/--output [OUTPUT]:** file location in which to store the generated output file. Defaults to "out_[YEAR]-[MONTH]-[DAY]--[HOUR]-[MINUTE]-[SECOND].csv" in the "output" folder in the test script's directory. The output will be a CSV file specifying the similarity results for each test file run.  
  *Example: to store the output in a file named "output.csv" in the "temp" directory in the test script's directory, specify "-o temp/output.csv"*

- **-r/--render:** specifies to render the results. If not specified, rendering is not performed. If specified, a window will open after all files are calculated with a GUI to view the rendered results.   
  *Example: if rendering is desired, specify "-r"*

# Return values: 
In mesh mode, the test script expects the function under test to return a Yx3 numpy array of floats containing the x, y, and z coordinates of vertices which are part of the segmented aneurysm. Y is a positive integer less than the total number of verticies in the mesh. Each row of the returned array should correspond to a vertex in the input mesh.

In voxel mode, the test script expects the function under test to return a 3D numpy array of ints with the same dimensions as the input voxel array. The returned voxel array should be a mask where elements with value "1" indicate voxels which are part of the segmented aneurysm, and elements with value "0" indicate voxels which are not part of the segmented aneurysm. Only voxels on the surface of the aneurysm (ie, voxels with at least one exposed face) should be included in the result.