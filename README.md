#Notebook problems
If there are problems like iopub limit exceeded, please set the limit manually using
`jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000`
In the Jupyter Notebook docs, they show that the default is:
`NotebookApp.iopub_data_rate_limit : Float
Default: 1000000

(bytes/sec) Maximum rate at which messages can be sent on iopub before they are limited.
`
So, just set something that is higher than that to allow for the iopub limit to be exceeded. 
#Installing MATLAB engine
This is a pain in the ass. The following might help for some. Did not help for me, except of course the last one, which is the official one. Do remember, if using conda and if sudo (root) does not use conda, and hence has a different version of python installed, use the proper path while setting up the manager for the jupyter notebook : 
- `conda install gcc`
- `libglut.so.3` problem solved: with `sudo apt-get install freeglut3`
- helpful pages: [Matlab engine for python](https://fr.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) as well as the installation of [matlab engine in nondefault locations](https://fr.mathworks.com/help/matlab/matlab_external/install-matlab-engine-api-for-python-in-nondefault-locations.html)

The jupyter notebook might show errors that look like the following:
`ImportError: /home/YOURUSERNAME/anaconda3/lib/python3.5/sitepackages/zmq/backend/cython/../../../../.././libstdc++.so.6: version 'CXXABI_1.3.8' not found (required by /usr/local/MATLAB/R2016b/extern/engines/python/dist/matlab/engine/glnxa64/../../../../../../../bin/glnxa64/libmx.so)`
So, one might search for the correct version of the CXXABI by
`scanelf -l -s CXXABI_1.3.8 | grep CXXABI_1.3.8`
which returned, say
`ET_DYN CXXABI_1.3.8 /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.23`

Append the following to the .bashrc file to change the LD_LIBRARY_PATH to make sure the correct path is searched. The following is an example. 
`export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.23:$LD_LIBRARY_PATH"`

#Using the MATLAB engine
Full on tower of babel problem. So, MATLAB and python use different default types for numbers. Python floats are converted to MATLAB doubles. So use floats when one expects MATLAB doubles and int where we want to convert it to int64. Thomas already has done the default ones in the code (which for some reason does not function on the mac for me) but one might have to look at the files to check the data type of the values that is taken by the program.

#Using MATLAB
You will need the following package(s):
- `psychtoolbox-3`

You can download and install that by following the instructions here : http://psychtoolbox.org/
However, this does not automatically add the psychtoolbox to your MATLAB search path (which can be checked by logging onto MATLAB and executing the path function). This can be overcome by the following :
- open the Psychtoolbox workspace of MATLAB with sudo (admin rights) (usually `sudo ptb3-matlab` works) which allows one to change the pathdef.m file that contains all the information about the paths that MATLAB is suppposed to look at for while searchig for functions. Use savepath after checking that the psychtoolbox3 functions exist in `path`. This should do it. 
- If it does not work, one may use the GUI. Go to matlab, outside the ptb workspace (i.e. do not use `ptb3-matlab`, but rather just `matlab`). Run `pathtool` and add the requisite folder which has the psychtools, along with all the requisite subfolders.

#Using psychopy
An easy way to generate stimuli would be to use the psychopy toolbox but that is available only for python2.7. Note that the grating and the stimulus represented would require both observer translation and rotation for it to work in all the cases for which the path integration would be required. 

---------------------------------------------------------------------------------------------------------------------------------------------------

Notes for me : 

It makes sense to have a more systematic search of the hemisphere rather than having a random search as it is shown that at certain regions of the hemisphere, the values of the residual functions are seemingly concentrated on a small part of the hemisphere, so we do not want to lose it as such.

There is something wrong with the calculatio of the translation direction, perhaps we can have a look at it in terms of the actual mathematical solving of the problem. Update : I found what was wrong, yay! I was using the wrong values of the sample velocity and the sample position