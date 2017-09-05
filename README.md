# Notebook problems
If there are problems like iopub limit exceeded, please set the limit manually using
`jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000`
In the Jupyter Notebook docs, they show that the default is:
`NotebookApp.iopub_data_rate_limit : Float
Default: 1000000

(bytes/sec) Maximum rate at which messages can be sent on iopub before they are limited.
`
So, just set something that is higher than that to allow for the iopub limit to be exceeded. 

# Installing MATLAB engine
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

# Using the MATLAB engine
Full on tower of babel problem. So, MATLAB and python use different default types for numbers. Python floats are converted to MATLAB doubles. So use floats when one expects MATLAB doubles and int where we want to convert it to int64. Thomas already has done the default ones in the code (which for some reason does not function on the mac for me) but one might have to look at the files to check the data type of the values that is taken by the program.

# Using MATLAB
You will need the following package(s):
- `psychtoolbox-3`

You can download and install that by following the instructions here : http://psychtoolbox.org/
However, this does not automatically add the psychtoolbox to your MATLAB search path (which can be checked by logging onto MATLAB and executing the path function). This can be overcome by the following :
- open the Psychtoolbox workspace of MATLAB with sudo (admin rights) (usually `sudo ptb3-matlab` works) which allows one to change the pathdef.m file that contains all the information about the paths that MATLAB is suppposed to look at for while searchig for functions. Use savepath after checking that the psychtoolbox3 functions exist in `path`. This should do it. 
- If it does not work, one may use the GUI. Go to matlab, outside the ptb workspace (i.e. do not use `ptb3-matlab`, but rather just `matlab`). Run `pathtool` and add the requisite folder which has the psychtools, along with all the requisite subfolders.

#Using psychopy
An easy way to generate stimuli would be to use the psychopy toolbox but that is available only for python2.7. Note that the grating and the stimulus represented would require both observer translation and rotation for it to work in all the cases for which the path integration would be required. 

---------------------------------------------------------------------------------------------------------------------------------------------------

# Notes for me : 

It makes sense to have a more systematic search of the hemisphere rather than having a random search as it is shown that at certain regions of the hemisphere, the values of the residual functions are seemingly concentrated on a small part of the hemisphere, so we do not want to lose it as such.

There is something wrong with the calculatio of the translation direction, perhaps we can have a look at it in terms of the actual mathematical solving of the problem. Update : I found what was wrong, yay! I was using the wrong values of the sample velocity and the sample position.

Okay, now we shall use the values of the optic flow generated artificially by me in order to test the robustness of the algorithm under question. It would also be nice if we could generate the optic flow values back from the values of the colour coded UCL images that we saw. From that, with the parallelized code, it should be enough for us to work on for the next few days.

So, there is some kind of stochasticity during the calculation of the subspace algorithm. Perhaps, we can look at some of the other ways to look at it? The 2012 paper needs to be looked at in a better way apparently. 

Also, to be looked into - the use of either python pool architecture or into some form of Numba or PyCUDA that can be useful. 

To split an array into many subarrays, one can use the following nice little piece of code, maybe there are other ways to do it though :
`for x_split in np.array_split(v_x,10,axis=1):
	for y_split in np.array_split(x_split,10,axis=2):`

Better is to think about and use the indices of the matrix rather than the values directly. Also, on using the parallelization, it helps to use not only for different areas but also for different time steps. 	

Pooling helps to get the speed increased, but not by a significant amount. (This is seen even later in the use of the code for the different patches to be used for the estimation of the value of the translation direction)

Notes for me : Please do check what is the kind of implementation that is being done by the linalg ortho code in the numpy package for the implementation seems to give different answers if what I have read from the paper is correct. 

The subset method works well and reasonably fast for the most part, even when there is noise, and extracts the direction of translation quite well, maybe we can have a look at other models to validate that the optic flow leading to the path integration indeed is useful. The review by Raudies and Neumann et al show that this is probably one of the best ways to estimate the ego motion from the optic flow values and although the neural substrate is not well defined. The parallel and fast calculation of it, from the Markus Lappe paper might be useful to implement.

## Using either of the two methods for calculation

The calculation of SVD decomposition and QR decomposition represent two parallel ways of finding out the direction of translation from the velocity matrices. The calculation of linalg.orth requires the svd decomposition anyways so using the calculation of the orthonormal basis for the nullspace (which again would require the use of either the QR decomposition or the SVD decomposition anyway). So I would suggest using the QR decomposition only once during the entire calculation as that might make the situation faster and more time efficient as well as easy to implement.  