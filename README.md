
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


## Beginning of writing for the project report 

This project deals with the estimation of path integration from soleley the optic flow, and how the capabilities of path integration changes with aging in the network. 

We define optic flow in our case with respect to the biological eye. Hence, we look at the change of structured light that is produced in the retina, due to a relative motion between the eyeball and the scene. Note that the relative motion of the eyeball and the scene may be caused by one or a combination of the following exhaustive scenarios : (1) The eyeball is fixed in the head and the head is fixed with respect to the body and there is movement of the centre of mass of the body in space. This is referred to as translation in this article. Disambiguating the direction of translation from just the optic flow to calculate the path taken by the observer is a major focus of this project. (2) The head is fixed with respect to the body (i.e. there are no neck movements). Anything in the visual field that moves with the head remians stable, (does not generate optic flow field vectors). This is the head centered frame of reference that is being spoken about in the following paragraph. (3) The eye-ball is rotating. Visual information leaving the retina is organized into a two-dimensional map or a retinotopic frame of reference. Everything that is associated with this frame of reference moves along with eye motion. 

The importance of optic flow in practical matters have been well studied in experiments and in theoretical models. Optic flow is continuously processed in our visual system and can help in solving various tasks. Such tasks may involve the estimation of self-motion which is the most important component of finding the path integration for a human observer, the segmentation of the scene into IMOs and rigid objects. (Note that the framework that we have used for this project assumes the scene to be completely rigid at all times and does not permit the existence of independently moving objects). Optic flow also allows for the estimation of relative depths of all visible and rigid objects. This can be done due to the phenomenon in humans that is known as the motion parallax, which will be (will it?) described in a later part of the article. 

These three proposed forms of optic flow generation might be related to the three frames of reference that the brain is purported to construct (Kandel Pg 498, Box 25-1) for visual perception and the control of movement : a retinotopic frame of reference, a head-centered frame of reference and a body-centred frame of reference. Each level of frame of reference is constructed on top of the other using some of the information available at each of the levels as compared to the others. For example, using the added infromation about the eye position, the brain constructs the head-centred frame of reference from the retinotopic frame of reference. Similarly, a body-centric frame of reference can be calculated from the head-centric frame of reference using the knowledge about the eye movement as well as the body movement. 

An analytical model of optic flow, majorly defined as retinal optic flow, has been studied since more than half a century ago, and it has been called by Horn (1986) as the motion field, which is a geometrical concept. The first part of this project focuses on the geometrical concept with slight regard towards the biological implementation of the problem. This motion field is determined by the `projection function' of the scene onto the retinal plane (the simplest model used here uses perspective projection), the movement of the observer and the geometry of the scene. In the literature, the motion field and the optic flow are often confounded for their similarity. It is to be noted that the optic flow is defined by the temporal change of the registered patterns of light. The motion field was, as is seen by the paper by Horn, the motion field was formulated for the case of a pinhole camera in locomoion, where locomotion is approximated by three translational and three rotational parameters. There are a diversity of techniques used to estimate the optic flow, both from the point of view of computational vision and from the point of view of neuroscience and psychophysics. 

The optic flow provides limited information about the ego-motion and the path taken by the observer because of the 3D to 2D projection (perspective projection) on the retinal frame. However, the movement of the observer leads to some more information about the 3D scene which is beyond the data that is accessible from a single 2D image because of which David Marr called it 2.5D in his book Vision. Using the motion field notation for the optic flow, one can hence estimate, using various algorithms, the self-motion of the observer, in terms of self-motion of the pinhole camera which is at the assumed centre of estimating the motion of the observer. The review article by Raudies and Neumann provides a good overall view of the proposed algorithms which are in use in the current time to estimate the veloctiy parameters form optic flow. Following the review, one of the methods was selected for its simplicity and robustness to noise. There have been direct methods also for the estimation of velocity parameters directly from the video without estimating the optic flow, but those are not discussed here (## POINT FOR ME : PERHAPS THE BRAIN DOES SOMETHING LIKE THIS and hence we might have something similar to this - Horn 1988 might come in use for this) Also interesting is the view that observers might use several different kinds of strategies to either generate the optic flow or to process it robustly. Plus, the view that too good and accurate optic flow is perhaps not needed and what the observer needs is a fast and crude approach, 

## Optic flow in neuroscience

Motion selective cells required for the calculation of optic flow are integrated within the brain's middle temporal are