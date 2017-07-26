import shutil
from gabor_kernel import gabor_3d_px
import numpy as np
import os
from multiprocessing import Pool

CPU_NUMBER = os.cpu_count()

#######################################################################################
### SimpleCell Class: Computes the responses of SIMPLE cells based on the sum of simple
###                    cell responses which are computed and summed inside the class
###                    much of the code here is a duplicate from the one of the ComplexCell class.
#######################################################################################

class SimpleCell(object):
    """
        Computes and stores the simple cell responses as well as the different filters needed to
        create those responses. The main method is `compute_responses`, which
        takes the image, convolves it with all the filters, and compute the normalisation required
        by the model of the Simple Cell. Then, the method `get_response` returns the response from
        a particular filter. 
        
        The parameters taken as input are as follows:
            - im_shape: shape of the videos that we'll treat (n_frames x x_size x y_size)
            - angles: vector of angles sensitivities
            - velocities: vector of velocities sensitivities
            - wavelength: vector of wavelength sensitivities
            - phases: vector of phases sensitivities
            - c: ratio between the time step and space step.
            - alpha: width of the filter in time
            - a_1, s_1, K: parameters proper to the simple cell model (cf. Simoncelli and Heeger's paper for more details)
            - filt_width: width of the filter (ratio multiplied with the wavelength to define the spatial extent of the filter)
            - deactivation_rate: rate of deactivation of the cells to simulate ageing.
            - directory_cell: where the responses are stored 
            - file_cell: name of the file
            - filters_theta_v: if it's ture, it means that angles and velocities have the same number of entries, and that
                               one angle will be associated with its velocity counterpart 
                               (not all combinations will be computed)
            - causal_time: whether the filter should be causal (set all entries with t > 0 to 0)
            - verbose: whether we want to get more information about the execution.
    """
    
    def __init__(self, im_shape, angles, velocities, wavelengths, phases, c, alpha, a_1, s_1, K, 
                 filt_width = 0.56, deactivation_rate = 0.0, directory_cell = 'simple_cell', file_cell = 'response' , 
                 filters_theta_v = False, causal_time = False, verbose = False):
        self.im_shape = np.array(im_shape)
        
        #directory_filt = 'filters',file_filt = 'gabor',
        
        if (self.im_shape%2 == 0).any():
            raise ValueError('This class can only take odd shapes as input.')
        
        self.angles =  np.array(angles) # array of preferred spatial response
        self.velocities = np.array(velocities) # array of preferred velocity response (- leftward, + rightward motion)
        self.wavelengths = np.array(wavelengths) # array of preferred wavelengths
        self.phases = np.array(phases) # array of preferred phases
        self.c = c # constant linking the spatial and time ratios
        self.alpha = alpha # constant fixing the width of the filter in time

        self.causal_time = causal_time

        
        self.a_1 = a_1
        self.s_1 = s_1
        self.K = K 
        self.filt_width = filt_width
        self.deactivation_rate = deactivation_rate        
        self.directory_cell = directory_cell
        self.file_cell = file_cell

        self.verbose = verbose
        
        # Transform the angles and velocities stimuli so that they are in the theta v form, i.e. couples (theta_i, v_i)
        if not filters_theta_v:
            len_ang = len(self.angles)
            self.angles = np.repeat(self.angles, len(self.velocities))
            self.velocities = np.tile(self.velocities, len_ang)
        
    def clean_directory(self):
        """
            Deletes the directory in which the simple cell responses to an image are stored upon
            deletion of the class instance (as the responses are specific to an image, it is not
            useful to keep those later on.
        """

        if os.path.isdir(self.directory_cell):
            shutil.rmtree(self.directory)

###################################################################################################
### 2 methods for computing the raw responses of simple cell to a stimulus
###     1. response_parallel: method which computes the responses over a certain list of
###                           of filters, can be called by several processes at the same time.
###                           The method creates and saves the corresponding filter as well.
###     2. compute_responses: splits the computation between different processes and calls 
###                           response_parallel from each process
###################################################################################################                                   
                                
    def compute_responses(self, stimulus):
        """
           Computes the responses of SimpleCells to a stimulus given as input, which has to be of the
           same shape of the im_shape attribute of the class. It creates a pool of workers, splits the
           vector of angles and velocities and computes the responses on different processes. Then, it sums
           the normalisation from all the processes and saves in order for it to be reused.
        """
            
        assert(len(self.angles) == len(self.velocities))
        
        if (stimulus.shape[0] != self.im_shape[0] or stimulus.shape[1] != self.im_shape[1] 
            or stimulus.shape[2] != self.im_shape[2]):
            raise ValueError('Incorrect shape of image for the given instance of SimpleCell')
            
        self.stimulus = stimulus
        
        O = len(self.wavelengths); P = len(self.phases)
        
        self.N_filters = len(self.angles)*O*P # number of filters
        
        # map linking a number between 0 and N_filters to its parameters
        self.filter_map_rad = np.zeros((self.N_filters, 4))
        
        if not os.path.isdir(self.directory_cell):
            os.mkdir(self.directory_cell)
            
        if self.verbose:
            print('CREATING A BANK OF',self.N_filters,'FILTERS OF SHAPE', self.im_shape[0],'x', 
                  self.im_shape[1],'x', self.im_shape[2])
            print('Process'.ljust(8),'Filter'.ljust(8),'Theta'.ljust(5),'v'.ljust(7), 'Lambda'.ljust(6), 
                  'Phi'.ljust(4), '|', 'ker > 0'.ljust(10), 'ker < 0'.ljust(10), 
                  'sum'.ljust(6),'max'.ljust(6),'min'.ljust(6),'abs'.ljust(6))
        
        angles_split = np.array_split(self.angles, CPU_NUMBER)
        velocities_split = np.array_split(self.velocities, CPU_NUMBER)
        
        for i, [theta, v] in enumerate(zip(self.angles, self.velocities)):
            for j, lambd in enumerate(self.wavelengths):
                for k, phi in enumerate(self.phases):
                    pos = O*P*i + P*j + k
                    self.filter_map_rad[pos, :] = np.array([theta, v, lambd, phi]) 
                            
        self.deactivation_map = np.random.choice([0, 1], size=(self.N_filters,self.im_shape[1], self.im_shape[2]), 
                                                       p=[ self.deactivation_rate, 1-self.deactivation_rate])
        pool = Pool(CPU_NUMBER)        
        # adding the semi-saturation constant
        
        normalisation = np.sum(pool.starmap(self.response_parallel, 
                                            zip(angles_split, velocities_split)), axis = 0) +  self.s_1

        pool.close()
        pool.join()
                    
        norm_path = self.directory_cell + '/normalisation.npy' 
        np.save(norm_path, normalisation)
                    
    def response_parallel(self, angles_split, velocities_split):
        """
            Creates a bank of 3D spatiotemporal Gabor Filters from the parameters given as input
            as well as from the class. Then, computes the raw responses to the stimulus given in 
            `compute_responses` and stores those.
            
            This method takes as input a split of the angles and velocities, and can be ran by several
            independent processes.

        """
        O = len(self.wavelengths); P = len(self.phases)
        
        fft_orig = np.fft.fftshift(np.fft.fftn(self.stimulus))
        normalisation = np.zeros(self.stimulus.shape)
        
        for i, [theta, v] in enumerate(zip(angles_split, velocities_split)):
            for j, lambd in enumerate(self.wavelengths):
                for k, phi in enumerate(self.phases):
                    pos = O*P*i + P*j + k
                    ## 1- CREATING THE KERNEL AND COMPUTING ITS FFT
                    sigma = self.filt_width*lambd 
                            
                    # Compute the psi angle from the required velocity and c constant : v = -c*tan(psi)
                    psi = np.arctan(v/self.c) 
                            
                    # k*cos(psi) = 2*pi/lambda (k*cos(psi) due to time shift)
                    frequency = 2*np.pi/lambd/np.cos(psi) 
                    _, _, _, ker = gabor_3d_px(size = self.im_shape, x_sig = sigma, 
                                                   y_sig = sigma, alpha = self.alpha, k = frequency,
                                                   c = self.c, theta = theta, psi = psi, phi = phi, 
                                                   causal_time = self.causal_time)
            
                    filt =  np.fft.fftshift(np.fft.fftn(np.fft.fftshift(ker)))
                        
                        ## 2- CONVOLVING THE STIMULUS WITH THE FILTER AND COMPUTING THE RAW MODEL RESPONSE
                    fft_new = fft_orig * filt   
                    response = np.real(np.fft.ifftn(np.fft.ifftshift(fft_new))) 
                    
                    if self.verbose:
                        print(str(os.getpid()).ljust(8), (str(pos+1)+'/'+str(len(angles_split)*O*P)).ljust(8), 
                              str(int(theta*180/np.pi)).ljust(5), str(round(v,3)).ljust(7), 
                              str(lambd).ljust(6), str(int(phi*180/np.pi)).ljust(4), '|', 
                              str(round(np.sum(response[response > 0]),3)).ljust(10),
                              str(round(np.sum(response[response < 0]), 3)).ljust(10), 
                              str(round(np.sum(response), 3)).ljust(6), 
                              str(round(response.max(),3)).ljust(6), 
                              str(round(response.min(), 3)).ljust(6),
                              str(round(np.sum(np.abs(response)), 3)).ljust(6))
                    
                    
                    response += self.a_1 # adding the spontaneous firing rate constant
                    response = self.deactivation_map[pos]*(response > 0) * response * response # half-squaring rectification
                    
                    response_path = self.get_resp_path(theta, v, lambd, phi)
                   
                    np.save(response_path, response)  
            
                
                    normalisation += response 
        return normalisation

###################################################################################################
### 2 methods to retrieve the response of the simple cells stored in the harddrive
###     1. get_filter: retrieves the filter (in real form) from either the position in the map
###        or the explicit arguments
###     2. get_filter_complex: same role, except that the Fourier version of the filter is returned
###     3. get_response: retrieves the actual simple cell response, normalising the output response.
###################################################################################################          
        
    
    def get_filter(self, *args):
        """
            Retrieves a filter, given 2 different types of input:
                1. The position of the filter between 0 and N_filters, as ordered
                   in the filter_map argument.
                2. The explicit combination of theta, v, lambd and psi that define a filter.
                
            If the input does not match one of those descriptions, the method will return an exception.
            Otherwise, it will return the filter if it was found.
        """
        if len(args) == 1 and isinstance(args[0], (int, np.int_)):
            pos = args[0]
            
            if pos > self.N_filters or pos < 0:
                raise ValueError('The argument should be in the range [0, '+str(self.N_filters)+'[.')
            
            theta, v, lambd, phi = self.filter_map_rad[pos]
        
        elif len(args) == 4 and all(isinstance(arg, (float,int,np.int_,np.float_)) for arg in args):
            
            [theta, v, lambd, phi] = args
        else:
            raise TypeError('Wrong type of input for the method get_filter.'+
                            'Please make sure you either pass an integer or all 4 ' +
                            'parameters uniquely identifying the desired filter.')
        
        sigma = self.filt_width*lambd 
                            
            # Compute the psi angle from the required velocity and c constant : v = -c*tan(psi)
        psi = np.arctan(v/self.c) 
                
            # k*cos(psi) = 2*pi/lambda (k*cos(psi) due to time shift)
        frequency = 2*np.pi/lambd/np.cos(psi) 
        _, _, _, ker = gabor_3d_px(size = self.im_shape, x_sig = sigma, 
                                       y_sig = sigma, alpha = self.alpha, k = frequency,
                                       c = self.c, theta = theta, psi = psi, phi = phi, 
                                       causal_time = self.causal_time)
        return ker

    
    def get_filter_complex(self, *args):
        """
            Short method to retrieve directly the real version of the filter from the complex one.
        """
        return np.real(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(self.get_filter(*args)))))
    
    def get_response(self, *args):
        """
            Retrieves the locally stored response. Has to be run after the 
            `compute_responses` method, or the result will not be found.
        """
        path = self.get_resp_path(*args)
        
        normalisation = np.load(self.directory_cell + '/normalisation.npy')
        response = np.load(path)      
        
        return self.K*response/normalisation
    
###################################################################################################
### 4 utility methods, doing some useful manipulations
###     1. int_to_str: converts an integer to its correct string version (1.0 -> '1')
###     2. get_filter_map: formats the filter_map into degrees from radians and fixs the velocity to 
###                        two decimals, which is useful for the printing to a file.
###     3. get_resp_path: Constructs the path to the file for a response from the parameters theta, v
################################################################################################### 

    def int_to_str(self, val):
        """
            Checks whether the given number is an integer, and if it is, cuts the decimal part. 
            It is useful to make sure that having inputs like 2 and 2.0 will give filters which
            have the same file name, which would not be the case otherwise.
        """
        if isinstance(val, (np.int_, int)):
            return str(val)
        else:
            return str(int(round(val,3))) if round(val,3).is_integer() else str(round(val,3)) 
        
    def get_filter_map(self):
        filter_map = self.filter_map_rad.copy()
        filter_map[:,0] = (np.array(filter_map[:,0])*180/np.pi).astype(int)
        filter_map[:,1] = np.round(filter_map[:,1],2)
        filter_map[:,3] = (np.array(filter_map[:,3])*180/np.pi).astype(int)
        
        return filter_map
    
    
    def get_resp_path(self, *args):
        """
            Returns the relative path to which a particular response is/will be stored.
        """
        #print(os.getpid(),args)
        if len(args) == 1 and isinstance(args[0], (int,np.int_)):
            theta, v, lambd, phi = self.filter_map_rad[args[0]]
        elif len(args) == 4 and all(isinstance(arg, (float,int,np.int_,np.float_)) for arg in args):
            theta, v, lambd, phi = args[0], args[1], args[2], args[3]
        else:
            raise TypeError('Wrong type of input for the method get_filter. '+
                            'Please make sure you either pass an integer or '+ 
                            'all 4 parameters uniquely identifying the desired Simple cell response.')
        
        im_shape_str = self.im_shape.astype(str)
        
        alpha_str = self.int_to_str(self.alpha * 100)
        c_str = self.int_to_str(self.c)
        
        theta_str = str(int(theta * 180 / np.pi)); v_str = self.int_to_str(v)        
        lambd_str = self.int_to_str(lambd) ; phi_str = str(int(phi * 180 / np.pi))
        response_path = (self.directory_cell + '/' + self.file_cell + '_' + im_shape_str[0] + '_' + 
                       im_shape_str[1] + '_' + im_shape_str[2])
             
        if self.causal_time:
            response_path += '_causal'  
            
        response_path += ('_' + alpha_str + '_' + c_str +
            '_' +  theta_str + '_' + v_str + '_' + lambd_str + '_' + phi_str + '.npy')  
        
        return response_path