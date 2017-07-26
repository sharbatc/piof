import numpy as np
from complex_cell import *
from gabor_kernel import *
import shutil
from numpy.linalg import norm
from multiprocessing import Pool
from functools import partial


CPU_NUMBER = os.cpu_count()

class MT_cell(object):
    """
        Creates a bank of MT neurons, which respond to stimuli moving at a certain velocity in a certain direction.
        The response is obtained from combining the responses of complex cells (in our case, we compute the response
        directly from simple cells, as complex cell only combine the responses of simple cells with 4 different
        phases spaced in steps of pi/2). 
        
        The attributes of the class are detailled in the __init__ method.
    """
    
    def __init__(self, im_shape, v_sensitivity, wavelengths, phases, c, alpha,  comp_mt_params,
                 deactivation_rate = 0.0, filt_width = 0.56,  causal_time = False, 
                 mt_directory = 'mt_cell', mt_file = 'response', comp_directory = 'complex_cell',
                 comp_file = 'response', verbose = False):
        """
            Detail of the inputs to the class:
            - im_shape: shape of the stimulus to which the MT_cell will react.
            - v_senstivity: an array of the shape 3 x N_MT_neurons, the first and second line give the
                            direction of the stimulus to which the neuron is sensitive (not normalised)
                            and the third line gives the magnitude of the velocity to which the neuron will react.
            - wavelengths: an array containing the different wavelengths to which the cells are sensitive.
            - phases: same principle than the wavelengths.
            - c: useful when time is converted to space and vice versa in the rotations, to account for the
                 different resolutions.
            - alpha: spread of the filter in time.
            - comp_mt_params: a set of 6 parameters, accounting for the spontaneous activations, semi-
                              saturation constants and maximal responses of both complex and MT cells.
            - deactivation rate: percentage of the repsonses that will be deactivated after computation
                                 in order to simulate ageing of the brain.
            - filt_width: ratio of the width of the filter with respect to the wavelength 
                          (cf. the computation of sigma further down in the compute_responses method)
            - directory and file: where the results will be stored
            - verbose: to get some extra information at execution.
              
            Detail of the attributes derivated from those:
            - mt_v: array of velocity sensitivities formatted into the correct way: i.e. 2D vector with 
                    the correctly expected magnitude. (e.g. Move from [ [1] [1] [2] ] to [sqrt 2, sqrt 2])
            - N_cells: number of mt cells that are simualted
            - mt_map: list which contains the angles theta and phi of the four complex cells that have to be
                      combined into one mt cell response.
            - thetas and velocities: parameters that will be passed to the complex cell declaration.
            - unique_pos: For each MT cell, tells whether the spatiotemporal parameters of the complex
                          cells appear for the first time or where was the first apparition. If it appears
                          for the first time, unique_pos will contain a None at the entry and if it already
                          appeared previously in mt_map, it will contain the location of the first apparition 
                          of this spatiotemporal orientation. This is especially useful for inhibition and
                          deactivation of cells.
            - deactivation_map: for each MT cell, mask that will be multiplied with the response to deactivate
                                a certain percentage of the reponses. 
        """

        ## 1. INITIALISE DIVERSE ATTRIBUTES OF THE CLASS
        self.shape = np.array(im_shape)

        self.wavelengths = wavelengths
        self.phases = phases
        self.c = c
        self.alpha = alpha
        self.a1, self.s1, self.K1, self.a2, self.s2, self.K2, self.inhib_resp, self.inhib_weight = comp_mt_params
        self.directory = mt_directory
        self.file = mt_file
        self.stimulus = np.zeros(im_shape)
        self.verbose = verbose
        self.filt_width = filt_width
        self.deactivation_rate = deactivation_rate         
        
        ## 2. COMPUTE THE SET OF ANGLES AND VELOCITIES NEEDED
        
        
        directions = v_sensitivity[:2, :].T
        if v_sensitivity.shape[0] > 2:
            magnitudes = v_sensitivity[2, :]
        else:
            magnitudes = np.array([np.sqrt(np.dot(v, v)) for v in directions])
            
        self. v_max = magnitudes.max()
        # Computing the 2D vector of velocity from the directions and magnitudes
        self.mt_v = np.array([magn*v/np.sqrt(np.dot(v, v)) if np.dot(v, v) != 0 else [0, 0] 
                 for v, magn in zip(directions, magnitudes)])
        
        # Declaring a few more attributes of the class
        self.N_cells = len(magnitudes)
        self.mt_map = np.zeros((self.N_cells, 4, 2))
        self.thetas = np.zeros((self.N_cells * 4))
        psis = np.zeros((self.N_cells* 4))
        
        for i, v in enumerate(self.mt_v):
            if (v == 0).all():
                angles_v = np.array([[0,0],[np.pi/4,0],[np.pi/2,0],[3*np.pi/4,0]])
            else:
                angles_v =  np.array(self.get_spacetime_angles(v))
            self.mt_map[i] = angles_v
            self.thetas[4*i:4*(i + 1)] = angles_v[:, 0]
            psis[4*i:4*(i + 1)] = angles_v[:, 1]
            
        self.velocities = self.c * np.tan(psis)

        ## 3. COMPUTE THE CROSS PRODUCT BETWEEN THE PLANE ORIENTATION AND THE DIRECTION OF THE DIFFERENT VECTORS.
        self.compute_cross_product()
        
        ## 4. COMPUTE THE MAP WHICH SAYS IF THE COMPLEX CELL COMPOSING AN MT CELL NEEDS TO BE RECOMPUTED
        self.unique_pos = list()

        for i, mt_cell in enumerate(self.mt_map):
            cell_pos_list = list()
            for j,orientation in enumerate(mt_cell):
                cell_pos, comp_pos = np.where(np.all(np.abs(self.mt_map-orientation)<1e-15,axis=2))
                cell_pos, comp_pos = cell_pos[0], comp_pos[0]
                if cell_pos == i and comp_pos == j: 
                    cell_pos_list.append([cell_pos, comp_pos])
                else:
                    cell_pos_list.append(None)
            self.unique_pos.append(cell_pos_list)
        self.unique_pos = np.array(self.unique_pos)
               
        ## 5. GENERATE THE COMPLEX CELLS AND THE AGEING MAP
        self.comp_directory = comp_directory
        self.comp_file = comp_file
        self.causal_time = causal_time
        self.wavelengths = wavelengths
        
        if (self.shape != [1,1,1]).all():
            self.generate_comp_ageing()
        else:
            print('At least one of the components of the shape of the image is invalid. Please change the attribute shape and run the method generate_comp_ageing to terminate the initialisation of the class.')
    
    def generate_comp_ageing(self):
        """
            Method which generates the complex cells and the deactivation linked to ageing.
        """
        self.comp_cell = ComplexCell(self.shape, self.thetas, self.velocities, self.wavelengths, 
                                     self.phases, self.c, self.alpha, self.a1, self.s1, self.K1, self.deactivation_rate,
                                     self.filt_width, directory_cell = self.comp_directory, file_cell = self.comp_file,
                                     filters_theta_v = True, causal_time = self.causal_time, verbose = self.verbose)
        
        if self.deactivation_rate > 0:
            self.deactivation_map = np.random.choice([0, 1], size=(int(self.N_cells),self.shape[1], self.shape[2]), 
                                                  p=[ self.deactivation_rate, 1-self.deactivation_rate])
    
    def clear_directory(self):
        """
            Removes the folder in which the results are stored
        """
        self.comp_cell.clear_directory()
        
        if os.path.isdir(self.directory):
            shutil.rmtree(self.directory)
    
###################################################################################################
### 3 methods that compute 4 couples of angles from a velocity v = (v_x, v_y), as described in the 
### appendix of the paper of Simoncelli and Heeger, 1996.
###     1. get_angles:           maps a space-time orientation vector in Fourier space to the corresponding 
###                              angles (theta, psi) which generate a Gabor filter
###     2. get_orientations:     from a given velocity v, computes the 4 vectors u_1, u_2, u_3, u_4
###     3. get_spacetime_angles: calls both previous methods to return the 4 couples of angles (theta, psi)
###################################################################################################      
        
    def get_angles(self, vect):
        """
            Transform a spatio-temporal unit vector for preferred orientation of 
            the filter into the two rotation angles to which it corresponds.
        """
        x_1, x_2, x_3 = vect
        
        return  -np.arctan2(x_2,x_1),  np.arcsin(-x_3)

    def get_orientations(self, v):
        """
            Computes the 4 vectors required to get a ring in Fourier domain which
            will detect a certain velocity
        """
        v_x, v_y = v
        if v_x == 0 and v_y == 0:
            u_1 = np.array([1, 0, 0])
            u_2 = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])
            u_3 = np.array([0, 1, 0])
            u_4 = np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0])
            
        else:
            s = np.sqrt(v_x**2 + v_y ** 2)
    
            u_1 = np.array([-v_x, -v_y, s**2])/np.sqrt(s**4 + s**2)
            u_2 = np.array([-v_y, v_x, 0])/s
            u_3 = (u_1 + u_2) / np.sqrt(2)
            u_4 = (u_1 - u_2) / np.sqrt(2)
        
        return u_1, u_2, u_3, u_4
    
    def get_spacetime_angles(self, v):
        """
            Extracts the 4 spatio-temporal unit vectors that we need in order
            to obtain a MT cell sensitive to the velocity v (2D vector with 
            the preferred x and y velocities of the cell).
        """
        u_1, u_2, u_3, u_4 = self.get_orientations(v)
    
        return self.get_angles(u_1), self.get_angles(u_2), self.get_angles(u_3), self.get_angles(u_4)
    
###################################################################################################
### 4 methods to compute the MT cells responses to a given stimulus.
###     
###     1. compute_responses_raw:    Computes the raw MT cell response for a subset of the total
###                                  MT cells that are simulated, and stores the response in the
###                                  self.directory directory, with file name raw_file
###     2. computes_responses_inhib: Based on the raw MT cell responses, computes the inhibition
###                                  from the V1 neurons which are not close to the preferred
###                                  velocity plane and stores it in the same directory, with 
###                                  file name self.file.
###     3. compute_responses:        Combines the two previous methods, and splits the whole MT cell
###                                  set into different subsets, which are delegated to different 
###                                  processes. 
###     4. get_response:             Loads the response which was computed by a call at compute_responses
###                                  normalises it and returns it.
###################################################################################################   
            
    def compute_responses_raw(self, cell_split, raw_file = 'mt_raw'):
        """
            Computes the raw response of the MT cells in the subset cell_split, and stores it
            with a file name starting with mt_raw. The computation is simply the sum of the responses
            of the complex cells with 4 different phases, all possible wavelengths along the 4 preferred
            directions which are given as 4 couples of (theta, psi) angles.
            
            The verbose mode will give details about the magnitude, mean, max, min of responses.
        """
        for num_cell in cell_split:
       
            if self.verbose:
                print(os.getpid(),'- Computing the raw response of the MT Cell n°',num_cell, 
                      'sensitive to v = [', round(self.mt_v[num_cell, 0],3), ',',round(self.mt_v[num_cell,1],3),
                      '] made of the following responses.\n','->'.ljust(9),
                      'Theta'.ljust(5),'Psi'.ljust(5), 
                      '|', 'sum'.ljust(15),'max'.ljust(5),'min'.ljust(5),'mean > 0'.ljust(10))
                
            response = np.zeros(self.shape)
            
            for i in range(4):
                theta, psi = self.mt_map[num_cell, i]
                v = self.c * np.tan(psi)
                response += self.comp_cell.get_response(theta, v)
                    
                if self.verbose:
                    print((str(os.getpid())+' -').ljust(10), str(int(theta*180/np.pi)).ljust(5), 
                        str(int(psi*180/np.pi)).ljust(5), '|', 
                        str(round(np.sum(response),3)).ljust(15),
                        str(round(response.max(), 3)).ljust(5), 
                        str(round(response.min(), 3)).ljust(5), 
                        str(round(response[response>0].mean(),3)).ljust(10))
                                   
            response_path = self.get_path(num_cell, raw_file)
            if self.deactivation_rate > 0:
                np.save(response_path, self.deactivation_map[num_cell]*response)  
            else:
                np.save(response_path, response)  
            
    def compute_responses_inhib(self, cell_split):
        """
            Compute the third and last part of the response output from the MT cells to the stimulus given as
            input of the compute_responses_raw model. Has to be ran after compute_responses_raw
            
            Compute the model output, especially adding inhibition from the V1 cells
            which are not close to a particular MT cell preferred velocity plane.
            (for each mt cell, load and subtract the responses of the cells which are not near the
            preferred velocity plane)
        """
        # 3. MT CELL WITH INHIBITION FROM OTHER CELLS
        # After getting the raw MT cell responses, we compute the inhibition
        # from the v1 cells which are not close to the velocity plane
        #print(os.getpid(), 'second_part', cell_split, cell_split is None)
        
        normalisation_response = np.zeros(self.shape)

        for i in cell_split:
           # print(os.getpid(), i-cell_split[0],'/',cell_split[-1]-cell_split[0])
                
            response_path_load = self.get_path(i, self.raw_file)
            response = np.load(response_path_load)
            
            for j in range(self.N_cells):
                if j != i: 
                    for k in range(4):
                        # We express the fact of not being close as the magnitude of the
                        # cross product between the normal vector to the velocity plane
                        # and the directional vector of the filter.
                        if self.cross_table[i,j,k] < self.inhib_resp and self.unique_pos[j,k] is not None:
                            #print(i,j,k)
                            # Retrieve and subtract the responses from filters
                            # with the same phase and wavelength
                            theta, psi = self.mt_map[j,k]
                            v = self.c * np.tan(psi)
                            response -= self.inhib_weight*self.comp_cell.get_response(theta, v)
                                
            
            response += self.a2 # adding the spontaneous firing rate constant
            response = (response > 0) * response * response # half-squaring rectification
            if self.verbose:
                print(os.getpid(),'- Computing the model response of the MT Cell n°',i, 
                      'sensitive to v = [', round(self.mt_v[i, 0],3), ',',round(self.mt_v[i,1],3),
                      '] rectified but not normalised.\n',os.getpid(),'- sum =', str(response.sum()).ljust(15),
                      'max =', str(round(response.max(), 3)).ljust(5), 
                      'min =', str(round(response.min(), 3)).ljust(5))
                     
                      
            response_path_save = self.get_path(i)
            np.save(response_path_save, response)
            
            normalisation_response += response

        return normalisation_response
        
        
    def compute_responses(self, stimulus, raw_file = 'mt_raw'):
        """
            Computes the complex cell respones if the stimulus is different that the one stored 
            (meaning that the previous iteration already worked on the same stimulus, and thus
            already computed the complex cell and raw responses). Then, splits the list of MT cells
            and delegates the computation to different processes, which will call
            compute_responses_raw and compute_responses_inhib.
        """
         
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)
        
        self.raw_file = raw_file 
        cell_split = np.array_split(np.arange(self.N_cells), CPU_NUMBER)
        
        diff_stimulus = False
        
        # If the stimulus is the same as the one stored, then the stored responses
        # already were computed. It is hence not needed to recompute them.
        if (np.array(stimulus.shape) != self.shape).any():
            raise ValueError('Wrong shape of image for this instance of MT cell')
        
        if (stimulus != self.stimulus).any():   
            diff_stimulus = True
            
            if self.verbose:
                print('Computing the complex cell responses')
            # 1. COMPLEX CELLS RESPONSES
            # This first line takes a long time to run, as we need to get all the possible responses
            # of the complex cell to the stimulus
            self.comp_cell.compute_responses(stimulus)
            self.stimulus = stimulus

        pool = Pool(CPU_NUMBER)        
        
        # 2. MT CELLS RAW RESPONSES
        
        # Computing and saving the responses to all the filters, as well
        # as the normalisation (sum of all the responses), needed for further
        # steps of the model.
        if diff_stimulus:            
            pool.map(partial(self.compute_responses_raw, raw_file = raw_file), cell_split)  
        # computing normalisation and adding the semi-saturation constant
        normalisation_response = np.sum(pool.map(self.compute_responses_inhib, cell_split),axis=0) + self.s2
        pool.close()
        pool.join()
                
        norm_path = self.directory + '/normalisation.npy' 
        np.save(norm_path, normalisation_response)   
        
    
    def get_response(self, *args):
        """
            Retrieves the locally stored response. Has to be run after the 
            `compute_responses` method, or the result will not be found.
        """
        #print(os.getpid(), 'get_response', args)
        path = self.get_path(*args)
        
        normalisation = np.load(self.directory + '/normalisation.npy')
        response = np.load(path)      
        
        return self.K2*response/normalisation

###################################################################################################
### 2 methods to return the optic flow from the MT Cell responses
###     
###     1. get_optic_flow:          Displays an arrow representing the preferred velocity of
###                                 the MT cell which fires the most, and returns the magnitude
###                                 of this firing.
###     2. get_optic_flow_combined: Combines the reponses of all the MT Cell at each location,
###                                 summing the preferred velocities according to the firing rates
###                                 of the cell at a given spatio-temporal location.
###
###################################################################################################       
    
    
    def get_optic_flow(self):
        """
            Returns the optic flow arrows that can be drawn using a quiver plot.
            The returned objects are:
            1. response: the maximal response at each pixel in space and time
            2. v_x : the x-component of the velocity of the MT cell which responds more strongly to the stimulus 
                        at each location.
            3. v_y : the y-component of the velocity of the MT cell which responds more strongly to the stimulus
                        at each location.
        """
        response = np.zeros(self.shape)
        v_x = np.zeros(self.shape)
        v_y = np.zeros(self.shape)

        for num_cell in range(self.N_cells):
            resp = self.get_response(num_cell)
            for t in range(self.shape[0]):
                for x in range(self.shape[1]):
                    for y in range(self.shape[2]):
                        if resp[t, x, y] > response[t, x, y]:
                            response[t, x, y] = resp[t, x, y]
                            v_x[t, x, y] = self.mt_v[num_cell,0]
                            v_y[t, x, y] = self.mt_v[num_cell,1]
        return response, v_x, v_y

    
    
    def get_optic_flow_combined(self):
        """
            Returns the optic flow arrows that can be drawn using a quiver plot.
            The returned objects are:
            1. response: the maximal response at each pixel in space and time
            2. v_x : the x-component of the v
        """
        response = np.zeros(self.shape)
        response_normalisation = np.zeros(self.shape)
        v_x = np.zeros(self.shape)
        v_y = np.zeros(self.shape)

        for num_cell in range(self.N_cells):
            resp = self.get_response(num_cell)
            v_x += resp*self.mt_v[num_cell,0]#/ (1.5 * self.v_max)
            v_y += resp*self.mt_v[num_cell,1]#/ (1.5 * self.v_max)
            response_normalisation += resp   
        # Divide by the normalisation of the activity in the case where everything isn't zero
        # as it would otherwise return an error.
        v_x[response_normalisation != 0] /= response_normalisation[response_normalisation != 0]
        v_y[response_normalisation != 0] /= response_normalisation[response_normalisation != 0]
        return v_x, v_y
       
    def get_optic_flow_direction(self):
        """
            Returns the direction in which the mt_cells respond more strongly. It computes
            a weighted average of the direction of the MT cells responses. 
            It is used for the MSTd cell heading direction detection.
        """
        #response = np.zeros(self.shape)
        response_normalisation = np.zeros(self.shape)
        v_x = np.zeros(self.shape)
        v_y = np.zeros(self.shape)

        for num_cell in range(self.N_cells):
            resp = self.get_response(num_cell)
            if (self.mt_v[num_cell,0] != 0 or self.mt_v[num_cell,1] != 0):
                v_x += resp*self.mt_v[num_cell,0]/norm(self.mt_v[num_cell])#/ (1.5 * self.v_max)
                v_y += resp*self.mt_v[num_cell,1]/norm(self.mt_v[num_cell])#/ (1.5 * self.v_max)
                response_normalisation += resp   
        # Divide by the normalisation of the activity in the case where everything isn't zero
        # as it would otherwise return an error.
        v_x[response_normalisation != 0] /= response_normalisation[response_normalisation != 0]
        v_y[response_normalisation != 0] /= response_normalisation[response_normalisation != 0]
        return v_x, v_y#, response_normalisation
    
###################################################################################################
### 4 methods to compute the MT cells responses to a given stimulus.
###     
###     1. int_to_str:            Converts an integer to its correct string version (1.0 -> '1')
###     2. get_path:              Constructs the path to the file from the parameters given as input
###     3. cross_plane_vect:      Computes the cross product between the normal vector of the velocity plane
###                               and a V1 neuron preferred orientation vector
###     4. compute_cross_product: Computes all possible combinations of plane normal vector and
###                               V1 neuron preferred orientation cross products. Delegates the computation
###                               to the compute_cross_parallel method.
###     5. compute_cross_product_parallel: Method which computes what is needed in compute_cross_product.
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
            return str(int(val)) if val.is_integer() else str(val) 

    def get_path(self, *args):
        """
            Returns the path to which a particular response is/will be stored from the
            velocity or position in the MT cell velocity array. The types of inputs are 
            as follows:
                1. Position in the mt_v velocity array
                2. Velocity given as (v_x, v_y)
                3. Velocity given as (v_x, v_y, magnitude)
                4. Position in the mt_v velocity array and file name
                5. Velocity given as (v_x, v_y) and file name
                6. Velocity given as (v_x, v_y, magnitude) and file name
            There are 6 cases because we store the both the raw responses and the inhibited ones
            under different file names
        """
        
        file_str = self.file
        error_msg = ('Wrong type of input for the method get_path. Please make sure you either '+
                    'pass an integer or 2/3 parameters uniquely identifying the desired filter.')
        if len(args) == 1 and isinstance(args[0], (int, np.int_)):
            v = self.mt_v[args[0]]

        elif len(args) == 2:
            if all(isinstance(arg, (float, int, np.int_, np.float_)) for arg in args):
                v = np.array(args)
                
            elif isinstance(args[0], (int, np.int_)) and isinstance(args[1], str):
                v = self.mt_v[args[0]]
                file_str = args[1]  
                
            else:
                raise TypeError(error_msg)
                
        elif len(args) == 3:
            if all(isinstance(arg, (float,int,np.int_,np.float_)) for arg in args):
                v = np.array(args[0:2])
                v = v/np.sqrt(np.dot(v,v)) if (v != 0).any() else 0
                v = args[2]*v
                
            elif all(isinstance(arg, (float, int, np.int_, np.float_)) for arg in args[:2]) and isinstance(args[2], str): 
                v = np.array(args[:2])
                file_str = args[2]
                
            else:
                raise TypeError(error_msg)
                
        elif len(args) == 4 and all(isinstance(arg, (float,int,np.int_,np.float_)) for arg in args[:3]) and (
            isinstance(args[3], str)):
                v = np.array(args[0:2])
                v = v/np.sqrt(np.dot(v,v)) if (v != 0).any() else 0
                v = args[2]*v
                file_str = args[3]
                
        else:
            raise TypeError(error_msg)
        
        v = np.round(v,3)
        
        vx = self.int_to_str(v[0])
        vy = self.int_to_str(v[1])
        
        response_path = self.directory + '/' + file_str + '_' + vx + '_' + vy + '.npy'
            
        return response_path
   
        
    def cross_plane_vect(self, v_plane, vect):
        """
            Computes the cross product between the normal preferred velocity plane
            of a MT cell (computed as the cross product between the vectors u_1 and
            u_2) and the preferred velocity vector of a V1 neuron.
            
            v_plane is the velocity we consider (e.g. v = (2, 0) ) 
            vect is a velocity vector for a V1 neuron.
        """
        if norm(vect) == 0:
            raise ValueError('Vector with norm zero passed in the method.')

        v_normal = -np.array(np.r_[v_plane, 1])
        v_normal = v_normal/np.linalg.norm(v_normal)
        
        return np.linalg.norm(np.cross(v_normal, vect/ np.linalg.norm(vect)))


    def compute_cross_product(self):
        """
            Creates a pool of workers which computes and stores all possible 
            combinations of cross products between the normals to the velocity 
            planes in Fourier space and the V1 neuron orientations.
        """
        
        pool = Pool(CPU_NUMBER)        
                
        cell_split = np.array_split(np.arange(self.N_cells), CPU_NUMBER)
        
        # Retrieving the pooling results as a list, the last step is to combine
        # it into the table we want.
        cross_list = pool.map(self.compute_cross_product_parallel, cell_split)
        
        pool.close()
        pool.join()
        
        
        self.cross_table = np.zeros((self.N_cells, self.N_cells, 4))
        
        
        for n_cells, cross in zip(cell_split, cross_list):
            self.cross_table[n_cells,:,:] = np.array(cross)
        
    def compute_cross_product_parallel(self, cell_split):
        """
            Computes and stores all possible combinations of cross products
            between the normals to the velocity planes in Fourier space and
            the V1 neuron orientations for a small part of the velocity planes.
        """    
        
        cross_table = np.zeros((len(cell_split), self.N_cells, 4))

        for i in cell_split:            
            v_plane = self.mt_v[i]
            for j in range(self.N_cells):
                u1j, u2j, u3j, u4j = self.get_orientations(self.mt_v[j])
                cross_table[i-cell_split[0], j, :] = [self.cross_plane_vect(v_plane, u1j),
                                                      self.cross_plane_vect(v_plane, u2j),
                                                      self.cross_plane_vect(v_plane, u3j),
                                                      self.cross_plane_vect(v_plane, u4j)]
    
        return cross_table