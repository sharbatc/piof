import numpy as np
from mt_cell import*
from mstd_cell import *



def optic_flow(stimulus):
    """
        Takes a stimulus given an input, and generates an MT cell class instance with the parameters
        defined in the function. Then, extracts the optic flow from the responses of the MT neurons
        to the stimulus given in input.
    """
    
    ## 1. DEFINITION OF THE PARAMETERS NEEDED FOR THE SIMULATION.
    shape = stimulus.shape  
    
    # *v_sensitivity:* velocity and direction selectivity of the MT cells generated:
    # [[0], [1], [2]] would mean a velocity of magnitude 2 in the direction [0,1] in the x-y plane.
    # Here, there are 25 different MT neurons generated.
    v_sensitivity = np.array([[0,  0, 1, 0,-1, 1, 1,-1, -1,  0, 1, 0,-1,  1, 1,-1, -1,  0,  1,  0, -1,  1,  1, -1, -1],
                              [0,  1, 0,-1, 0, 1,-1, 1, -1,  1, 0,-1, 0,  1,-1, 1, -1,  1,  0, -1,  0,  1, -1,  1, -1],
                              [0,  1, 1, 1, 1, 1, 1, 1,  1,  5, 5, 5, 5,  5, 5, 5,  5,  10, 10, 10, 10, 10, 10, 10, 10]])
    
    # The wavelengths sensitivity of the V1 cells that are pooled for a MT cell: each MT cell will pool
    # responses from the V1 cells which have these wavelengths.
    # Same goes for the phases, except that 4 phases in steps of 90 degrees are required to get the
    # correct complex cell responses.
    wavelengths = np.array([3, 10, 20])
    phases = np.pi*np.array([-0.5, 0, 0.5, 1])

    # Parameters for the MT cell class. Those are respectively:
    # - a_1: spontaneous firing for simple cells.
    # - s_1: semi-saturation constant for simple cells.
    # - K_1: maximal reachable response for simple cells
    # - a_2: spontaneous firing for MT cells.
    # - s_2: semi-saturation constant for MT cells.
    # - K_2: maximal reachable response for MT cells
    # - thresh_inhib: threshold below which the responses are 
    #                 considered as inhibitory.
    # - w_inhib: synaptic weight for inhibition.
    params =  np.array([0.07, 0.2, 4, 0.8, 1, 1.8, 0.5, 1])
    
    alpha = .07
    c = 1
    
    deactivation_rate = 0.0  
    filt_width = 0.56
    
    ## 2. DEFINING THE INSTANCE OF THE MT CELL CLASS
    mt_cell = MT_cell(shape, v_sensitivity, wavelengths, phases, c, alpha, params, 
                                          deactivation_rate = deactivation_rate, filt_width = filt_width,
                                          causal_time = True, verbose = False)
    
    ## 3. COMPUTING THE RESPONSE OF THE MT CELLS TO THE STIMULUS AND EXTRACTING THE OPTIC FLOW FROM IT.
    mt_cell.compute_responses(stimulus)
            
    v_x, v_y = mt_cell.get_optic_flow_combined()
    
    return v_x, v_y

## This one is used to generate some random optic flows for getting the mmonocular ego motion extraction to work. 
def calculate_A(f,x,y):
    A = np.array([[-f,0,x],[0,-f,y]])
    assert A.shape == (2,3), "Wrong shape of A(x,y) matrix, should be {} but is {}".format((2,3),A.shape)
    return A

def calculate_B(f,x,y):
    B = np.array([[(x*y)/f, -(f + (x*x)/f), y],[f + (y*y)/f, -(x*y)/f, -x]])
    assert B.shape == (2,3), "Wrong shape of B(x,y) matrix, should be {} but is {}".format((2,3),B.shape)
    return B

# Will generate from arbitrary depth and given translation and rotation, a optic flow field at (x,y) 
def get_point_optic_flow(f,x,y,T,Omega):
    assert T.shape == (3,1), "Wrong shape of T matrix, should be {} but is".format((3,1),T.shape)
    assert Omega.shape == (3,1), "Wrong shape of Omega matrix, should be {} but is".format((3,1),Omega.shape)

    Z = np.random.rand() #depth, arbitrary
    A_xy = calculate_A(f,x,y)
    B_xy = calculate_B(f,x,y)

    translation = (1/Z)*np.dot(A_xy,T)
    rotation = np.dot(B_xy,Omega)

    v_xy = translation + rotation 
    assert v_xy.shape == (2,1), "Wrong shape of velocity matrix, should be {} but is".format((2,1),v_xy.shape)
    return v_xy

# Will generate the whole optic flow field for a scene size of (X,Y) pixels #can and should be parallelised 
def get_artificial_optic_flow(f,X,Y,T,Omega):
    assert T.shape == (3,1), "Wrong shape of T matrix, should be {} but is".format((3,1),T.shape)
    assert Omega.shape == (3,1), "Wrong shape of Omega matrix, should be {} but is".format((3,1),Omega.shape)

    O_field = np.zeros((X,Y))

    for i in np.arange(0,X,1):
        for j in np.arange(0,Y,1):
            O_field(i,j) = get_point_optic_flow(f,i,j,T,Omega)