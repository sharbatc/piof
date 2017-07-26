from pylink import *
import socket
import os
import shutil
from time import sleep
import gc
import numpy as np
import timeit
import matplotlib.pyplot as plt
from lxml import etree
from mt_cell import*
from mstd_cell import *
from IPython import display
import os
##
## These first functions are called by the python server that is created below, 
## parse and then compute the required results before sending a response 
## to the Unity simulation for the next step of simulation.
##

def prepareSimulation(data, pos_cube, alpha, tol, verbose):
    """
        Prepares the simulation by transmitting to unity the number of frames 
        needed in order for the video to be long enough (If it is not, the filters
        response will not be complete.).
    """
    strlist=data.split()
    
    im_shape = np.array(strlist[1:3]).astype(int)
    OFD_px = float(strlist[3])

    # Setting the size of the time kernel based on the desired tolerance.
    a_t = alpha/np.sqrt(2*np.pi)
    t_min = np.ceil(np.sqrt(-2/alpha**2*np.log(tol/a_t))).astype(int)
    t_min = t_min+1 if t_min%2==0 else t_min
    
    #t_min = 51
    
    im_shape = np.array(np.r_[t_min, im_shape])

    if (verbose):
        print('Target video shape', im_shape, ' - OFD_px =', OFD_px, end = ' - ')  
    
    tosend = ('initialisation finished '+ str(t_min) + ' ' + 
             str(pos_cube[0]) + ' ' + str(pos_cube[1]) +'\n' )
    client.sendall(tosend.encode())
    
    return im_shape, OFD_px
    
def simulationStep(data, im_shape,  params_simulation, trajectory, virt_traj, cross_pos, mt_cell, mstd, count_step, verbose):
    """
        Performing a complete step of simulation:
            
            1. Extracts:
                - im_shape: shape of saved video
                - phi: angle at which the camera is currently looking
                - psi_g: the angle to the goal
                - v: the currently velocity of motion
                - dt: the time step (frames will be separated by dt seconds)
                
            2. Computes the update of the position in two different fashions:
                a) If neuronal is true: uses the neuronal implementation to compute
                   the MT and MSTd cells responses in order to get the Focus of Expansion.
                   The resulting computation is very SLOW. 
                b) If neuronal is false: takes the ideal position of the Focus of Expansion, 
                   which is the centre of the image + the shift in pixels of the FoE. 
                   The resulting computation is fast.
            3. Transmits to unity the update to make to the angle phi
            
        Moreover, it records the position of the cross from the neuronal model and saves it at the end of the simulation. 
        The next time, if a matching cross position file is found, the cross position from the file is used instead of
        performing the computations with the neuronal model once more.
    """

    # Extracting the parameters from the list.
    k, w, OFD, pos_cube, v, dist_max, dt , tol, horizon_y, neuronal, load_results, folder_results, OFD_px = params_simulation
    
    data_path = (folder_results + '/cross_pos_' + str(k) + '_' + str(w) + '_' + 
             str(OFD) + '_' + str(pos_cube[0]) + '_' + str(pos_cube[1]) + '_' + str(dt) + '_' + str(v) + '.npy')
    #Parsing the updated data from the string received from Unity
    strlist = data.split()
    numbers = np.array(strlist[1:]).astype(float)
    
    pos_cube_x = numbers[0]  

    posxy = np.array(numbers[1:3])
    realxy = np.array(numbers[3:])
    
    virt_traj.append(posxy)
    trajectory.append(realxy)
    
    stimulus = np.zeros(im_shape)
    if (verbose):
        print("Simulation step:", end = ' ')
        
    ## 1st type of simulation
    if neuronal:
        
        if (not os.path.isdir(folder_results)) and load_results:
            os.mkdir(folder_results)
                
                
        if  load_results and (os.path.isfile(data_path)):
            cross_pos_res = np.load(data_path)
            x_pos_flow = cross_pos_res[count_step]

        else:
            if verbose and load_results and count_step == 0:
                print('Loading the cross_position_results failed, proceeding with a normal simulation.')  
            params_simulation[10] = False    
            # 1. Getting the stimulus
            stimulus = parse_frames(im_shape)
            
            np.save('my_stimulus.npy',stimulus)
            
            
            # 2. Extracting the angle from the optic flow (Takes a looooooooong time)
            if (mt_cell.shape != im_shape).any():
                print('New stimulus shape: generating the mt cell instance again.')
                mt_cell.shape = im_shape
                mt_cell.generate_comp_ageing()
            mt_cell.compute_responses(stimulus)
            
            
            v_x, v_y = mt_cell.get_optic_flow_direction()
    
            # Setting all the information of motion in the sky to zero, as it is actually immobile.
            v_x[:, :,:horizon_y] = 0 
            v_y[:, :,:horizon_y] = 0
            
            if(mstd.shape != im_shape[1:]).any():
                print('New stimulus shape: generating the mstd cell instance again.')
                mstd.init_again(im_shape[1:], im_shape[1]/2, horizon_y, 7)
            
            resp = mstd.compute_correlation(v_x, v_y)
            
            x_pos_flow = mstd.get_heading_x()
            # Updating the position of the cross with the latest computed cross position.
            cross_list = list()
            
            for i in range(im_shape[0]):
                [x_point, y_point] = mstd.pattern_map[np.unravel_index(np.argmax(resp[i]),resp[i].shape)]
                cross_list.append(x_point)        
            cross_pos.append(cross_list)                
            np.save(data_path, cross_pos)
        x_pos_ego = im_shape[1]/2        
        # 3. Update the angle for the next step of simulation
        dpos = dt * k *(w*v*(x_pos_flow - pos_cube_x) + (x_pos_ego-pos_cube_x))
             
        
    ## 2nd type of simulation
    else:
        
        x_pos_ego = im_shape[1]/2
        x_pos_flow = x_pos_ego + OFD_px
        dpos = dt * k *(w*v*(x_pos_flow - pos_cube_x) + (x_pos_ego - pos_cube_x))
        
        # Updating the position of the cross with the latest computed cross position.
        cross_pos.append(x_pos_flow)
             
    if (verbose):
        print ("pos_cube_x=", np.round(pos_cube_x,2), "pos_cross_x = ", np.round(x_pos_flow,2), 
               "real heading direction = ", np.round(x_pos_ego,2),  " dpos=", np.round(dpos,2), end = ' - ');   
    
    tosend = "simulation step finished " + str(dpos) + "\n"
    client.sendall(tosend.encode())
    
    
    count_step +=1    
    
    return trajectory, virt_traj, cross_pos, count_step, params_simulation#, stimulus

def format_time(time):
    """
        Formats the time to write it with the print function()
    """
    if time < 60:
        return str(np.round(time,1)) + 's'
    else:
        minutes = int(time//60)
        seconds = int(time%60)
        return str(minutes) + 'm ' + str(seconds) + 's' 
    
def parse_frames(im_shape):
    """
        For a given shape of image im_shape, retrieves from the xml file
        the corresponding video and formats it in the way we need for the
        simulation.
    """

    # We tranpose the image afterwards to make it usable with our definition.
    n_frames, width, height  = im_shape
    image = np.zeros((n_frames, height, width));
    
    tree = etree.parse("../unity/OpticFlow/frames.xml")
    for i, array in enumerate(tree.xpath("/ArrayOfArrayOfFloat/ArrayOfFloat")):
        for j, entries in enumerate(array.xpath("float")):
            #print(i,j)
            #print(i,j, j//width, j%width, float(entries.text))
            image[i, height-1-j//width,j%width ] = float(entries.text)
    image = image.transpose(0,2,1)
    #image = image[:,:,::-1]
    return  (image*255).astype(int)

### 
### Server for communication between Unity and Python computations.
###


def run_python_server(params_simulation, mt_cell, mstd, plot_flag = False, verbose = False):
    """
        Along with the unity script, runs a simulation of heading using optic flow for different parameters, hereafter described.
        
        1. params_simulation, contains different parameters for the simulation:
            - k: turning rate for the differential equation.
            - w: amount of optic flow that is extracted from the scene (used in the differential equation).
            - OFD: Optic Flow Drift, set it to zero if you want the Focus of Expansion to match the center of the image
                    captured by the camera. A positive OFD will shift the FoE to the right. The unit is degrees (the pixel
                    value corresponding will be stored in the OFD_px variable.
            - pos_cube: the position of the cube (which is a circle at the moment ^^'), to be put where you want to have your goal.
            - speed: the velocity of motion in m/s
            - dist_max: the maximal number of steps that will be performed. Useful if you don't want to steer all the way 
                        to the goal, but want to stop before.
            - dt: the time step of the simulation. Each python computation will consist of a chunk of images spanning
                   over n_frames each separated by dt ([s])
            - tol: the tolerance of error for the space kernel (will tell how large the temporal dimension of the Gabor filter
                   should be in order to reach a certain precision in the time kernel).
            - horizon_y: the position of the horizon in pixels on the sampled image (to be set manually at the moment, though
                         it should not be complicated to determine it programatically)
            - neuronal: a boolean. True means that we want to perform the simulation using the optic flow Focus of Expansion 
                        from the implemented model, which takes a long time to run. False means that we suppose the FoE to
                        always be at the same place (centre of the image + OFD_px).
            - load_results: if you want to load the FoE position which was computed during a neuronal simulation (useful in
                            order to avoid redoing the entire simulation each time, and rather loads the FoE positions extracted
                            from the neuronal model.
            - folder_results: where the results will be stored.
       2. params_mt, v_sensitivity, wavelengths, phases, alpha, c are MT cells parameters, and are more explicitely describe
           in the MT cell class.
           
       Note that we record a trajectory and a virtual_trajectory, because for the sake of the simulation, in order to reproduce
       an Optic Flow which is shifted from 10° from the FoE while walking straight to the target, 
       we have to use some artifacts, namely, move the goal at a certain velocity to create this illusion.
    """
    # Extracting the parameters from the parameters list
    k,w, OFD, pos_cube, speed, dist_max, dt , tol, horizon_y, neuronal, load_results, folder_results = params_simulation
    
    # Defining the structures that will contain the trajectories of the subject
    trajectory = list()
    virt_traj = list()
    cross_pos = list()
    
    stim_list = list()
    # Server information
    host = ''
    port = 8002

    backlog = 2
    size = 1024
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host,port))
    s.listen(backlog)
    stop=0;
    stopin=True

    im_shape = np.zeros((2))
    start_time = timeit.default_timer()   
    elapsed = 0
    OFD_px = 0
    count_step = 0
    
    if (plot_flag):
        fig,ax = plt.subplots(1,1)
        ax.set_title('Heading towards the goal with OFD = ' + str(OFD) + ', k=' + str(k) + ', w='+ str(w))
    # While the server is running    
    while stop == 0:
        try:
            if (verbose):
                print('Waiting for Unity.')
            global client
            
            client, address = s.accept()
            if (verbose):
                print('Unity connected.')
            stopin=True
            isconnected = True
            
            # For the first connection, we send the string below to Unity
            if isconnected:
     
                tosend = 'ready '+ str(OFD) + ' ' + str(neuronal) + ' ' + str(speed) + ' ' + str(dist_max) + ' ' + str(dt) +'\n'
                client.sendall(tosend.encode())
            
            # Then, we go into this loop to communicate with unity until the connection is broken or the quit message is received.
            while stopin:
                
                data = client.recv(size).decode()
                data = data.rstrip('\r\n')
                
                # If the data string received is not empty
                if data:
                    
                    # For the first connection with the server, initialisation phase of the simulation
                    if data.find("ready") > -1:
                        im_shape, OFD_px = prepareSimulation(data, pos_cube, mt_cell.alpha, tol, verbose)
                        params_simulation.append(OFD_px)
                        
                        #Plotting part from the position of the goal
                        trajectory.append([0,0])
                        virt_traj.append([0,0])
                        
                        if (plot_flag):
                            traj_plot = np.array(trajectory)
                            ax.set_ylim([0, max(traj_plot[:,1].max(),1)])
                            ax.set_xlim([np.min([traj_plot[:,0].min(),pos_cube[0],-1]), 
                                         np.max([traj_plot[:,0].max(), pos_cube[0],1])])
                    
    
                            ax.scatter(pos_cube[0], pos_cube[1], color = 'r', marker = 'x')
                            line, = ax.plot(traj_plot[:,0], traj_plot[:,1], color= 'k', marker = '.')
                    
                    # Called every step of the simulation
                    elif data.find("step") > -1:
                    
                        trajectory, virt_traj, cross_pos, count_step, params_simulation = simulationStep(data, im_shape, 
                                                                          params_simulation, trajectory, virt_traj, 
                                                                          cross_pos,  mt_cell, mstd, count_step, verbose)
                        #stim_list.append(stimulus)
                        if (plot_flag):
                            traj_plot = np.array(trajectory)
                            line.set_xdata(traj_plot[:,0])
                            line.set_ydata(traj_plot[:,1])
                            
                            ax.set_ylim([0, max(traj_plot[:,1].max(),1)])
                            ax.set_xlim([np.min([traj_plot[:,0].min(),pos_cube[0],-1]), 
                                         np.max([traj_plot[:,0].max(), pos_cube[0],1])])   
                            fig.canvas.draw()
                        
                    # Terminates the simulation (Received either upon end of the simulation or if Unity was quit).
                    elif data == "quit":
                        tosend = 'terminated' + '\n'
                        client.sendall(tosend.encode())
                        if verbose:
                            print('Terminated' , end = ' - ')
                        sleep(0.1);
                        client.close()
                        stop = 1;
                        stopin = False;
                    
                    elapsed = timeit.default_timer() - start_time
                    if (verbose):
                        print('Elapsed time: ', format_time(elapsed))
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            try:
                if connection:
                    client.close()
            except: pass
            break
    s.shutdown
    s.close();
    
    return np.array(trajectory), np.array(virt_traj), np.array(cross_pos), #np.concatenate(stim_list)


##
## Heading error computation functions: Once the trajectory is obtained from the unity simulation, 
## we compute the **heading error** at a location by taking the scalar product between the egocentric 
## direction to the goal and the actual heading direction.
##

def get_HE_arrows(trajectory, pos_cube):
    HE = np.zeros(trajectory.shape[0])
    HE[0] = 0
    
    curr_dir = (trajectory[1:] - trajectory[:-1])
    # normalising the vectors
    curr_dir /= np.sqrt(np.einsum('ij,ij->i', curr_dir, curr_dir))[:,None]
    
    goal_dir = np.array([pos_cube - curr_pos for curr_pos in trajectory[1:]])
    goal_dir/= np.sqrt(np.einsum('ij,ij->i', goal_dir, goal_dir))[:,None]
    return curr_dir, goal_dir
    
def get_HE(pos_y, trajectory, pos_cube):
    pos_x = np.interp(pos_y, trajectory[:,1], trajectory[:,0])
    
    curr_dir = (trajectory[1:] - trajectory[:-1])
    # normalising the vectors
    curr_dir /= np.sqrt(np.einsum('ij,ij->i', curr_dir, curr_dir))[:,None]
    
    min_pos = np.min(np.where(trajectory[:, 1] > pos_y))
    dir_traj =  curr_dir[min_pos]
        
    dir_cube = pos_cube - np.array([pos_x, pos_y])
    dir_cube /= np.linalg.norm(dir_cube)
    
    HE = np.arccos(np.dot(dir_traj, dir_cube))*180/np.pi
    return HE

def get_traj_drift_vertical(pos_y, trajectory):
    pos_x = np.interp(pos_y, trajectory[:,1], trajectory[:,0])
    return pos_x