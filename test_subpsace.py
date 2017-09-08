import numpy as np
import scipy.linalg as linalg
import itertools
from JSAnimation import IPython_display
import matplotlib.pyplot as plt
from matplotlib import animation
from multiprocessing import Pool, Manager, Process,TimeoutError
import time
import os


def generate_artificial_flow(size,depth_matrix,T,omega):
    """
        Generates an artificial optic flow motion field with given translation,
        rotation, and depth matrix. The origin of motion is assumed to be at 
        the centre of the retinal patch. This is just one optic flow motion field
        that would be calculated between say, two frames of a video
    """
    v_x,v_y = np.zeros(size),np.zeros(size);
    for x in range(size[0]):
        for y in range(size[1]):
            x_scaled = np.int(x/2) #to recentre the focus point
            y_scaled = np.int(y/2) #if not done, just one quadrant visible

            ### Longuet -Higgins and Prazdny's motion field formulation begins

            A = np.array([[-f,0,x_scaled],[0,-f,y_scaled]])
            B = np.array([[(x_scaled*y_scaled)/f, -(f + (x_scaled*x_scaled)/f),y],\
                [f + (y_scaled*y_scaled)/f, -(x_scaled*y_scaled)/f, -x_scaled]])
            v_x[x,y],v_y[x,y] = np.dot(A,T)/depth_matrix[x,y] + np.dot(B,omega)

            ### Longuet-Higgins and Prazdny's motion field formulation ends
    return v_x, v_y


def calculate_CT(sample_points, T): 
    """
        As defined by Heeger and Jepson, represents a matrix collecting the 
        perspective projections for all the different sample points
    """
    N = np.shape(sample_points)[0]; #justincase
    A_T = np.zeros([2*N,N]) # preallocate ndarrays for storing the matrices
    B = np.zeros([2*N,3])
    
    for i in np.arange(0,N,1):
        x,y = sample_points[i,0],sample_points[i,1]
        x_scaled = x - (size[0]/2)
        y_scaled = y - (size[1]/2)
        
        #calculating A_T
        A = np.array([[-f,0,x],[0,-f,y]])
        AtimesT = np.dot(A,T)
        A_T[2*i,i], A_T[2*i+1,i] = AtimesT[0], AtimesT[1]
        
        #calculating B
        B[2*i] = np.array([(x*y)/f, -(f + (x*x)/f), y])
        B[2*i+1] = np.array([f + (y*y)/f, -(x*y)/f, -x])
    
    return np.concatenate((A_T,B),axis=1)

def calculate_projected_CT(sample_points,T):
    """
        Orthogonal complement would have required one more calculation in 
        calculating the null space so the projection is better to use as we
        are using the QR decomposition anyway.
    """
    N = np.shape(sample_points)[0]; #justincase
    CT = calculate_CT(sample_points,T)
    CTbar, r = np.linalg.qr(CT)
    I = np.identity(2*N)
    cc = np.dot(CTbar,np.transpose(CTbar))
    return (I - cc)

def calculate_v(sample_points):
    """
        These would change with incoming optic flow motion fields
    """
    sample_v_x, sample_v_y = v_x[sample_points[:,0],sample_points[:,1]], \
                            v_y[sample_points[:,0],sample_points[:,1]]
    v = np.vstack((sample_v_x,sample_v_y)).reshape((-1),order='F').reshape(2*N,1)
    return v

def calculate_CT_parallel(params):
    idtheta, idphi,patch_id = params[0],params[1],params[2]
    theta,phi = idtheta/100,idphi/100
    x = np.cos(theta)*np.sin(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(phi)
    T = np.array([[x],[y],[z]])
    sample_points = im_patches[:,:,patch_id]
    projected_CT = calculate_projected_CT(sample_points,T) #the time consumer
    np.save('ct_estimate/ct_estimate_patchid{}_idtheta{}_idphi{}.npy'\
        .format(patch_id,idtheta,idphi),projected_CT)

    
## if time permits, try double optimisations, but remember that daemonic processes
## cannnot have children
## that was called calculate_CT_parallel_outer, and hence the moniker inner above

def calculate_E(params): #tentative, check use of sample_points
    idtheta, idphi, patch_id = params[0],params[1],params[2]    
    sample_points = im_patches[:,:,patch_id]
    projected_CT = np.load('ct_estimate/ct_estimate_patchid{}_idtheta{}_idphi{}.npy'\
        .format(patch_id,idtheta,idphi))
    v = calculate_v(sample_points)
    E_T = (np.linalg.norm(np.dot(projected_CT,v)))**2
    return (patch_id,idtheta,idphi,E_T)