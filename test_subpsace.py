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


fig = plt.figure('Depth figure')
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
im = ax.imshow(p, cmap="jet", origin="lower")
fig.colorbar(im)
ax.set_title('Actual image depth values')
plt.savefig('figs/artificial_depth.png',bbox_inches='tight')


# In[5]:


get_ipython().magic('matplotlib inline')
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
im = ax.imshow(v_y, cmap="jet",origin="lower")
fig.colorbar(im)
ax.set_title('Image velocity values')


# # Calculating the estimated translation
# 
# The initial used algorithm is one of many, and is one of the best and most robust according to the review written by Raudies and Neumann.

# In[6]:


x_lim = size[0]
y_lim = size[1]
N = 10 #number of random points in each patch
f = 15 #focal length

num_patches = 0
im_patches = np.transpose([[np.random.randint(0,10,size=N),np.random.randint(0,1,size=N)]])
for x_split in np.arange(0,x_lim,np.int(x_lim/4)):
    for y_split in np.arange(0,y_lim,np.int(y_lim/4)):
        sample_points = np.transpose([np.random.randint(x_split,x_split+np.int(x_lim/4),size=N),np.random.randint(y_split,y_split+np.int(y_lim/4),size=N)])
        im_patches = np.dstack((im_patches,sample_points))
        num_patches += 1
im_patches = im_patches[:,:,1:]


# ### Storing beforehand the values of the $C(T)$ matrix in memory for each candidate translation direction and each image patch
# 
# These would need to be speedily accessed and used for different values of the speed as it begins to be available to us. 

# In[20]:


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
    sample_v_x, sample_v_y = v_x[sample_points[:,0],sample_points[:,1]], v_y[sample_points[:,0],sample_points[:,1]]
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
    np.save('ct_estimate/ct_estimate_patchid{}_idtheta{}_idphi{}.npy'.format(patch_id,idtheta,idphi),projected_CT)

    
## if time permits, try double optimisations, but remember that daemonic processes cannnot have children
## that was called calculate_CT_parallel_outer, and hence the moniker inner above

def calculate_E(params): #tentative, check use of sample_points, T or using patch_no, phi, theta
    idtheta, idphi, patch_id = params[0],params[1],params[2]    
    sample_points = im_patches[:,:,patch_id]
    projected_CT = np.load('ct_estimate/ct_estimate_patchid{}_idtheta{}_idphi{}.npy'.format(patch_id,idtheta,idphi))
    v = calculate_v(sample_points)
    E_T = (np.linalg.norm(np.dot(projected_CT,v)))**2
    return (patch_id,idtheta,idphi,E_T)
    


# In[31]:


get_ipython().run_cell_magic('time', '', "##this is to be ran only once at the beginning of the video\nCPU_NUMBER = os.cpu_count()\nif __name__ == '__main__':\n    search_range = np.arange(0,np.int(100*np.pi),1)\n    patch_id_range = range(16)\n    paramlist = list(itertools.product(search_range,search_range,patch_id_range))\n    pool = Pool(CPU_NUMBER)\n    pool.map(calculate_CT_parallel_inner,paramlist)")


# In[32]:


get_ipython().run_cell_magic('time', '', "\n## this is to be run as the video runs - repeatedly\nCPU_NUMBER = os.cpu_count()\nif __name__=='__main__':\n    search_range = np.arange(0,np.int(100*np.pi),1)\n    patch_id_range = range(16)\n    paramlist = list(itertools.product(search_range,search_range,patch_id_range))\n    pool = Pool(CPU_NUMBER)\n    res = pool.map(calculate_E,paramlist)")


# In[44]:


E = np.zeros((16,np.size(search_range),np.size(search_range)))
print(E[0,0,1])


# In[50]:


get_ipython().run_cell_magic('time', '', "E = np.zeros((16,np.size(search_range),np.size(search_range)))\nfor item in res:\n    E[item[0],item[1],item[2]] = item[3]\n    \nfor patch_id in range(16):\n    np.save('e_estimate/e_estimate_patchid{}.npy'.format(patch_id),E[patch_id])\n    ")


# In[51]:


E = np.zeros((np.size(search_range),np.size(search_range)))

for i in range(16):
    E_temp = np.load('e_estimate/e_estimate_patchid{}.npy'.format(i))
    E = E + E_temp


# In[53]:


np.load('e_estimate/e_estimate_patchid{}.npy'.format(2))
#plotting residual surfaces
get_ipython().magic('matplotlib inline')
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('theta')
ax.set_ylabel('phi')
im = ax.imshow(E, cmap="jet",origin = "lower")
fig.colorbar(im)
ax.set_title('Residual function values')
idtheta, idphi = np.unravel_index(E.argmin(), E.shape)
theta, phi = search_range[idtheta], search_range[idphi]
x_final = np.cos(theta)*np.sin(phi)
y_final = np.sin(theta)*np.sin(phi)
z_final = np.cos(phi)
T_final = np.array([[x_final],[y_final],[z_final]])
print(T_final)


# In[54]:


get_ipython().run_cell_magic('time', '', "## run through all candidate translation directions\nsearch_range =np.arange(0,np.pi,0.01)\nE = np.zeros((np.size(search_range),np.size(search_range)))\n\nfor idtheta,theta in enumerate(list(search_range)):\n    for idphi,phi in enumerate(list(search_range)):\n        x = np.cos(theta)*np.sin(phi)\n        y = np.sin(theta)*np.sin(phi)\n        z = np.cos(phi)\n        T = np.array([[x],[y],[z]])\n\n        CT = calculate_CT_estimate(sample_points, T)\n        v_t = np.vstack((sample_v_x,sample_v_y)).reshape((-1),order='F').reshape(1,2*N) \n        #not sure where I got (-1) from but hey it works #v is concat of all sample point vels \n        v = np.transpose(v_t)\n        CTbar, r = np.linalg.qr(CT)\n        I = np.identity(2*N)\n        cc = np.dot(CTbar,np.transpose(CTbar))\n        E_T = (np.linalg.norm(np.dot((I - cc),v)))**2\n        E[idtheta,idphi] += E_T\n\n        \nidtheta, idphi = np.unravel_index(E.argmin(), E.shape)\ntheta, phi = search_range[idtheta], search_range[idphi]\nx_final = np.cos(theta)*np.sin(phi)\ny_final = np.sin(theta)*np.sin(phi)\nz_final = np.cos(phi)\nT_final = np.array([[x_final],[y_final],[z_final]])\nprint(T_final)")


# In[ ]:


get_ipython().run_cell_magic('time', '', '#estimating rotation given the translation\n\nN = np.shape(sample_points)[0]; #justincase\nf = 15\nsum_left = np.zeros([3,3])\nsum_right = np.zeros([3,1])\n\nfor time in np.arange(20,21,1):\n    for i in np.arange(0,N,1):\n        x,y = sample_points[i,0],sample_points[i,1]\n\n        #calculating d\n        A = np.array([[-f,0,x],[0,-f,y]])\n        AtimesT = np.dot(A,T)\n        d = np.array([[AtimesT[1,0]],[-AtimesT[0,0]]])\n        di = d/linalg.norm(d)\n\n        #calculating left term in omega estimate\n        Bi = np.array([[(x*y)/f, -(f + (x*x)/f), y],[f + (y*y)/f, -(x*y)/f, -x]])\n        sum_left += np.dot(np.transpose(Bi),np.dot(di,np.dot(np.transpose(di),Bi)))\n        \n        #calculating right term in omega estimate\n        sample_v_x,sample_v_y = v_x[x,y],v_y[x,y]\n        vi = np.array([[sample_v_x],[sample_v_y]])\n        sum_right += np.dot(np.transpose(Bi),np.dot(di,np.dot(np.transpose(di),vi)))\n\nomega = np.dot(linalg.inv(sum_left),sum_right)\nprint(omega)')


# In[ ]:


#calculating depth
depth_mat0 = np.zeros([x_lim,y_lim])
depth_mat1 = np.zeros([x_lim,y_lim])
for x in np.arange(0,x_lim,1):
    for y in np.arange(0,y_lim,1):
        A = np.array([[-f,0,x],[0,-f,y]])
        B = np.array([[(x*y)/f, -(f + (x*x)/f), y],[f + (y*y)/f, -(x*y)/f, -x]])
        v = np.array([[v_x[x,y]],[v_y[x,y]]])
        scaled_AT = v - np.dot(B,omega)
        act_AT = np.dot(A,T)
        depth0 = (act_AT/scaled_AT)[0]
        depth1 = (act_AT/scaled_AT)[1]
        depth_mat0[x,y] = depth0
        depth_mat1[x,y] = depth1


# In[ ]:


#plotting depth
get_ipython().magic('matplotlib inline')
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('x')
ax.set_ylabel('y')
im = ax.imshow(depth_mat1, cmap="jet",origin = "lower")
fig.colorbar(im)
ax.set_title('Calculated image depth values')
plt.savefig('figs/calculated_depth.png', bbox_inches='tight')


# In[ ]:




