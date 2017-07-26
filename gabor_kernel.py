import numpy as np
from scipy.optimize import brentq
from helpers import *


#######################################################################################
### Resolution Class: Gives the Resolution we will use for the implementation of the filter
### Links values in degrees or seconds to values in pixels or steps
#######################################################################################

class ResolutionInfo(object):
    def __init__(self, pixel_per_degree=10.0, step_per_second=1000.0, input_luminosity_range=1.0):
        self.pixel_per_degree = pixel_per_degree
        self.step_per_second = step_per_second
    def degree_to_pixel(self,degree):
        if self.pixel_per_degree is None:
            return default_resolution.degree_to_pixel(degree)
        return float(degree) * self.pixel_per_degree
    def pixel_to_degree(self,pixel):
        if self.pixel_per_degree is None:
            return default_resolution.pixel_to_degree(pixel)
        return float(pixel) / self.pixel_per_degree
    def second_to_step(self,t):
        if self.step_per_second is None:
            return default_resolution.second_to_step(t)
        return float(t) * self.step_per_second
    def step_to_second(self,step):
        if self.step_per_second is None:
            return default_resolution.step_to_second(step)
        return float(step) / self.step_per_second
    
    
#######################################################################################
### 2D Gabor Filter, spatial dimensions only
### The parameters are:
###    - *size* : size of the filter (array with the x-size, y-size)
###    - *x_sig, y_sig*: standard deviations in the x,y dimensions
###    - *theta, k, phi*: respectively $\theta$ is the preferred orientation of the Gabor Filter,
###      $k$ the preferred spatial frequency, and $\phi$ is the phase shift.
###    - *resolution, even*: an instance of the class *ResolutionInfo*, containing all the 
###      information relating the pixel to degrees.
#######################################################################################
def gabor_2d_px(size, x_sig, y_sig, k, theta, phi):
    """
        A 2D Gabor Filter, with the units given in pixels only.

        size corresponds to the size [deg x deg] that we want our filter to have. 
        N.B. It will always get to an odd sized filter (e.g. 200x200 will yield a filter of size 201x201).

        x_sig and y_sig are the standard deviations in x and y direction [px].
        
        theta [rad] is the preferred orientation of the Gabor Filter, k the preferred spatial frequency [1/px], 
        and phi [rad] is the phase shift.
    """
    theta = normalise_angle(theta)
    phi = normalise_angle(phi)
    
    if x_sig == 0 or y_sig == 0:
        return np.ones((1,1))
    
    #floor so we always keep an odd size for the filter

    x_min = np.floor(size[0]/2)
    y_min = np.floor(size[1]/2)
    
    #Generating the meshgrid
    (x, y) = np.meshgrid(np.arange(-x_min,x_min+1),
                         np.arange(-y_min,y_min+1) )
    #print(x.shape)
    if x_min < 1.0 or y_min < 1.0:
        kernel = np.ones(2) if even else np.ones(1)
    else:
        # Rotation 
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = - x * np.sin(theta) + y * np.cos(theta)

        kernel = np.exp(-.5 * (x_theta ** 2 / x_sig ** 2 + y_theta ** 2 / y_sig ** 2)) * np.cos(k * x_theta - phi)
        kernel = kernel /((2*np.pi) * x_sig * y_sig)
        
    if np.any(np.array(kernel.shape) == 0):
        return np.ones((1,1))
    return x,y,kernel


def gabor_2d(size, x_sig, y_sig, k, theta, phi, resolution=None):
    """
        A 2D Gabor Filter.
        
        size corresponds to the size [deg x deg] that we want our filter to have. 
        N.B. It will always get to an odd sized filter.
        
        x_sig and y_sig are the standard deviations in x and y direction [deg].
        
        theta [rad] is the preferred orientation of the Gabor Filter, k the preferred spatial frequency [1/deg], 
        and phi [rad] is the phase shift.
    """    
    if resolution is None:
        return np.ones((1,1))
    size = [resolution.degree_to_pixel(size_elem) for size_elem in np.array(size)]
    x_sig = resolution.degree_to_pixel(x_sig)
    y_sig = resolution.degree_to_pixel(y_sig)
    k = 1/resolution.degree_to_pixel(1/k)
    
    return gabor_2d_px(size, x_sig, y_sig, k, theta, phi)


#######################################################################################
### 1D in space, 1D in time Gabor Filter
### Implements the multiplication of a 1D spatial Gabor filter with a temporal kernel,
### and the rotation intricating $x$ and $\tau$ in order to make it non separable.
### Should be very similar to the Gabor 2D previously programmed.
###
### The parameters are:
###    - *x_sig, alpha*: standard deviations in the x dimension and temporal range of the filter
###    - *k, c, phi, psi*:  $k$ the preferred spatial frequency, $c$ the relation between space and time,
###       in units of [deg/second] or [px/step], $\phi$ is the phase shift and $psi$ is the temporal shifting angle.
###    - *resolution*: an instance of the class *ResolutionInfo*, containing all the 
###      information relating the pixel to degrees.
#######################################################################################

def gabor_1d_1d_px(size, x_sig, alpha, k, c, psi, phi):
    """
        Computes directly in pixels and steps the 1D spatial Gabor x 1D time kernel.
        It is called from the gabor_1d_1d function.
        
        size corresponds to the size [steps x px] that we want our filter to have. 
        N.B. It will always get to an odd sized filter.
    """
    
    size = np.array(size)
    
    phi = normalise_angle(phi)
    psi = normalise_angle(psi)
    
    if x_sig == 0 or alpha == 0:
        return np.ones((1,1))
    #Working first without rotation, determining where to stop depending on the desired precision

    t_min = int(np.floor(size[0]/2.0))
    x_min = int(np.floor(size[1]/2.0))
    D_t = make_D_t(alpha)
    
    #print('Range of filter: X=[',-x_min_psi,',',x_max_psi,']; T= [',-t_min_psi,',',t_max_psi,']')
    
    #Generating the meshgrid
    (X, T) = np.meshgrid(np.arange(-x_min,x_min+1), np.arange(-t_min,t_min+1))
    if x_min < 1.0 or t_min < 1.0:
        kernel = np.ones(1)
    else:
        # Rotation 
        X_psi = X * np.cos(psi) - T*c * np.sin(psi)
        T_psi = X/c * np.sin(psi) + T * np.cos(psi)
        
        #D_t = make_D_t(alpha)
        # Spatiotemporal kernel
        kernel = np.exp(-.5 * (X_psi ** 2 / x_sig ** 2 + T_psi**2 * alpha**2)) * np.cos(k * X_psi - phi)#*D_t(-T_psi)
        kernel = kernel * alpha/(2*np.pi * x_sig)
        #kernel[t_min:] = 0
    if np.any(np.array(kernel.shape) == 0):
        return np.ones((1,1))
    return X, T, kernel


def gabor_1d_1d(size, x_sig, alpha, k, c, psi, phi, resolution=None):
    """
        A 2D Gabor Filter.
        
        size corresponds to the size [sec x deg] that we want our filter to have. 
        N.B. It will always get to an odd sized filter.

        x_sig [deg] standard deviation, alpha [1/s] time parameter
        
        k [1/deg] is the preferred spatial frequency of the Gabor filter, and phi [rad] is the phase shift. 
        psi [rad] is the rotation between time and space, and c [deg/s] is the parameter linking
        "spatial steps" to "time steps".
    """
    if resolution is None:
        return np.ones((1, 1))
    
    # normalizing and fixing the parameters (degree, seconds to px, steps)
    size = np.array(size)
    size[0] = resolution.second_to_step(size[0])
    size[1] = resolution.degree_to_pixel(size[1])    
    c = resolution.degree_to_pixel(c) / resolution.second_to_step(1)
    k = 1 / resolution.degree_to_pixel(1 / k)
    alpha = 1 / resolution.second_to_step(1 / alpha)
    x_sig = resolution.degree_to_pixel(x_sig) 

    return gabor_1d_1d_px(size, x_sig, alpha, k, c, psi, phi)

#######################################################################################
### 3D (Spatiotemporal) Gabor Filter
### Implements the combination of both previous filters. 
###
### The parameters are:
###     - x_sig, y_sig are the standard deviations in x and y direction,
###       alpha is the time parameter.
###     - theta is the preferred orientation of the Gabor Filter, psi [rad] is the 
###       rotation between time and space, and phi [rad] is the phase shift.
###       x_sig standard deviation, alpha time parameter
###     - k is the preferred spatial frequency of the Gabor filter 
###       and c is the parameter linking "spatial steps" to "time steps".
#######################################################################################

def gabor_3d_px(size, x_sig, y_sig, alpha, k, c, theta, psi, phi, causal_time = False):
    """
        Returns a 3D spatiotemporal Gabor Filter, combining the previous implementations into 
        one filter to rule them all.
        - size corresponds to the size [steps x px x px] that we want our filter to have. 
          N.B. It will always get to an odd sized filter.
        - x_sig, y_sig [px] are the standard deviations in x and y direction,
          alpha [1/step] is the time parameter.
        - theta [rad] is the preferred orientation of the Gabor Filter, psi [rad] is the 
          rotation between time and space, and phi [rad] is the phase shift.
                x_sig standard deviation, alpha time parameter
        - k [1/px] is the preferred spatial frequency of the Gabor filter 
          and c [px/step] is the parameter linking "spatial steps" to "time steps".
        
    """
    
    if x_sig == 0 or y_sig == 0:
        return np.ones((1,1,1))
    
    size = np.array(size)
    
    # Angles in ]-pi,pi]
    theta = normalise_angle(theta)
    psi = normalise_angle(psi)
    phi = normalise_angle(phi)
    
    
    t_min = int(np.floor(size[0]/2))
    x_min = int(np.floor(size[1]/2))
    y_min = int(np.floor(size[2]/2))

    D_t = make_D_t(alpha)
    
    #print('Range of filter: T=[',-t_min_psi,t_max_psi,']; X=['
    #      ,-x_min_psi,x_max_psi,']; Y=[',-y_min_theta,y_min_theta,']')
    
    (X, T, Y) = np.meshgrid(np.arange(-x_min,x_min+1), np.arange(-t_min,t_min+1), np.arange(-y_min,y_min+1))
    if np.any(np.array(X.shape) == 0):
        return np.ones((1,1,1))
    
    if x_min < 1.0 or t_min < 1.0:
        kernel = np.ones(1)
    else:
        # Rotation of the coordinates
        X_psi = (X * np.cos(theta) + Y * np.sin(theta)) * np.cos(psi) - T*c * np.sin(psi)
        Y_theta = - X * np.sin(theta) + Y * np.cos(theta)
        T_psi = (X * np.cos(theta) + Y * np.sin(theta))/c * np.sin(psi) + T * np.cos(psi)

        # Spatiotemporal kernel
        
               
        #
        if causal_time:
            kernel = (np.exp(-.5 * (X_psi ** 2 / x_sig ** 2 + Y_theta**2 /y_sig**2)) 
                          * np.exp(-alpha*np.abs(T_psi)))#* np.cos(k * X_psi - phi)
            kernel[T < 0] = 0 # Setting the causality explicitely 
            kernel /= np.sum(kernel) # Normalising the kernel numerically -> Gaussian yield positive entries
            kernel = kernel * np.cos(k * X_psi - phi)
           
        else:
            kernel = np.exp(-.5 * (X_psi ** 2 / x_sig ** 2 + Y_theta**2 /y_sig**2 
                                   + alpha**2 * T_psi**2)) * np.cos(k * X_psi - phi)
            
            kernel = kernel * alpha/((2*np.pi)**1.5 * x_sig * y_sig)
        
    if np.any(np.array(kernel.shape) == 0):
        return np.ones((1,1,1))
    
    return T, X, Y, kernel


def gabor_3d(size, x_sig, y_sig, alpha, k, c, theta, psi, phi, resolution=None, causal_time = False):
    """
        Returns a 3D spatiotemporal Gabor Filter, combining the previous implementations into 
        one filter to rule them all.
        - size corresponds to the size [sec x deg x deg] that we want our filter to have. 
          N.B. It will always get to an odd sized filter.        
        - x_sig, y_sig [deg] are the standard deviations in x and y direction,
          alpha [1/s] is the time parameter.
        - theta [rad] is the preferred orientation of the Gabor Filter, psi [rad] is the 
          rotation between time and space, and phi [rad] is the phase shift.
                x_sig standard deviation, alpha time parameter
        - k [1/deg] is the preferred spatial frequency of the Gabor filter 
          and c [deg/s] is the parameter linking "spatial steps" to "time steps".
    """
    if resolution is None:
        return np.ones((1,1))
    
    # values from degree and seconds to pixels/steps
    size = np.array(size)
    size[0] = resolution.second_to_step(size[0])
    size[1] = resolution.degree_to_pixel(size[1])
    size[2] = resolution.degree_to_pixel(size[2])
    x_sig = resolution.degree_to_pixel(x_sig) 
    y_sig = resolution.degree_to_pixel(y_sig)
    alpha = 1/resolution.second_to_step(1/alpha)
    k = 1/resolution.degree_to_pixel(1/k)
    c = resolution.degree_to_pixel(c)/resolution.second_to_step(1)

    return gabor_3d_px(size, x_sig, y_sig, alpha, k, c, theta, psi, phi, causal_time)