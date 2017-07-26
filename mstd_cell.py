import numpy as np


class MSTdCell(object):
    """
        Implement the MSTd cell model from the article: 
        
        Layton, O. W., & Fajen, B. R. (2016). Competitive dynamics in MSTd: 
        A mechanism for robust heading perception based on optic flow. *PLoS Comput Biol, 12(6), e1004942*.
        
        The computation of equation 20 is performed, but not the competitive dynamics which follows. 
        What it does is finding out which radial template of optic flow is closest to the combined
        response of the MT cells computed before.
    """
    
    def __init__(self, im_shape, centre_x, centre_y, spacing, deactivation = 0.0, lambd_ = 0.001,
                 thresh_ = 0.0001):
        """
            Arguments of the class:
                - im_shape: size of the stimulus which is passed.
                - v_x: x-component of the optic flow field.
                - v_y: y-component of the optic_flow field.
                - spacing: spacing between the centres of two radial optic flow patterns.
                - deactivation: rate of deactivation of the cell, in order to simulate ageing.
            The class generates radial optic flow templates on a grid larger than the image, and each template
            center (x0, y0) is at a distance spacing from the other.
        """
        
        self.deactivation_rate = deactivation
        self.lambd_ = lambd_
        self.thresh = thresh_
                       
        self.init_again(im_shape, centre_x, centre_y, spacing)
        
    def init_again(self, im_shape, centre_x, centre_y, spacing):
        """
            Performs the initialisation.
        """
        self.shape = np.array(im_shape)

        self.spacing = spacing

        self.x_grid = np.arange(centre_x%self.spacing, self.shape[0], self.spacing)
        self.y_grid = np.arange(centre_y%self.spacing, self.shape[1], self.spacing)
        
        self.deactivation_map = np.random.choice([0, 1], size=(len(self.x_grid), len(self.y_grid)), 
                                                p=[self.deactivation_rate, 1-self.deactivation_rate])        
    def compute_correlation(self, v_x, v_y):
        """
            Computes the correlation of the radial optic flow patterns with the velocity field (v_x, v_y).
            The radial flow templates are centred at the middle of the image, and separated by a distance
            `spacing` from each other. 
            The response then takes the shape (T, I, J), where T is the number of time steps, I is the
            horizontal number of templates and J the vertical one.
        """
        # self.centre = np.floor(self.large_shape/2).astype(int)
        # kind of subsampling, we only make the correlation of the templates centred
        # at these particular locations (i.e. if we know that the template should be centred at the pixels
        # 151, 75, then we'll pass that as an argument, and it will guarantee that, given the spacing that was
        # specified in the class, we get this particular pattern).,

        if (np.array(v_x.shape[1:]) != self.shape).any():
            raise ValueError('Invalid size of optic flow field for the current instance of the class.')
        
        self.v_x = v_x
        self.v_y = v_y
        
        len_t = self.v_y.shape[0]
        
        self.resp = np.zeros((len_t, len(self.x_grid), len(self.y_grid)))
        self.pattern_map = np.zeros((len(self.x_grid), len(self.y_grid),2))
        
        for i, x0 in enumerate(self.x_grid):
            for j, y0 in enumerate(self.y_grid):
                pattern_x, pattern_y = self.get_pattern(x0, y0)
                self.pattern_map[i, j] = [x0, y0]
                
                for t in range(0, len_t):    
                    #Complicated formula to perform the multiplication of the pattern with the optic flow
                    #only at the places inside the original image, which are non zero.
                    #Note that both v_x and pattern_x are normalised, so the scalar product will be on the same
                    #scale at every location.
                    self.resp[t, i,  j] = ((self.v_x[t]* pattern_x).sum() + (self.v_y[t]*pattern_y).sum())
                    

        
        self.resp = self.deactivation_map*self.resp
        return self.resp
    
    def get_pattern(self, x0, y0):
        """
           Returns the radial flow template centred at (x0, y0), with an expontential decay from the
           FoE to simulate a more biological receptive field for the cell. All the vectors with a 
           magnitude smaller than a threshold are set to zero, and the response is normalised by
           the number of non-zero vectors in order for the MSTd cells at the boundary of the visual
           field to be able to compete with the ones inside it.
        """    
        pattern_x, pattern_y = np.meshgrid(np.arange(0, self.shape[0]), np.arange(0, self.shape[1]))
        pattern_x, pattern_y = pattern_x.T, pattern_y.T
    
        pattern_y_base =  (self.shape[1]-y0-pattern_y).astype(float)
        pattern_x_base =  (pattern_x - x0).astype(float)
        
        pattern_x, pattern_y = np.array(pattern_x_base), np.array(pattern_y_base)
        
        if self.lambd_ >= 0:
            
            pattern_norm = np.sqrt(pattern_x**2 + pattern_y **2)
            pattern_x[pattern_norm !=0] /= pattern_norm[pattern_norm!=0]
            pattern_y[pattern_norm !=0] /= pattern_norm[pattern_norm!=0]
            
             #print(pattern_x_base.max(), pattern_x_base.min())
            
            pattern_x, pattern_y = (pattern_x*np.exp(-self.lambd_*(pattern_y_base**2/2+pattern_x_base**2/2)), 
                               pattern_y*np.exp(-self.lambd_*(pattern_y_base**2/2+pattern_x_base**2/2)))
    
            pattern_norm = np.sqrt(pattern_x**2 + pattern_y **2)
        
            pattern_x[pattern_norm < self.thresh] = 0
            pattern_y[pattern_norm < self.thresh] = 0
            pattern_x /= (pattern_norm > self.thresh).sum()    
            pattern_y /= (pattern_norm > self.thresh).sum()
        elif self.lambd_ < 0:
            raise ValueError('Error, lambda cannot take a negative value.')
   
        return pattern_x, pattern_y
    
    def get_pattern_full(self, x0, y0):
        """
            Returns the radial flow template centred at (x0, y0), with each arrow normalised.
        """
        
        pattern_x, pattern_y = np.meshgrid(np.arange(0, self.shape[0]), np.arange(0, self.shape[1]))
        pattern_x, pattern_y = pattern_x.T, pattern_y.T

        pattern_y =  (y0-pattern_y).astype(float)
        pattern_x =  (pattern_x - x0).astype(float)
        
        pattern_norm = np.sqrt(pattern_x**2 + pattern_y **2)
        
        pattern_x[pattern_x != 0] =  pattern_x[pattern_x != 0]/pattern_norm[pattern_x != 0] 
        pattern_y[pattern_y != 0] = pattern_y[pattern_y != 0]/pattern_norm[pattern_y != 0]

        return pattern_x, pattern_y

    def get_max_resp(self, t):
        """
            At a time t, returns the position in pixels of the centre of the radial flow template
            with the largest response.
        """
        return self.pattern_map[np.unravel_index(np.argmax(self.resp[t]),self.resp[t].shape)]
    
    def get_heading_x(self):
        len_t = 0
        heading_x = 0;
        for t in range(self.v_y.shape[0]):
            len_t +=1
            heading_x += self.get_max_resp(t)[0]
        return heading_x/len_t