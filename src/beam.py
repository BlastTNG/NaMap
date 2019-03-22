from kapteyn import kmpfit
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from photutils import find_peaks
from astropy.stats import sigma_clipped_stats

class beam(object):

    def __init__(self, data):

        self.data = data
        self.param = np.array([])


    def multivariate_gaussian_2d(self, params):

        (x, y) = self.data
        for i in range(np.size(params)/6):
            j = i*6
            amp = params[j]
            xo = float(params[j+1])
            yo = float(params[j+2])
            sigma_x = params[j+3]
            sigma_y = params[j+4]
            theta = params[j+5]    
            a = (np.cos(theta)**2)/(2*sigma_x**2)+(np.sin(theta)**2)/(2*sigma_y**2)
            b = -(np.sin(2*theta))/(4*sigma_x**2)+(np.sin(2*theta))/(4*sigma_y**2)
            c = (np.sin(theta)**2)/(2*sigma_x**2)+(np.cos(theta)**2)/(2*sigma_y**2)
            if i == 0:
                multivariate_gaussian = amp*np.exp(-(a*((x-xo)**2)+2*b*(x-xo)*(y-yo)+c*((y-yo)**2)))
            else:
                multivariate_gaussian += amp*np.exp(-(a*((x-xo)**2)+2*b*(x-xo)*(y-yo)+c*((y-yo)**2)))
        
        return np.ravel(multivariate_gaussian)

    def residuals(params, x, y, err, maxv):
        dat = self.multivariate_gaussian_2d(x, params)
        index, = np.where(y>=0.2*maxv)

        return (y[index]-dat[index]) / err[index]

    def peak_finder(self):

        bs = 5

        mean, median, std = sigma_clipped_stats(self.data, sigma=3.0)
        threshold = median+(5.*std)
        tbl = find_peaks(self.data, threshold, box_size=bs)
        tbl['peak_value'].info.format = '%.8g'

        for i in range(len(tbl['peak_value'])):
            guess_temp = np.array([tbl['peak_value'][i], x[tbl['x_peak'][i]], \
                                  y[tbl['y_peak'][i]], 1., 1., 0.])
            guess = np.append(guess, guess_temp)
            index_x = x[tbl['x_peak'][i]]
            index_y = y[tbl['y_peak'][i]]
            x_i = np.append(x_i, index_x)
            y_i = np.append(y_i, index_y)
            mask[index_y-bs:index_y+bs, index_x-bs:index_x+bs] = True

        if np.size(self.param) == 0:
            self.param = guess_temp
            self.mask = mask 

        else:
            self.param = np.append(self.param, guess_temp)

    def beam_fit(self):

        p = least_squares(residuals_val, x0=self.param, \
                          args=(xy_mesh, np.ravel(self.data),\
                                np.ones(len(np.ravel(self.data))), np.amax(self.data)), \
                          method='lm')

        return p



