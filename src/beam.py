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

        self.xgrid = np.arange(len(self.data[0,:]))
        self.ygrid = np.arange(len(self.data[:,0]))
        self.xy_mesh = np.meshgrid(self.xgrid,self.ygrid)


    def multivariate_gaussian_2d(self, params):

        (x, y) = self.xy_mesh
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

    def peak_finder(self, map_data, mask = False):

        bs = 5

        mean, median, std = sigma_clipped_stats(self.data, sigma=3.0)
        threshold = median+(5.*std)
        if mask == False:
            tbl = find_peaks(map_data, threshold, box_size=bs)
        else:
            tbl = find_peaks(map_data, threshold, box_size=bs, mask = self.mask)
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
                self.mask = np.logical_or(self.mask, mask)

    def fit(self):

        p = least_squares(self.residuals, x0=self.param, \
                          args=(self.xy_mesh, np.ravel(self.data),\
                                np.ones(len(np.ravel(self.data))), np.amax(self.data)), \
                          method='lm')
        J = p.jac
        cov = np.linalg.inv(J.T.dot(J))
        var = np.sqrt(np.diagonal(cov))

        return p, var

    def beam_fit(self, peak_number = 0):

        if peak_number == 0:
            self.peak_finder(map_data = self.data)
            peak_number_ini = np.size(self.param)/6
        else:
            self.param  = 0

        while peak_found > 0:

            fit_param, var = self.fit()

            fit_data = multivariate_gaussian_2d(self.xy_mesh, fit_param.x)\
                            .reshape(np.outer(self.xgrid, self.ygrid).shape)
            res = self.data-fit_data

            self.peak_finder(map_data=res)

            peak_number = np.size(self.param)/6

            peak_found = peak_number-peak_number_ini

            peak_number_ini = peak_number

        return fit_data, fit_param.x, var






