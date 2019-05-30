import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from photutils import find_peaks
from astropy.stats import sigma_clipped_stats
from astropy import wcs, coordinates

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

    def residuals(self, params, x, y, err, maxv):
        dat = self.multivariate_gaussian_2d(params)
        index, = np.where(y>=0.2*maxv)

        return (y[index]-dat[index]) / err[index]

    def peak_finder(self, map_data, mask_pf = False):

        bs = 5

        mean, median, std = sigma_clipped_stats(self.data, sigma=3.0)
        threshold = median+(5.*std)
        if mask_pf is False:
            tbl = find_peaks(map_data, threshold, box_size=bs)
        else:
            self.mask = mask_pf.copy()
            tbl = find_peaks(map_data, threshold, box_size=bs, mask = self.mask)
        tbl['peak_value'].info.format = '%.8g'

        guess = np.array([])

        x_i = np.array([])
        y_i = np.array([])

        for i in range(len(tbl['peak_value'])):
            guess_temp = np.array([tbl['peak_value'][i], self.xgrid[tbl['x_peak'][i]], \
                                  self.ygrid[tbl['y_peak'][i]], 1., 1., 0.])
            guess = np.append(guess, guess_temp)
            index_x = self.xgrid[tbl['x_peak'][i]]
            index_y = self.ygrid[tbl['y_peak'][i]]
            x_i = np.append(x_i, index_x)
            y_i = np.append(y_i, index_y)
            mask_pf[index_y-bs:index_y+bs, index_x-bs:index_x+bs] = True

            if np.size(self.param) == 0:
                self.param = guess_temp
                self.mask = mask_pf 

            else:
                self.param = np.append(self.param, guess_temp)
                self.mask = np.logical_or(self.mask, mask_pf)

    def fit(self):

        p = least_squares(self.residuals, x0=self.param, \
                          args=(self.xy_mesh, np.ravel(self.data),\
                                np.ones(len(np.ravel(self.data))), np.amax(self.data)), \
                          method='lm')
        J = p.jac
        cov = np.linalg.inv(J.T.dot(J))
        var = np.sqrt(np.diagonal(cov))

        return p, var

    def beam_fit(self, peak_number = 0, mask_pf= False):

        if peak_number == 0:
            self.peak_finder(map_data = self.data, mask_pf = mask_pf)
            peak_number_ini = np.size(self.param)/6
            peak_found = peak_number_ini
        else:
            self.param  = 0

        while peak_found > 0:

            fit_param, var = self.fit()

            fit_data = self.multivariate_gaussian_2d(fit_param.x)\
                            .reshape(np.outer(self.xgrid, self.ygrid).shape)
            res = self.data-fit_data

            self.peak_finder(map_data=res)

            peak_number = np.size(self.param)/6

            peak_found = peak_number-peak_number_ini

            peak_number_ini = peak_number

        return fit_data, fit_param.x, var

class computeoffset():

    def __init__(self, data, angX_center, angY_center, ctype):

        self.data = data
        self.angX_center = angX_center #coord1 center map 
        self.angY_center = angY_center #coord2 center map
        self.ctype = ctype

    def centroid(self, threshold=0.275):

        maxval = np.max(self.data)
        minval = np.min(self.data)
        y_max, x_max = np.where(self.data == maxval)

        lt_inds = np.where(self.data < threshold*maxval)
        gt_inds = np.where(self.data > threshold*maxval)

        weight = np.zeros((self.data.shape[0], self.data.shape[1]))
        weight[gt_inds] = 1.
        a = self.data[gt_inds]
        flux = np.sum(a)
        x_range = np.arange(0, self.data.shape[0])
        y_range = np.arange(0, self.data.shape[1])

        yy, xx = np.meshgrid(y_range, x_range)

        x_c = np.sum(xx*weight*self.data)/flux
        y_c = np.sum(yy*weight*self.data)/flux

        return np.rint(x_c), np.rint(y_c)

    def offset(self, wcs_trans, threshold=0.275, return_pixel=False, altitude=0., lon=0., lat=0.):

        x_c, y_c = self.centroid(threshold=threshold)

        coord_centre = coordinates.SkyCoord(self.angX_center, self.angY_center, unit='deg')

        if return_pixel is True:
        
            x_map, y_map = wcs.utils.skycoord_to_pixel(coord_centre, wcs_trans)

            x_off = x_map-x_c
            y_off = y_map-y_c

            return x_off, y_off
        
        else:
            print
            coord = wcs.utils.pixel_to_skycoord(x_c, y_c, wcs_trans)
                        
            offset_angle = coord_centre.spherical_offsets_to(coord)

            if self.ctype == 'AZ and EL':

                return offset_angle.az, offset_angle.alt

            elif self.ctype == 'RA and DEC':

                return offset_angle.ra, offset_angle.dec

            elif self.ctype == 'CROSS-EL and EL':

                return offset_angle.az*np.cos(offset_angle.alt), offset_angle.alt




