import numpy as np
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
import matplotlib.pyplot as plt
=======
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> 8989c24... Correct calculation of coordinates
=======
from scipy.linalg import svd
>>>>>>> caf44ca... Solved some bugs
from scipy.optimize import least_squares
from photutils import find_peaks
from astropy.stats import sigma_clipped_stats
from astropy import wcs, coordinates

#import matplotlib.pyplot as plt


class beam(object):

    def __init__(self, data, param = None):

        self.data = data
        self.param = param

        self.xgrid = np.arange(len(self.data[0,:]))
        self.ygrid = np.arange(len(self.data[:,0]))
        self.xy_mesh = np.meshgrid(self.xgrid,self.ygrid)

<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> 4ee3dbe... Fixed bug in selecting data
    def multivariate_gaussian_2d(self, params):

        (x, y) = self.xy_mesh
        for i in range(int(np.size(params)/6)):
            j = i*6
            amp = params[j]
            xo = float(params[j+1])
            yo = float(params[j+2])
            sigma_x = params[j+3]
            sigma_y = params[j+4]
<<<<<<< HEAD
<<<<<<< HEAD
            theta = params[j+5]    
=======
            theta = params[j+5]   
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
            theta = params[j+5]   
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)
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

        x_lim = np.size(self.xgrid)
        y_lim = np.size(self.ygrid)
        fact = 20.

        bs = np.array([int(np.floor(y_lim/fact)), int(np.floor(x_lim/fact))])

        mean, median, std = sigma_clipped_stats(self.data, sigma=3.0)
        threshold = median+(5.*std)
<<<<<<< HEAD
<<<<<<< HEAD
        if mask_pf == False:
=======
        if mask_pf is False:
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
        if mask_pf is False:
>>>>>>> 4ee3dbe... Fixed bug in selecting data
            tbl = find_peaks(map_data, threshold, box_size=bs)
            mask_pf = np.zeros_like(self.xy_mesh[0])
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
            mask_pf[index_y-bs[1]:index_y+bs[1], index_x-bs[0]:index_x+bs[0]] = True

            if self.param is None:
                self.param = guess_temp
                self.mask = mask_pf.copy()

            else:
                self.param = np.append(self.param, guess_temp)
                self.mask = np.logical_or(self.mask, mask_pf)

    def fit(self):
        try:
            print('PARAM', self.param)
            p = least_squares(self.residuals, x0=self.param, \
                              args=(self.xy_mesh, np.ravel(self.data),\
                                    np.ones(len(np.ravel(self.data))), np.amax(self.data)), \
                              method='lm')
            
            # J = p.jac
            # cov = np.linalg.inv(J.T.dot(J))
            # var = np.sqrt(np.diagonal(cov))

            _, s, VT = svd(p.jac, full_matrices=False)
            threshold = np.finfo(float).eps * max(p.jac.shape) * s[0]
            s = s[s > threshold]
            VT = VT[:s.size]
            var = np.dot(VT.T / s**2, VT)
            print('VAR', np.sqrt(np.diag(var)))
            return p, var
        except np.linalg.LinAlgError:
            msg = 'Fit not converged'
            return msg, 0
        # except ValueError:
        #     msg = 'Too Many '

    def beam_fit(self, mask_pf= False):

        if self.param is not None:
            peak_found = np.size(self.param)/6
            force_fit = True
        else:
            self.peak_finder(map_data = self.data, mask_pf = mask_pf)
            peak_number_ini = np.size(self.param)/6
            peak_found = peak_number_ini
            force_fit = False

        while peak_found > 0:
            fit_param, var = self.fit()
            if isinstance(fit_param, str):
                msg = 'fit not converged'
                break
            else:
                fit_data = self.multivariate_gaussian_2d(fit_param.x).reshape(np.outer(self.ygrid, self.xgrid).shape)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
            fit_data = self.multivariate_gaussian_2d(fit_param.x)\
                            .reshape(np.outer(self.xgrid, self.ygrid).shape)
=======
            fit_data = self.multivariate_gaussian_2d(fit_param.x).reshape(np.outer(self.xgrid, self.ygrid).shape)
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
            fit_data = self.multivariate_gaussian_2d(fit_param.x).reshape(np.outer(self.xgrid, self.ygrid).shape)
>>>>>>> 8989c24... Correct calculation of coordinates
            res = self.data-fit_data

            self.peak_finder(map_data=res)

            peak_number = np.size(self.param)/6
=======
                if force_fit is False:
                    res = self.data-fit_data
>>>>>>> 1531050... Added option to choose beam fitting parameters

                    self.peak_finder(map_data=res)

                    peak_number = np.size(self.param)/6
                    peak_found = peak_number-peak_number_ini
                    peak_number_ini = peak_number
                else:
                    peak_found = -1

        if isinstance(fit_param, str):
            return msg, 0, 0
        else:
            print('PARAM_FIT', fit_param.x)
            return fit_data, fit_param.x, var
        

<<<<<<< HEAD
class computeoffset():

    def __init__(self, data, angX_center, angY_center, ctype):

        self.data = data
        self.angX_center = angX_center #coord1 center map 
        self.angY_center = angY_center #coord2 center map
        self.ctype = ctype

    def centroid(self, threshold=0.275):

<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> 8989c24... Correct calculation of coordinates
        '''
        For more information about centroid calculation see Shariff, PhD Thesis, 2016
        '''

<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> 8989c24... Correct calculation of coordinates
        maxval = np.max(self.data)
        minval = np.min(self.data)
        y_max, x_max = np.where(self.data == maxval)

        lt_inds = np.where(self.data < threshold*maxval)
        gt_inds = np.where(self.data > threshold*maxval)

<<<<<<< HEAD
<<<<<<< HEAD
        weight = np.zeros((self.data.shape[1], self.data.shape[0]))
=======
        weight = np.zeros((self.data.shape[0], self.data.shape[1]))
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
        weight = np.zeros((self.data.shape[0], self.data.shape[1]))
>>>>>>> 4ee3dbe... Fixed bug in selecting data
        weight[gt_inds] = 1.
        a = self.data[gt_inds]
        flux = np.sum(a)
        x_range = np.arange(0, self.data.shape[0])
        y_range = np.arange(0, self.data.shape[1])

<<<<<<< HEAD
<<<<<<< HEAD
        xx, yy = np.meshgrid(x_range, y_range)
=======
        yy, xx = np.meshgrid(y_range, x_range)
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
        yy, xx = np.meshgrid(y_range, x_range)
>>>>>>> 4ee3dbe... Fixed bug in selecting data

        x_c = np.sum(xx*weight*self.data)/flux
        y_c = np.sum(yy*weight*self.data)/flux

        return np.rint(x_c), np.rint(y_c)

    def offset(self, wcs_trans, threshold=0.275, return_pixel=False, altitude=0., lon=0., lat=0.):
<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> ccf3a8f... Added output for offset calculation
        x_c, y_c = self.centroid(threshold=threshold)

        coord_centre = coordinates.SkyCoord(self.angX_center, self.angY_center, unit='deg')

<<<<<<< HEAD
<<<<<<< HEAD
        if return_pixel == True:
=======
        if return_pixel is True:
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
        if return_pixel is True:
>>>>>>> 4ee3dbe... Fixed bug in selecting data
        
            x_map, y_map = wcs.utils.skycoord_to_pixel(coord_centre, wcs_trans)

            x_off = x_map-x_c
            y_off = y_map-y_c

            return x_off, y_off
        
        else:
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
            print
>>>>>>> 4ee3dbe... Fixed bug in selecting data
=======
>>>>>>> ccf3a8f... Added output for offset calculation
            coord = wcs.utils.pixel_to_skycoord(x_c, y_c, wcs_trans)
                        
            offset_angle = coord_centre.spherical_offsets_to(coord)

            return offset_angle[0].degree, offset_angle[1].degree

            # if self.ctype == 'AZ and EL':
=======
>>>>>>> 8989c24... Correct calculation of coordinates

            if self.ctype == 'RA and DEC':
                coord = wcs.utils.pixel_to_skycoord(x_c, y_c, wcs_trans)
                offset_angle = coord_centre.spherical_offsets_to(coord)

                return offset_angle[0].degree, offset_angle[1].degree

            elif self.ctype == 'CROSS-EL and EL':
                coord = wcs_trans.wcs_pix2world(x_c, y_c, 1.)

<<<<<<< HEAD
<<<<<<< HEAD
                return offset_angle.az*np.cos(offset_angle.alt), offset_angle.alt
=======

            if self.ctype == 'RA and DEC':
                coord = wcs.utils.pixel_to_skycoord(x_c, y_c, wcs_trans)
                offset_angle = coord_centre.spherical_offsets_to(coord)

                return offset_angle[0].degree, offset_angle[1].degree

            elif self.ctype == 'CROSS-EL and EL':
                coord = wcs_trans.wcs_pix2world(x_c, y_c, 1.)

                return coord[0]-self.angX_center, coord[1]-self.angY_center
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
            #     return offset_angle.az*np.cos(offset_angle.alt), offset_angle.alt
>>>>>>> ccf3a8f... Added output for offset calculation
=======
                return coord[0]-self.angX_center, coord[1]-self.angY_center
>>>>>>> 8989c24... Correct calculation of coordinates
=======

>>>>>>> caf44ca... Solved some bugs




