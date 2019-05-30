import numpy as np
import copy
from astropy import wcs, coordinates
from astropy.convolution import Gaussian2DKernel, convolve
import matplotlib.pyplot as plt


class maps():

    def __init__(self, ctype, crpix, cdelt, crval, data, coord1, coord2, convolution, std, Ionly=True):

        self.ctype = ctype
        self.crpix = crpix
        self.cdelt = cdelt
        self.crval = crval
        self.coord1 = coord1
        self.coord2 = coord2
        self.data = data
        self.w = 0.
        self.proj = 0.
        self.convolution = convolution
        self.std = std
        self.Ionly = Ionly

    def wcs_proj(self):
        wcsworld = wcs_world(self.ctype, self.crpix, self.cdelt, self.crval)

        self.w, self.proj = wcsworld.world(np.transpose(np.array([self.coord1, self.coord2])))
        # print('Coordinates')
        # print(np.transpose(np.array([self.coord1, self.coord2])))
        # print('XandY')
        # print(self.w)
        # print(np.amax(self.w[:,0]),np.amin(self.w[:,0]))

    def map2d(self):
        mapmaker = mapmaking(self.data, 1., 1.2, 1, np.floor(self.w).astype(int))
        if self.Ionly:
            Imap = mapmaker.map_singledetector_Ionly(self.crpix)

            if not self.convolution:
                return Imap
            else:
                return mapmaker.convolution(self.std, Imap)
        else:        
            Imap, Qmap, Umap = mapmaker.map_singledetector(self.crpix)
            if not self.convolution:
                return Imap, Qmap, Umap
            else:
                Imap_con = mapmaker.convolution(self.std, Imap)
                Qmap_con = mapmaker.convolution(self.std, Qmap)
                Umap_con = mapmaker.convolution(self.std, Umap)
                return Imap_con, Qmap_con, Umap_con

        # if np.size(param['file_repository']['detlist']) > 1:
        #     if param['map_parameters']['Ionly'].lower() == 'true':
        #         finalI = mapmaker.map_multidetector_Ionly()
        #     else:
        #         finalI = mapmaker.map_multidetector()

        # else:
        #     if param['map_parameters']['Ionly'][0].strip().lower() == 'true':
        #         finalI = mapmaker.map_singledetector_Ionly(param['map_parameters']['crpix'])
        #     else:
        #         finalI = mapmaker.map_singledetector()

        # if param['map_parameters']['conv'].lower() != 'na':
        #     finalI_conv = mapmaker.convolution(param['map_parameters']['stdev'], finalI)


class wcs_world():

    def __init__(self, ctype, crpix, crdelt, crval):

        self.ctype = ctype
        self.crdelt = crdelt
        self.crpix = crpix
        self.crval = crval

    def world(self, coord):
        
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = self.crpix
        w.wcs.cdelt = self.crdelt
        w.wcs.crval = self.crval
        if self.ctype == 'RA and DEC':
            w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        elif self.ctype == 'AZ and EL' or self.ctype == 'CROSS-EL and EL':
            w.wcs.ctype = ["TLON-ARC", "TLAT-ARC"]
        world = w.wcs_world2pix(coord, 1)

        return world, w

class mapmaking(object):

    def __init__(self, data, weight, polangle, number, pixelmap):

        self.data = data
        self.weight = weight
        self.polangle = polangle
        self.number = number
        self.pixelmap = pixelmap

    def map_param(self, crpix, value=None, sigma=None, angle=None):

        if value is None:
            value = self.data.copy()
            if np.size(self.weight) > 1:
                sigma = self.weight.copy()
            else:
                sigma = copy.copy(self.weight)
            if np.size(self.polangle) > 1:
                angle = self.polangle.copy()
            else:
                angle = copy.copy(self.polangle)*np.ones(np.size(value))

        '''
        sigma is the inverse of the sqared white noise value, so it is 1/n**2
        ''' 
        
        x_map = self.pixelmap[:,0]   #RA 
        y_map = self.pixelmap[:,1]   #DEC

        # print('Coordinates in X and Y')
        # print(x_map)
        # print(y_map)
        # print(np.amax(x_map),np.amin(x_map))
        # print(np.amin(y_map),np.amax(y_map))
        
        if (np.amin(x_map)) <= 0:
            x_map = np.floor(x_map+np.abs(np.amin(x_map)))
        else:
            x_map = np.floor(x_map-np.amin(x_map))
        if (np.amin(y_map)) <= 0:
            y_map = np.floor(y_map+np.abs(np.amin(y_map)))
        else:
            y_map = np.floor(y_map-np.amin(y_map))

        x_len = np.amax(x_map)-np.amin(x_map)+1
        param = x_map+y_map*x_len
        param = param.astype(int)

        flux = value

        cos = np.cos(2.*angle)
        sin = np.sin(2.*angle)

        I_est_flat = np.bincount(param, weights=flux)*sigma
        Q_est_flat = np.bincount(param, weights=flux*cos)*sigma
        U_est_flat = np.bincount(param, weights=flux*sin)*sigma

        N_hits_flat = 0.5*np.bincount(param)*sigma
        c_flat = np.bincount(param, weights=0.5*cos)*sigma
        c2_flat = np.bincount(param, weights=0.5*cos**2)*sigma
        s_flat = np.bincount(param, weights=0.5*sin)*sigma
        s2_flat = N_hits_flat-c2_flat
        m_flat = np.bincount(param, weights=0.5*cos*sin)*sigma
        Delta = (c_flat**2*(c2_flat-N_hits_flat)+2*s_flat*c_flat*m_flat-c2_flat*s_flat**2-\
                 N_hits_flat*(c2_flat**2+m_flat**2-c2_flat*N_hits_flat))
        A = -(c2_flat**2+m_flat**2-c2_flat*N_hits_flat)
        B = c_flat*(c2_flat-N_hits_flat)+s_flat*m_flat
        C = c_flat*m_flat-s_flat*c2_flat
        D = -((c2_flat-N_hits_flat)*N_hits_flat+s_flat**2)
        E = c_flat*s_flat-m_flat*N_hits_flat
        F = c2_flat*N_hits_flat-c_flat**2

        return I_est_flat, Q_est_flat, U_est_flat, N_hits_flat, Delta, A, B, C, D, E, F, param

    def map_singledetector_Ionly(self, crpix, value=None, sigma=None, angle=None):

        value =self.map_param(crpix=crpix, value=value, sigma=sigma, angle=angle)

        I_flat = np.zeros(len(value[0]))

        I_flat[np.nonzero(value[0])] = value[0][np.nonzero(value[0])]/value[3][np.nonzero(value[0])]

        x_len = np.amax(self.pixelmap[:,0])-np.amin(self.pixelmap[:,0])
        y_len = np.amax(self.pixelmap[:,1])-np.amin(self.pixelmap[:,1])

        if len(I_flat) < (x_len+1)*(y_len+1):
            valmax = (x_len+1)*(y_len+1)
            pmax = np.amax(value[-1])
            I_fin = 0.*np.arange(pmax+1, valmax)
            
            I_flat = np.append(I_flat, I_fin)

        I_pixel = np.reshape(I_flat, (y_len+1,x_len+1))
    
        return I_pixel

    def map_multidetectors_Ionly(self):

        for i in range(self.number):
            mapvalues = self.map_singledetector_Ionly(value=self.data[i],sigma=self.weight[i],\
                                                      angle=self.polangle[i])
            if i == 0:
                I_pixel = mapvalues[0].copy()
            else:
                I_pixel += mapvalues[0]

        return I_pixel 

    def map_singledetector(self, crpix, value=None, sigma=None, angle=None):


        (I_est_flat, Q_est_flat, U_est_flat, N_hits_flat, Delta, \
         A, B, C, D, E, F, param) = self.map_param(crpix=crpix, value=value, sigma=sigma,angle=angle)

        I_pixel_flat = np.zeros(len(I_est_flat))
        Q_pixel_flat = np.zeros(len(Q_est_flat))
        U_pixel_flat = np.zeros(len(U_est_flat))

        index, = np.where(np.abs(Delta)>0.)
        
        I_pixel_flat[index] = (A[index]*I_est_flat[index]+B[index]*Q_est_flat[index]+\
                               C[index]*U_est_flat[index])/Delta[index]
        Q_pixel_flat[index] = (B[index]*I_est_flat[index]+D[index]*Q_est_flat[index]+\
                               E[index]*U_est_flat[index])/Delta[index]
        U_pixel_flat[index] = (C[index]*I_est_flat[index]+E[index]*Q_est_flat[index]+\
                               F[index]*U_est_flat[index])/Delta[index]

        x_len = np.amax(self.pixelmap[:,0])-np.amin(self.pixelmap[:,0])
        y_len = np.amax(self.pixelmap[:,1])-np.amin(self.pixelmap[:,1])

        if len(I_est_flat) < (x_len+1)*(y_len+1):
            valmax = (x_len+1)*(y_len+1)
            pmax = np.amax(param)
            I_fin = 0.*np.arange(pmax+1, valmax)
            Q_fin = 0.*np.arange(pmax+1, valmax)
            U_fin = 0.*np.arange(pmax+1, valmax)
            
            I_pixel_flat = np.append(I_pixel_flat, I_fin)
            Q_pixel_flat = np.append(Q_pixel_flat, Q_fin)
            U_pixel_flat = np.append(U_pixel_flat, U_fin)

        ind_pol, = np.nonzero(Q_pixel_flat)
        pol = np.sqrt(Q_pixel_flat**2+U_pixel_flat**2)

        I_pixel = np.reshape(I_pixel_flat, (y_len+1,x_len+1))
        Q_pixel = np.reshape(Q_pixel_flat, (y_len+1,x_len+1))
        U_pixel = np.reshape(U_pixel_flat, (y_len+1,x_len+1))

        return I_pixel, Q_pixel, U_pixel

    def map_multidetectors(self):

        for i in range(self.number):

            mapvalues = self.map_singledetector(value=self.data[i],sigma=self.weight[i],\
                                           angle=self.polangle[i])
            
            if i == 0:
                I_map = mapvalues[0].copy()
                Q_map = mapvalues[1].copy()
                U_map = mapvalues[2].copy()
            else:
                I_map += mapvalues[0]
                Q_map += mapvalues[1]
                U_map += mapvalues[2]

        return I_map, Q_map, U_map

    def convolution(self, std, map_value):

        #The standard deviation is in pixel value

        kernel = Gaussian2DKernel(stddev=std)

        convolved_map = convolve(map_value, kernel)

        return convolved_map

    