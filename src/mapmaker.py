import numpy as np
import copy
from astropy import wcs
from astropy.convolution import Gaussian2DKernel, convolve
import matplotlib.pyplot as plt

class maps():

    '''
    Wrapper class for the wcs_word class and the mapmaking class.
    In this way in the gui.py only one class is called
    '''

    def __init__(self, ctype, crpix, cdelt, crval, data, coord1, coord2, convolution, std, Ionly=True):

        self.ctype = ctype             #see wcs_world for explanation of this parameter
        self.crpix = crpix             #see wcs_world for explanation of this parameter
        self.cdelt = cdelt             #see wcs_world for explanation of this parameter
        self.crval = crval             #see wcs_world for explanation of this parameter
        self.coord1 = coord1           #array of the first coordinate
        self.coord2 = coord2           #array of the second coordinate
        self.data = data               #cleaned TOD that is used to create a map
        self.w = 0.                    #initialization of the coordinates of the map in pixel coordinates
        self.proj = 0.                 #inizialization of the wcs of the map. see wcs_world for more explanation about projections
        self.convolution = convolution #parameters to check if the convolution is required
        self.std = float(std)          #std of the gaussian is the convolution is required
        self.Ionly = Ionly             #paramters to check if only I is required to be computed

    def wcs_proj(self):

        '''
        Function to compute the projection and the pixel coordinates
        '''

        wcsworld = wcs_world(self.ctype, self.crpix, self.cdelt, self.crval)

        if np.size(np.shape(self.data)) == 1:
            print('COORD_1',np.transpose(np.array([self.coord1, self.coord2])))
            try:
                self.w, self.proj = wcsworld.world(np.transpose(np.array([self.coord1[0], self.coord2[0]])))
            except RuntimeError:
                self.w, self.proj = wcsworld.world(np.transpose(np.array([self.coord1, self.coord2])))
        else:

            if np.size(np.shape(self.coord1)) == 1:
                self.w, self.proj = wcsworld.world(np.transpose(np.array([self.coord1, self.coord2])))
            else:
                print('SIZE', np.size(np.shape(self.data)), len(self.coord1), len(self.coord2))
                self.w = np.zeros((np.size(np.shape(self.data)), len(self.coord1[0]), 2))
                print('Coord',self.coord1)
                for i in range(np.size(np.shape(self.data))):
                    print('COORD',np.transpose(np.array([self.coord1[i], self.coord2[i]])))
                    print('SHAPE', np.shape(self.w[i,:,:]))
                    self.w[i,:,:], self.proj = wcsworld.world(np.transpose(np.array([self.coord1[i], self.coord2[i]])))
                    plt.plot(self.coord1[i])

                plt.show()

    def map2d(self):

        '''
        Function to generate the maps using the pixel coordinates to bin
        '''

        if np.size(np.shape(self.data)) == 1:
            mapmaker = mapmaking(self.data, 1., 1.2, 1, np.floor(self.w).astype(int))
            if self.Ionly:
                Imap = mapmaker.map_singledetector_Ionly(self.crpix)

                if not self.convolution:
                    return Imap
                else:
                    std_pixel = self.std/3600./self.cdelt[0]
                    
                    return mapmaker.convolution(std_pixel, Imap)
            else:        
                Imap, Qmap, Umap = mapmaker.map_singledetector(self.crpix)
                if not self.convolution:
                    return Imap, Qmap, Umap
                else:
                    Imap_con = mapmaker.convolution(self.std, Imap)
                    Qmap_con = mapmaker.convolution(self.std, Qmap)
                    Umap_con = mapmaker.convolution(self.std, Umap)
                    return Imap_con, Qmap_con, Umap_con

        else:
            mapmaker = mapmaking(self.data, 1., 1.2, np.size(np.shape(self.data)), np.floor(self.w).astype(int))
            if self.Ionly:
                print('Multi', np.size(np.shape(self.data)))
                Imap = mapmaker.map_multidetectors_Ionly(self.crpix)

                if not self.convolution:
                    return Imap
                else:
                    std_pixel = self.std/3600./self.cdelt[0]
                    
                    return mapmaker.convolution(std_pixel, Imap)
            else:        
                Imap, Qmap, Umap = mapmaker.map_singledetector(self.crpix)
                if not self.convolution:
                    return Imap, Qmap, Umap
                else:
                    Imap_con = mapmaker.convolution(self.std, Imap)
                    Qmap_con = mapmaker.convolution(self.std, Qmap)
                    Umap_con = mapmaker.convolution(self.std, Umap)
                    return Imap_con, Qmap_con, Umap_con


class wcs_world():

    '''
    Class to generate a wcs using astropy routines.
    '''

    def __init__(self, ctype, crpix, crdelt, crval):

        self.ctype = ctype    #ctype of the map, which projection is used to convert coordinates to pixel numbers
        self.crdelt = crdelt  #cdelt of the map, distance in deg between two close pixels
        self.crpix = crpix    #crpix of the map, central pixel of the map in pixel coordinates
        self.crval = crval    #crval of the map, central pixel of the map in sky/telescope (depending on the system) coordinates

    def world(self, coord):
        
        '''
        Function for creating a wcs projection and a pixel coordinates 
        from sky/telescope coordinates
        '''

        w = wcs.WCS(naxis=2)
        w.wcs.crpix = self.crpix
        w.wcs.cdelt = self.crdelt
        w.wcs.crval = self.crval
        if self.ctype == 'RA and DEC':
            w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        elif self.ctype == 'AZ and EL':
            w.wcs.ctype = ["TLON-ARC", "TLAT-ARC"]
        elif self.ctype == 'CROSS-EL and EL':
            w.wcs.ctype = ["TLON-CAR", "TLAT-CAR"]
        print('Coordinates', coord, type(coord), np.shape(coord))
        world = w.wcs_world2pix(coord, 1)
        #f = open('/Users/ian/git/gabsmap/mapmaker/coordarr.txt','w')
        #for i in range(len(coord)):
        #    print(coord[i][0],'\t',coord[i][1],file = f)
        #f.close()
        #print('printing coords')
        #print(coord)

        return world, w

class mapmaking(object):

    '''
    Class to generate the maps. For more information about the system to be solved
    check Moncelsi et al. 2012
    '''

    def __init__(self, data, weight, polangle, number, pixelmap):

        self.data = data               #detector TOD
        self.weight = weight           #weights associated with the detector values
        self.polangle = polangle       #polarization angles of each detector
        self.number = number           
        self.pixelmap = pixelmap       #Coordinates of each point in the TOD in pixel coordinates

    def map_param(self, crpix, idxpixel, value=None, sigma=None, angle=None):

        '''
        Function to calculate the parameters of the map. Parameters follow the same 
        naming scheme used in the paper
        '''

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
        
        x_map = idxpixel[:,0]   #RA 
        y_map = idxpixel[:,1]   #DEC
        
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

        print('Param', param, np.size(param))
        print('FLUX', flux, np.size(flux))

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

    def map_singledetector_Ionly(self, crpix, value=None, sigma=None, angle=None, idxpixel = None):
        
        '''
        Function to reshape the previous array to create a 2D map for a single detector
        if only I map is requested
        '''

        if idxpixel is None:
            idxpixel = self.pixelmap.copy()
        else:
            idxpixel = idxpixel
        value =self.map_param(crpix=crpix, idxpixel = idxpixel, value=value, sigma=sigma, angle=angle)

        I_flat = np.zeros(len(value[0]))

        I_flat[np.nonzero(value[0])] = value[0][np.nonzero(value[0])]/value[3][np.nonzero(value[0])]

        x_len = np.amax(idxpixel[:,0])-np.amin(idxpixel[:,0])
        y_len = np.amax(idxpixel[:,1])-np.amin(idxpixel[:,1])

        print('Pixel_MIN', np.amin(idxpixel[:,0]), np.amin(idxpixel[:,1]))
        print('Pixel_MIN', np.amax(idxpixel[:,0]), np.amax(idxpixel[:,1]))

        if len(I_flat) < (x_len+1)*(y_len+1):
            valmax = (x_len+1)*(y_len+1)
            pmax = np.amax(value[-1])
            I_fin = 0.*np.arange(pmax+1, valmax)
            
            I_flat = np.append(I_flat, I_fin)

        I_pixel = np.reshape(I_flat, (y_len+1,x_len+1))
    
        return I_pixel

    def map_multidetectors_Ionly(self, crpix):
        print('Multi x2', self.pixelmap)

        Xmin = np.inf
        Xmax = -np.inf
        Ymin = np.inf
        Ymax = -np.inf

        for i in range(self.number):
            if np.size(np.shape(self.pixelmap)) == 2:
                idxpixel = self.pixelmap.copy()
                Xmin, Xmax = np.amin(idxpixel[:, 0]), np.amax(idxpixel[:, 0])
                Ymin, Ymax = np.amin(idxpixel[:, 1]), np.amax(idxpixel[:,1])
                break
            else:
                idxpixel = self.pixelmap[i].copy()
                Xmin = np.amin(np.array([Xmin,np.amin(idxpixel[:, 0])]))
                Xmax = np.amax(np.array([Xmax,np.amax(idxpixel[:, 0])]))
                Ymin = np.amin(np.array([Ymin,np.amin(idxpixel[:, 1])]))
                Ymax = np.amax(np.array([Ymax,np.amax(idxpixel[:, 1])]))
                print('Values', Xmin, Xmax, Ymin, Ymax)
        
        finalmap = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))

        for i in range(self.number):
            print('Det #', i)
            if np.size(np.shape(self.pixelmap)) == 2:
                idxpixel = self.pixelmap.copy()
            else:
                idxpixel = self.pixelmap[i].copy()
            print('Pixel_MIN', np.amin(idxpixel[:,0]), np.amin(idxpixel[:,1]))
            print('Pixel_MIN', np.amax(idxpixel[:,0]), np.amax(idxpixel[:,1]))
            mapvalues = self.map_singledetector_Ionly(crpix = crpix, value=self.data[i],sigma=self.weight,\
                                                      angle=self.polangle, idxpixel = idxpixel)
            print('MapShape', np.shape(mapvalues))

            Xmin_map_temp, Xmax_map_temp = np.amin(idxpixel[:,0]), np.amax(idxpixel[:,0])
            Ymin_map_temp, Ymax_map_temp = np.amin(idxpixel[:,1]), np.amax(idxpixel[:,1])

            index1x = int(Xmin_map_temp-Xmin)
            index2x = int(index1x + np.abs(Xmax_map_temp-Xmin_map_temp))
            index1y = int(Ymin_map_temp-Ymin)
            index2y = int(index1y + np.abs(Ymax_map_temp-Ymin_map_temp))
            print('Indices', index1x,index2x,index1y,index2y)
            print(np.shape(finalmap), np.shape(mapvalues))
            finalmap[index1y:index2y+1,index1x:index2x+1] += mapvalues
            # if i == 0:
            #     I_pixel = mapvalues.copy()
            #     Xmin_map_temp, Xmax_map_temp = np.amin(idxpixel[:,0]), np.amax(idxpixel[:,0])
            #     Ymin_map_temp, Ymax_map_temp = np.amin(idxpixel[:,1]), np.amax(idxpixel[:,1])
            # else:
            #     Xmin, Xmax = np.amin(idxpixel[:,0]), np.amax(idxpixel[:,0])
            #     Ymin, Ymax = np.amin(idxpixel[:,1]), np.amax(idxpixel[:,1])
                
            #     Xmin_map = np.amin(np.array([Xmin, Xmin_map_temp]))
            #     Xmax_map = np.amax(np.array([Xmax, Xmax_map_temp]))
            #     Ymin_map = np.amin(np.array([Ymin, Ymin_map_temp]))
            #     Ymax_map = np.amax(np.array([Ymax, Ymax_map_temp]))



            #     I_pixel += mapvalues

        return finalmap

    def map_singledetector(self, crpix, value=None, sigma=None, angle=None):

        '''
        Function to reshape the previous array to create a 2D map for a single detector
        if also polarization maps are requested
        '''


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

        '''
        Function to convolve the maps with a gaussian.
        STD is in pixel values
        '''

        kernel = Gaussian2DKernel(x_stddev=std)

        convolved_map = convolve(map_value, kernel)

        return convolved_map

    