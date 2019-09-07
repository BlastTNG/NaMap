import numpy as np
from astropy import wcs
from astropy.convolution import Gaussian2DKernel, convolve
import matplotlib.pyplot as plt

class maps():

    '''
    Wrapper class for the wcs_word class and the mapmaking class.
    In this way in the gui.py only one class is called
    '''

    def __init__(self, ctype, crpix, cdelt, crval, data, coord1, coord2, convolution, std, Ionly=True, pol_angle=0.,noise=1.):

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
        self.pol_angle = pol_angle     #polariztion angle
        self.noise = noise             #white level noise of detector(s)

    def wcs_proj(self):

        '''
        Function to compute the projection and the pixel coordinates
        '''

        wcsworld = wcs_world(self.ctype, self.crpix, self.cdelt, self.crval)

        if np.size(np.shape(self.data)) == 1:
            try:
                self.w, self.proj = wcsworld.world(np.transpose(np.array([self.coord1[0], self.coord2[0]])))
            except RuntimeError:
                self.w, self.proj = wcsworld.world(np.transpose(np.array([self.coord1, self.coord2])))
        else:

            if np.size(np.shape(self.coord1)) == 1:
                self.w, self.proj = wcsworld.world(np.transpose(np.array([self.coord1, self.coord2])))
            else:
                self.w = np.zeros((np.size(np.shape(self.data)), len(self.coord1[0]), 2))
                for i in range(np.size(np.shape(self.data))):
                    self.w[i,:,:], self.proj = wcsworld.world(np.transpose(np.array([self.coord1[i], self.coord2[i]])))
                    plt.plot(self.coord1[i])

                plt.show()

    def map2d(self):

        '''
        Function to generate the maps using the pixel coordinates to bin
        '''

        if np.size(np.shape(self.data)) == 1:
            mapmaker = mapmaking(self.data, self.noise, self.pol_angle, 1, np.floor(self.w).astype(int))
            if self.Ionly:
                Imap = mapmaker.map_singledetector_Ionly(self.crpix)

                if not self.convolution:
                    return Imap
                else:
                    std_pixel = self.std/3600./np.abs(self.cdelt[0])
                    
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
            mapmaker = mapmaking(self.data, self.noise, self.pol_angle, np.size(np.shape(self.data)), np.floor(self.w).astype(int))
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
        self.number = number           #Number of detectors to be mapped
        self.pixelmap = pixelmap       #Coordinates of each point in the TOD in pixel coordinates

    def map_param(self, crpix, idxpixel, value=None, noise=None, angle=None):

        '''
        Function to calculate the parameters of the map. Parameters follow the same 
        naming scheme used in the paper
        '''

        if value is None:
            value = self.data.copy()
        if noise is not None:
            sigma = 1/noise**2
        else:
            sigma = 1.
        if np.size(angle) > 1:
            angle = angle.copy()
        else:
            angle = angle*np.ones(np.size(value))

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

    def map_singledetector_Ionly(self, crpix, value=None, noise=None, angle=None, idxpixel = None):
        
        '''
        Function to reshape the previous array to create a 2D map for a single detector
        if only I map is requested
        '''

        if value is None:
            value = self.data.copy()
        else:
            value = value

        if idxpixel is None:
            idxpixel = self.pixelmap.copy()
        else:
            idxpixel = idxpixel
        
        if noise is None:
            noise = 1/self.weight**2
        else:
            noise = noise
        
        if angle is None:
            angle = self.polangle
        else:
            angle = angle

        print('Noise', noise)
        value =self.map_param(crpix=crpix, idxpixel = idxpixel, value=value, noise=noise, angle=angle)

        I_flat = np.zeros(len(value[0]))

        I_flat[np.nonzero(value[0])] = value[0][np.nonzero(value[0])]/value[3][np.nonzero(value[0])]

        x_len = np.amax(idxpixel[:,0])-np.amin(idxpixel[:,0])
        y_len = np.amax(idxpixel[:,1])-np.amin(idxpixel[:,1])

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
        
        finalmap_num = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_den = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))

        for i in range(self.number):
            print('Det #', i)
            if np.size(np.shape(self.pixelmap)) == 2:
                idxpixel = self.pixelmap.copy()
            else:
                idxpixel = self.pixelmap[i].copy()
            print('Pixel_MIN', np.amin(idxpixel[:,0]), np.amin(idxpixel[:,1]))
            print('Pixel_MIN', np.amax(idxpixel[:,0]), np.amax(idxpixel[:,1]))
            # mapvalues = self.map_singledetector_Ionly(crpix = crpix, value=self.data[i],noise=1/self.weight[i]**2,\
            #                                           angle=self.polangle[i], idxpixel = idxpixel)

            value = self.map_param(crpix=crpix, idxpixel = idxpixel, value=self.data[i], noise=1/self.weight[i]**2, angle=self.polangle[i])

            num_temp_flat = np.zeros(len(value[0]))
            num_temp_flat[np.nonzero(value[0])] = value[0][np.nonzero(value[0])]
            
            den_temp_flat = np.zeros_like(num_temp_flat)
            den_temp_flat[np.nonzero(value[0])] = value[3][np.nonzero(value[0])]

            Xmin_map_temp, Xmax_map_temp = np.amin(idxpixel[:,0]), np.amax(idxpixel[:,0])
            Ymin_map_temp, Ymax_map_temp = np.amin(idxpixel[:,1]), np.amax(idxpixel[:,1])

            index1x = int(Xmin_map_temp-Xmin)
            index2x = int(index1x + np.abs(Xmax_map_temp-Xmin_map_temp))
            index1y = int(Ymin_map_temp-Ymin)
            index2y = int(index1y + np.abs(Ymax_map_temp-Ymin_map_temp))
            print('Indices', index1x,index2x,index1y,index2y)

            x_len = Xmax_map_temp-Xmin_map_temp
            y_len = Ymax_map_temp-Ymin_map_temp

            if len(value[0]) < (x_len+1)*(y_len+1):
                valmax = (x_len+1)*(y_len+1)
                pmax = np.amax(value[-1])
                num_temp_fin = 0.*np.arange(pmax+1, valmax)
                den_temp_fin = np.ones(np.abs(pmax+1-valmax))
                
                temp_map_num_flat = np.append(num_temp_flat, num_temp_fin)
                temp_map_den_flat = np.append(den_temp_flat, den_temp_fin)

            temp_map_num = np.reshape(temp_map_num_flat, (y_len+1,x_len+1))
            temp_map_den = np.reshape(temp_map_den_flat, (y_len+1,x_len+1))

            finalmap_num[index1y:index2y+1,index1x:index2x+1] += temp_map_num
            finalmap_den[index1y:index2y+1,index1x:index2x+1] += temp_map_den

        finalmap = finalmap_num/finalmap_den

        return finalmap

    def map_singledetector(self, crpix, value=None, sigma=None, angle=None, idxpixel=None):

        '''
        Function to reshape the previous array to create a 2D map for a single detector
        if also polarization maps are requested
        '''

        if idxpixel is None:
            idxpixel = self.pixelmap.copy()
        else:
            idxpixel = idxpixel

        (I_est_flat, Q_est_flat, U_est_flat, N_hits_flat, Delta, \
         A, B, C, D, E, F, param) = self.map_param(crpix=crpix, idxpixel=idxpixel, value=value, \
                                                   sigma=1/self.weight**2,angle=self.polangle)

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

        x_len = np.amax(idxpixel[:,0])-np.amin(idxpixel[:,0])
        y_len = np.amax(idxpixel[:,1])-np.amin(idxpixel[:,1])

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

    def map_multidetectors(self, crpix):


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
        
        finalmap_I_num = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_Q_num = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_U_num = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_den = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))

        for i in range(self.number):
            if np.size(np.shape(self.pixelmap)) == 2:
                idxpixel = self.pixelmap.copy()
            else:
                idxpixel = self.pixelmap[i].copy()

            # mapvalues = self.map_singledetector(crpix = crpix, value=self.data[i],sigma=1/self.weight[i]**2,\
            #                                     angle=self.polangle[i], idxpixel = idxpixel)

            (I_est_flat, Q_est_flat, U_est_flat, N_hits_flat, Delta, \
             A, B, C, D, E, F, param) = self.map_param(crpix=crpix, idxpixel=idxpixel, value=self.data[i], \
                                                       sigma=1/self.weight[i]**2,angle=self.polangle[i])

            I_num_flat = np.zeros(len(I_est_flat))
            Q_num_flat = np.zeros(len(Q_est_flat))
            U_num_flat = np.zeros(len(U_est_flat))
            den_flat = np.zeros(len(I_est_flat))


            index, = np.where(np.abs(Delta)>0.)
            
            I_num_flat[index] = (A[index]*I_est_flat[index]+B[index]*Q_est_flat[index]+\
                                 C[index]*U_est_flat[index])
            Q_num_flat[index] = (B[index]*I_est_flat[index]+D[index]*Q_est_flat[index]+\
                                 E[index]*U_est_flat[index])
            U_num_flat[index] = (C[index]*I_est_flat[index]+E[index]*Q_est_flat[index]+\
                                 F[index]*U_est_flat[index])

            den_flat[index] = Delta[index]

            Xmin_map_temp, Xmax_map_temp = np.amin(idxpixel[:,0]), np.amax(idxpixel[:,0])
            Ymin_map_temp, Ymax_map_temp = np.amin(idxpixel[:,1]), np.amax(idxpixel[:,1])

            index1x = int(Xmin_map_temp-Xmin)
            index2x = int(index1x + np.abs(Xmax_map_temp-Xmin_map_temp))
            index1y = int(Ymin_map_temp-Ymin)
            index2y = int(index1y + np.abs(Ymax_map_temp-Ymin_map_temp))

            x_len = Xmax_map_temp-Xmin_map_temp
            y_len = Ymax_map_temp-Ymin_map_temp

            if len(I_num_flat) < (x_len+1)*(y_len+1):
                valmax = (x_len+1)*(y_len+1)
                pmax = np.amax(param)
                I_num_fin = 0.*np.arange(pmax+1, valmax)
                Q_num_fin = 0.*np.arange(pmax+1, valmax)
                U_num_fin = 0.*np.arange(pmax+1, valmax)
                den_fin = np.ones(np.abs(pmax+1-valmax))
                
                I_num_flat = np.append(I_num_flat, I_num_fin)
                Q_num_flat = np.append(Q_num_flat, Q_num_fin)
                U_num_flat = np.append(U_num_flat, U_num_fin)
                den_flat = np.append(den_flat, den_fin)


            I_temp_num = np.reshape(I_num_flat, (y_len+1,x_len+1))
            Q_temp_num = np.reshape(Q_num_flat, (y_len+1,x_len+1))
            U_temp_num = np.reshape(U_num_flat, (y_len+1,x_len+1))
            temp_den = np.reshape(den_flat, (y_len+1,x_len+1))

            finalmap_I_num[index1y:index2y+1,index1x:index2x+1] += I_temp_num
            finalmap_Q_num[index1y:index2y+1,index1x:index2x+1] += Q_temp_num
            finalmap_U_num[index1y:index2y+1,index1x:index2x+1] += U_temp_num
            finalmap_den[index1y:index2y+1,index1x:index2x+1] += temp_den

        finalmap_I = finalmap_I_num/finalmap_den
        finalmap_Q = finalmap_Q_num/finalmap_den
        finalmap_U = finalmap_U_num/finalmap_den

        return finalmap_I, finalmap_Q, finalmap_U


    def convolution(self, std, map_value):

        '''
        Function to convolve the maps with a gaussian.
        STD is in pixel values
        '''

        kernel = Gaussian2DKernel(x_stddev=std)

        convolved_map = convolve(map_value, kernel)

        return convolved_map

    