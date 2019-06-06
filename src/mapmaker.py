import numpy as np
import copy
<<<<<<< HEAD
<<<<<<< HEAD
from astropy import wcs, coordinates
=======
from astropy import wcs
>>>>>>> 8989c24... Correct calculation of coordinates
from astropy.convolution import Gaussian2DKernel, convolve

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
        self.coord1 = np.degrees(coord1)          #array of the first coordinate
        self.coord2 = np.degrees(coord2)           #array of the second coordinate
        self.data = data               #cleaned TOD that is used to create a map
        self.w = 0.                    #initialization of the coordinates of the map in pixel coordinates
        self.proj = 0.                 #inizialization of the wcs of the map. see wcs_world for more explanation about projections
        self.convolution = convolution #parameters to check if the convolution is required
        self.std = float(std)          #std of the gaussian is the convolution is required
        self.Ionly = Ionly             #paramters to check if only I is required to be computed

    def wcs_proj(self):
<<<<<<< HEAD
<<<<<<< HEAD

=======
from astropy import wcs
from astropy.convolution import Gaussian2DKernel, convolve

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
        self.coord1 = np.degrees(coord1)          #array of the first coordinate
        self.coord2 = np.degrees(coord2)           #array of the second coordinate
        self.data = data               #cleaned TOD that is used to create a map
        self.w = 0.                    #initialization of the coordinates of the map in pixel coordinates
        self.proj = 0.                 #inizialization of the wcs of the map. see wcs_world for more explanation about projections
        self.convolution = convolution #parameters to check if the convolution is required
        self.std = float(std)          #std of the gaussian is the convolution is required
        self.Ionly = Ionly             #paramters to check if only I is required to be computed

    def wcs_proj(self):
=======
>>>>>>> 651e1e6... Commented files

        '''
        Function to compute the projection and the pixel coordinates
        '''

<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> 4ee3dbe... Fixed bug in selecting data
=======
>>>>>>> 651e1e6... Commented files
        wcsworld = wcs_world(self.ctype, self.crpix, self.cdelt, self.crval)

        self.w, self.proj = wcsworld.world(np.transpose(np.array([self.coord1, self.coord2])))

    def map2d(self):

        '''
        Function to generate the maps using the pixel coordinates to bin
        '''

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

<<<<<<< HEAD
<<<<<<< HEAD
        mapmaker = mapmaking(self.data, 1., 1., 1, np.floor(self.w).astype(int))
        finalI = mapmaker.map_singledetector_Ionly(self.crpix)
        if self.convolution == False:
            return finalI
        elif self.convolution == True:
            convolution_map = mapmaker.convolution(self.std, finalI)
            return convolution_map
=======
        '''
        Function to generate the maps using the pixel coordinates to bin
        '''

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

>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)
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

<<<<<<< HEAD
<<<<<<< HEAD
=======
    '''
    Class to generate a wcs using astropy routines.
    '''

>>>>>>> 651e1e6... Commented files
    def __init__(self, ctype, crpix, crdelt, crval):

        self.ctype = ctype    #ctype of the map, which projection is used to convert coordinates to pixel numbers
        self.crdelt = crdelt  #cdelt of the map, distance in deg between two close pixels
        self.crpix = crpix    #crpix of the map, central pixel of the map in pixel coordinates
        self.crval = crval    #crval of the map, central pixel of the map in sky/telescope (depending on the system) coordinates

    def world(self, coord):
        
<<<<<<< HEAD
=======
    '''
    Class to generate a wcs using astropy routines.
    '''

    def __init__(self, ctype, crpix, crdelt, crval):

        self.ctype = ctype    #ctype of the map, which projection is used to convert coordinates to pixel numbers
        self.crdelt = crdelt  #cdelt of the map, distance in deg between two close pixels
        self.crpix = crpix    #crpix of the map, central pixel of the map in pixel coordinates
        self.crval = crval    #crval of the map, central pixel of the map in sky/telescope (depending on the system) coordinates

    def world(self, coord):
        
=======
>>>>>>> 651e1e6... Commented files
        '''
        Function for creating a wcs projection and a pixel coordinates 
        from sky/telescope coordinates
        '''

<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> 651e1e6... Commented files
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = self.crpix
        w.wcs.cdelt = self.crdelt
        w.wcs.crval = self.crval
<<<<<<< HEAD
<<<<<<< HEAD
        if self.ctype.lower() == 'RA and DEC':
            w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
        elif self.ctype.lower() == 'AZ and EL' or self.ctype.lower() == 'CROSS-EL and EL':
=======
        if self.ctype == 'RA and DEC':
            w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
<<<<<<< HEAD
        elif self.ctype == 'AZ and EL' or self.ctype == 'CROSS-EL and EL':
>>>>>>> 4ee3dbe... Fixed bug in selecting data
            w.wcs.ctype = ["TLON-ARC", "TLAT-ARC"]
=======
        if self.ctype == 'RA and DEC':
            w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
=======
>>>>>>> 8989c24... Correct calculation of coordinates
        elif self.ctype == 'AZ and EL':
            w.wcs.ctype = ["TLON-ARC", "TLAT-ARC"]
        elif self.ctype == 'CROSS-EL and EL':
            w.wcs.ctype = ["TLON-CAR", "TLAT-CAR"]
<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> 8989c24... Correct calculation of coordinates
        world = w.wcs_world2pix(coord, 1)

        return world, w

class mapmaking(object):

<<<<<<< HEAD
<<<<<<< HEAD
=======
    '''
    Class to generate the maps. For more information about the system to be solved
    check Moncelsi et al. 2012
    '''

>>>>>>> 651e1e6... Commented files
    def __init__(self, data, weight, polangle, number, pixelmap):

        self.data = data               #detector TOD
        self.weight = weight           #weights associated with the detector values
        self.polangle = polangle       #polarization angles of each detector
        self.number = number           
        self.pixelmap = pixelmap       #Coordinates of each point in the TOD in pixel coordinates

    def map_param(self, crpix, value=None, sigma=None, angle=None):

<<<<<<< HEAD
<<<<<<< HEAD
        if value == None:
=======
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

    def map_param(self, crpix, value=None, sigma=None, angle=None):

=======
>>>>>>> 651e1e6... Commented files
        '''
        Function to calculate the parameters of the map. Parameters follow the same 
        naming scheme used in the paper
        '''

<<<<<<< HEAD
        if value is None:
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
=======
>>>>>>> 651e1e6... Commented files
        if value is None:
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)
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
        
<<<<<<< HEAD
<<<<<<< HEAD
        if np.abs(np.amin(x_map)) <= 0:
=======
        if (np.amin(x_map)) <= 0:
<<<<<<< HEAD
            print('MIN')
>>>>>>> 4ee3dbe... Fixed bug in selecting data
            x_map = np.floor(x_map+np.abs(np.amin(x_map)))
        else:
            x_map = np.floor(x_map-np.amin(x_map))
        if (np.amin(y_map)) <= 0:
            y_map = np.floor(y_map+np.abs(np.amin(y_map)))
        else:
<<<<<<< HEAD
            y_map = np.round(y_map-np.amin(y_map))
=======
        if (np.amin(x_map)) <= 0:
=======
>>>>>>> 6acdf4e... Solved a memory leak when trying to replot with different parameters
            x_map = np.floor(x_map+np.abs(np.amin(x_map)))
        else:
            x_map = np.floor(x_map-np.amin(x_map))
        if (np.amin(y_map)) <= 0:
            y_map = np.floor(y_map+np.abs(np.amin(y_map)))
        else:
            y_map = np.floor(y_map-np.amin(y_map))
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
            y_map = np.floor(y_map-np.amin(y_map))
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)

        x_len = np.amax(x_map)-np.amin(x_map)+1
        param = x_map+y_map*x_len
        param = param.astype(int)

        flux = value

        cos = np.cos(2.*angle)
        sin = np.sin(2.*angle)

        I_est_flat = np.bincount(param, weights=flux)*sigma
<<<<<<< HEAD
<<<<<<< HEAD
        Q_est_flat = np.bincount(param, weights=flux*cos)
        U_est_flat = np.bincount(param, weights=flux*sin)
=======
        Q_est_flat = np.bincount(param, weights=flux*cos)*sigma
        U_est_flat = np.bincount(param, weights=flux*sin)*sigma
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
        Q_est_flat = np.bincount(param, weights=flux*cos)*sigma
        U_est_flat = np.bincount(param, weights=flux*sin)*sigma
<<<<<<< HEAD
        
        
        print(np.amax(I_est_flat))
        print(type(sigma), sigma)
        print(angle)
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)
=======
>>>>>>> 4ee3dbe... Fixed bug in selecting data

        N_hits_flat = 0.5*np.bincount(param)*sigma
        c_flat = np.bincount(param, weights=0.5*cos)*sigma
        c2_flat = np.bincount(param, weights=0.5*cos**2)*sigma
        s_flat = np.bincount(param, weights=0.5*sin)*sigma
        s2_flat = N_hits_flat-c2_flat
        m_flat = np.bincount(param, weights=0.5*cos*sin)*sigma
<<<<<<< HEAD
<<<<<<< HEAD

<<<<<<< HEAD
        Delta = c_flat**2*(c2_flat-N_hits_flat)+2*s_flat*c_flat*m_flat-c2_flat*s_flat**2-\
                N_hits_flat*(c2_flat**2+m_flat**2-c2_flat*N_hits_flat)
=======
=======
>>>>>>> 4ee3dbe... Fixed bug in selecting data
        Delta = (c_flat**2*(c2_flat-N_hits_flat)+2*s_flat*c_flat*m_flat-c2_flat*s_flat**2-\
                 N_hits_flat*(c2_flat**2+m_flat**2-c2_flat*N_hits_flat))
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
        Delta = (c_flat**2*(c2_flat-N_hits_flat)+2*s_flat*c_flat*m_flat-c2_flat*s_flat**2-\
                 N_hits_flat*(c2_flat**2+m_flat**2-c2_flat*N_hits_flat))
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)
        A = -(c2_flat**2+m_flat**2-c2_flat*N_hits_flat)
        B = c_flat*(c2_flat-N_hits_flat)+s_flat*m_flat
        C = c_flat*m_flat-s_flat*c2_flat
        D = -((c2_flat-N_hits_flat)*N_hits_flat+s_flat**2)
        E = c_flat*s_flat-m_flat*N_hits_flat
        F = c2_flat*N_hits_flat-c_flat**2

        return I_est_flat, Q_est_flat, U_est_flat, N_hits_flat, Delta, A, B, C, D, E, F, param

    def map_singledetector_Ionly(self, crpix, value=None, sigma=None, angle=None):
<<<<<<< HEAD
<<<<<<< HEAD

        value =self.map_param(crpix=crpix, value=value, sigma=sigma, angle=angle)

        I_flat = np.zeros(len(value[0]))

<<<<<<< HEAD
        I[np.nonzero(value[0])] = value[0][np.nonzero(value[0])]/value[3][np.nonzero(value[0])]
=======
=======
>>>>>>> 651e1e6... Commented files
        
        '''
        Function to reshape the previous array to create a 2D map for a single detector
        if only I map is requested
        '''

        value =self.map_param(crpix=crpix, value=value, sigma=sigma, angle=angle)

        I_flat = np.zeros(len(value[0]))

        I_flat[np.nonzero(value[0])] = value[0][np.nonzero(value[0])]/value[3][np.nonzero(value[0])]
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
        I_flat[np.nonzero(value[0])] = value[0][np.nonzero(value[0])]/value[3][np.nonzero(value[0])]
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)

        x_len = np.amax(self.pixelmap[:,0])-np.amin(self.pixelmap[:,0])
        y_len = np.amax(self.pixelmap[:,1])-np.amin(self.pixelmap[:,1])

<<<<<<< HEAD
<<<<<<< HEAD
        if len(I) < (x_len+1)*(y_len+1):
=======
        if len(I_flat) < (x_len+1)*(y_len+1):
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
        if len(I_flat) < (x_len+1)*(y_len+1):
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)
            valmax = (x_len+1)*(y_len+1)
            pmax = np.amax(value[-1])
            I_fin = 0.*np.arange(pmax+1, valmax)
            
<<<<<<< HEAD
<<<<<<< HEAD
            I = np.append(I, I_fin)

        I_pixel = np.reshape(I, (y_len+1,x_len+1))
        
        x_map = self.pixelmap[:,0]
        y_map = self.pixelmap[:,1]

        x_min = np.amin(x_map)
        y_min = np.amin(y_map)

=======
            I_flat = np.append(I_flat, I_fin)

        I_pixel = np.reshape(I_flat, (y_len+1,x_len+1))
    
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
            I_flat = np.append(I_flat, I_fin)

        I_pixel = np.reshape(I_flat, (y_len+1,x_len+1))
    
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)
        return I_pixel

    def map_multidetectors_Ionly(self):

        for i in range(self.number):
            mapvalues = self.map_singledetector_Ionly(value=self.data[i],sigma=self.weight[i],\
<<<<<<< HEAD
<<<<<<< HEAD
                                                 angle=self.polangle[i])
=======
                                                      angle=self.polangle[i])
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
                                                      angle=self.polangle[i])
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)
            if i == 0:
                I_pixel = mapvalues[0].copy()
            else:
                I_pixel += mapvalues[0]

        return I_pixel 

<<<<<<< HEAD
<<<<<<< HEAD
    def map_singledetector(self, value=None, sigma=None, angle=None):
=======
    def map_singledetector(self, crpix, value=None, sigma=None, angle=None):
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)


        (I_est_flat, Q_est_flat, U_est_flat, N_hits_flat, Delta, \
         A, B, C, D, E, F, param) = self.map_param(crpix=crpix, value=value, sigma=sigma,angle=angle)

<<<<<<< HEAD
        I_pixel_flat = (A*I_est_flat+B*Q_est_flat+C*U_est_flat)/Delta
        Q_pixel_flat = (B*I_est_flat+D*Q_est_flat+E*U_est_flat)/Delta
        U_pixel_flat = (C*I_est_flat+E*Q_est_flat+F*U_est_flat)/Delta
=======
    def map_singledetector(self, crpix, value=None, sigma=None, angle=None):

        '''
        Function to reshape the previous array to create a 2D map for a single detector
        if also polarization maps are requested
        '''


        (I_est_flat, Q_est_flat, U_est_flat, N_hits_flat, Delta, \
         A, B, C, D, E, F, param) = self.map_param(crpix=crpix, value=value, sigma=sigma,angle=angle)

=======
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)
        I_pixel_flat = np.zeros(len(I_est_flat))
        Q_pixel_flat = np.zeros(len(Q_est_flat))
        U_pixel_flat = np.zeros(len(U_est_flat))

<<<<<<< HEAD
<<<<<<< HEAD
        index, = np.where(np.abs(Delta)>0.)
        
=======
        #index = np.nonzero(I_est_flat)
        #idx = np.nonzero(N_hits_flat)
        #idx_delta = np.nonzero(Delta)
        print('Test')
        print(A[16668],B[16668], C[16668], D[16668], E[16668], F[16668], N_hits_flat[16668], Delta[16668], np.bincount(param)[16668])
        print(I_est_flat[593])
        print(Q_est_flat[593])
        print(U_est_flat[593])
        #print(index)
        #print(idx)
        #print(idx_delta)

        index, = np.where(np.abs(Delta)>0.)
        
        # plt.plot(Delta)
        # plt.show()
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)
=======
        index, = np.where(np.abs(Delta)>0.)
        
>>>>>>> 4ee3dbe... Fixed bug in selecting data
        I_pixel_flat[index] = (A[index]*I_est_flat[index]+B[index]*Q_est_flat[index]+\
                               C[index]*U_est_flat[index])/Delta[index]
        Q_pixel_flat[index] = (B[index]*I_est_flat[index]+D[index]*Q_est_flat[index]+\
                               E[index]*U_est_flat[index])/Delta[index]
        U_pixel_flat[index] = (C[index]*I_est_flat[index]+E[index]*Q_est_flat[index]+\
                               F[index]*U_est_flat[index])/Delta[index]
<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======

<<<<<<< HEAD
        print('Delta', np.size(index))
        print('I',np.size(np.nonzero(I_pixel_flat)[0]))
        print(index)
        print(np.nonzero(I_pixel_flat))
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)

=======
>>>>>>> 4ee3dbe... Fixed bug in selecting data
        x_len = np.amax(self.pixelmap[:,0])-np.amin(self.pixelmap[:,0])
        y_len = np.amax(self.pixelmap[:,1])-np.amin(self.pixelmap[:,1])

        if len(I_est_flat) < (x_len+1)*(y_len+1):
            valmax = (x_len+1)*(y_len+1)
<<<<<<< HEAD
<<<<<<< HEAD
            pmax = np.amax(value[-1])
=======
            pmax = np.amax(param)
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
            pmax = np.amax(param)
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)
            I_fin = 0.*np.arange(pmax+1, valmax)
            Q_fin = 0.*np.arange(pmax+1, valmax)
            U_fin = 0.*np.arange(pmax+1, valmax)
            
            I_pixel_flat = np.append(I_pixel_flat, I_fin)
            Q_pixel_flat = np.append(Q_pixel_flat, Q_fin)
            U_pixel_flat = np.append(U_pixel_flat, U_fin)

<<<<<<< HEAD
<<<<<<< HEAD
=======
        ind_pol, = np.nonzero(Q_pixel_flat)
        pol = np.sqrt(Q_pixel_flat**2+U_pixel_flat**2)

>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
        ind_pol, = np.nonzero(Q_pixel_flat)
        pol = np.sqrt(Q_pixel_flat**2+U_pixel_flat**2)

>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)
        I_pixel = np.reshape(I_pixel_flat, (y_len+1,x_len+1))
        Q_pixel = np.reshape(Q_pixel_flat, (y_len+1,x_len+1))
        U_pixel = np.reshape(U_pixel_flat, (y_len+1,x_len+1))

        return I_pixel, Q_pixel, U_pixel

    def map_multidetectors(self):

        for i in range(self.number):

            mapvalues = self.map_singledetector(value=self.data[i],sigma=self.weight[i],\
<<<<<<< HEAD
<<<<<<< HEAD
                                           angle=self.polangle[i])
=======
                                                angle=self.polangle[i])
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
                                                angle=self.polangle[i])
>>>>>>> 8989c24... Correct calculation of coordinates
            
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

<<<<<<< HEAD
<<<<<<< HEAD
        #The standard deviation is in pixel value
=======
        '''
        Function to convolve the maps with a gaussian.
        STD is in pixel values
        '''
>>>>>>> 651e1e6... Commented files

<<<<<<< HEAD
        kernel = Gaussian2DKernel(stddev=std)
=======
        '''
        Function to convolve the maps with a gaussian.
        STD is in pixel values
        '''

        kernel = Gaussian2DKernel(x_stddev=std)
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
        kernel = Gaussian2DKernel(x_stddev=std)
>>>>>>> fa75c3a... Reduced memory consumption

        convolved_map = convolve(map_value, kernel)

        return convolved_map

    