import numpy as np
<<<<<<< HEAD
import copy
<<<<<<< HEAD
<<<<<<< HEAD
from astropy import wcs, coordinates
=======
=======
>>>>>>> d11dfe9... Solved pointing, multidetectors stacking and loading bugs
from astropy import wcs
>>>>>>> 8989c24... Correct calculation of coordinates
from astropy.convolution import Gaussian2DKernel, convolve

class maps():

    '''
    Wrapper class for the wcs_word class and the mapmaking class.
    In this way in the gui.py only one class is called
    '''

    def __init__(self, ctype, crpix, cdelt, crval, data, coord1, coord2, convolution, std, Ionly=True, pol_angle=0.,noise=1., \
                 telcoord=False, parang=None):

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
        self.telcoord = telcoord       #If True the map is drawn in telescope coordinates. That means that the projected plane is rotated
        if parang is not None:
            self.parang = np.radians(parang)   #Parallactic Angle. This is used to compute the pixel indices in telescopes coordinates
        else:
            self.parang = parang

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
<<<<<<< HEAD

<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> 4ee3dbe... Fixed bug in selecting data
=======
>>>>>>> 651e1e6... Commented files
        wcsworld = wcs_world(self.ctype, self.crpix, self.cdelt, self.crval)
=======
        print('IN')
        wcsworld = wcs_world(self.ctype, self.crpix, self.cdelt, self.crval, self.telcoord, self.parang)
>>>>>>> 59060e4... Correct telescope coordinates calculation
=======
        wcsworld = wcs_world(self.ctype, self.crpix, self.cdelt, self.crval, self.telcoord)
>>>>>>> 5af97ad... Corrected calculation of Polarization Angle and bug fix

        if np.size(np.shape(self.data)) == 1:

            if np.size(np.shape(self.coord1)) != 1:
                coord_array = np.transpose(np.array([self.coord1[0], self.coord2[0]]))
            else:
                coord_array = np.transpose(np.array([self.coord1, self.coord2]))
            try:
                self.w, self.proj = wcsworld.world(coord_array, self.parang)
            except RuntimeError:
                self.w, self.proj = wcsworld.world(coord_array, self.parang)
        else:
            if np.size(np.shape(self.coord1)) == 1:
                self.w, self.proj = wcsworld.world(np.transpose(np.array([self.coord1, self.coord2])), self.parang[0,:])
            else:
                self.w = np.zeros((np.size(np.shape(self.data)), len(self.coord1[0]), 2))
                for i in range(np.size(np.shape(self.data))):
                    self.w[i,:,:], self.proj = wcsworld.world(np.transpose(np.array([self.coord1[i], self.coord2[i]])), self.parang[i,:])

    def map2d(self):

        '''
        Function to generate the maps using the pixel coordinates to bin
        '''

<<<<<<< HEAD
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
=======
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
>>>>>>> db74452... Solved a bug on applying the offset

        else:
            mapmaker = mapmaking(self.data, self.noise, self.pol_angle, np.size(np.shape(self.data)), np.floor(self.w).astype(int))
            if self.Ionly:
                Imap = mapmaker.map_multidetectors_Ionly(self.crpix)

                if not self.convolution:
                    return Imap
                else:
                    std_pixel = self.std/3600./self.cdelt[0]
                    
                    return mapmaker.convolution(std_pixel, Imap)
            else:        
                Imap, Qmap, Umap = mapmaker.map_multidetectors(self.crpix)
                if not self.convolution:
                    return Imap, Qmap, Umap
                else:
                    Imap_con = mapmaker.convolution(self.std, Imap)
                    Qmap_con = mapmaker.convolution(self.std, Qmap)
                    Umap_con = mapmaker.convolution(self.std, Umap)
                    return Imap_con, Qmap_con, Umap_con


class wcs_world():

<<<<<<< HEAD
<<<<<<< HEAD
=======
    '''
    Class to generate a wcs using astropy routines.
    '''

<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> 651e1e6... Commented files
    def __init__(self, ctype, crpix, crdelt, crval):
=======
    def __init__(self, ctype, crpix, crdelt, crval, telcoord=False, parang=None):
>>>>>>> 59060e4... Correct telescope coordinates calculation
=======
    def __init__(self, ctype, crpix, crdelt, crval, telcoord=False):
>>>>>>> 5af97ad... Corrected calculation of Polarization Angle and bug fix

        self.ctype = ctype    #ctype of the map, which projection is used to convert coordinates to pixel numbers
        self.crdelt = crdelt  #cdelt of the map, distance in deg between two close pixels
        self.crpix = crpix    #crpix of the map, central pixel of the map in pixel coordinates
        self.crval = crval    #crval of the map, central pixel of the map in sky/telescope (depending on the system) coordinates
        self.telcoord = telcoord #Telescope coordinates boolean value. Check map class for more explanation

    def world(self, coord, parang):
        
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
<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> 8989c24... Correct calculation of coordinates
=======
        print('Coordinates', coord, type(coord), np.shape(coord))
>>>>>>> 26b52b6... Solved projections in plotting and add plotting functions
        world = w.wcs_world2pix(coord, 1)
        #f = open('/Users/ian/git/gabsmap/mapmaker/coordarr.txt','w')
        #for i in range(len(coord)):
        #    print(coord[i][0],'\t',coord[i][1],file = f)
        #f.close()
        #print('printing coords')
        #print(coord)
=======
>>>>>>> 59060e4... Correct telescope coordinates calculation

        if self.telcoord is False:
            if self.ctype == 'RA and DEC':
                w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            elif self.ctype == 'AZ and EL':
                w.wcs.ctype = ["TLON-ARC", "TLAT-ARC"]
            elif self.ctype == 'CROSS-EL and EL':
                w.wcs.ctype = ["TLON-CAR", "TLAT-CAR"]
            world = w.all_world2pix(coord, 1)
            
        else:
            w.wcs.ctype = ["TLON-TAN", "TLAT-TAN"]
            world = np.zeros_like(coord)
            px = w.wcs.s2p(coord, 1)
            #Use the parallactic angle to rotate the projected plane
            world[:,0] = (px['imgcrd'][:,0]*np.cos(parang)-px['imgcrd'][:,1]*np.sin(parang))/self.crdelt[0]+self.crpix[0]
            world[:,1] = (px['imgcrd'][:,0]*np.sin(parang)+px['imgcrd'][:,1]*np.cos(parang))/self.crdelt[1]+self.crpix[1]
        
        print('WORLD', world, np.shape(world))
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
        self.number = number           #Number of detectors to be mapped
        self.pixelmap = pixelmap       #Coordinates of each point in the TOD in pixel coordinates

    def map_param(self, crpix, idxpixel, value=None, noise=None, angle=None):

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
<<<<<<< HEAD
        Q_est_flat = np.bincount(param, weights=flux*cos)
        U_est_flat = np.bincount(param, weights=flux*sin)
=======
=======
        print('ARRAY', param, np.size(param))
        print('FLUX', flux, np.size(flux))
        print('COS', cos, np.size(cos))
>>>>>>> 6f562c7... Solved array shape issue with parallactic angle
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
=======
>>>>>>> 59060e4... Correct telescope coordinates calculation

        return I_est_flat, Q_est_flat, U_est_flat, N_hits_flat, c_flat, c2_flat, s_flat, s2_flat, m_flat, param

<<<<<<< HEAD
<<<<<<< HEAD
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
=======
    def map_singledetector_Ionly(self, crpix, value=None, sigma=None, angle=None, idxpixel = None):
>>>>>>> db74452... Solved a bug on applying the offset
=======
    def map_singledetector_Ionly(self, crpix, value=None, noise=None, angle=None, idxpixel = None):
>>>>>>> d11dfe9... Solved pointing, multidetectors stacking and loading bugs
        
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
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
        I_flat[np.nonzero(value[0])] = value[0][np.nonzero(value[0])]/value[3][np.nonzero(value[0])]
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)

        x_len = np.amax(idxpixel[:,0])-np.amin(idxpixel[:,0])
        y_len = np.amax(idxpixel[:,1])-np.amin(idxpixel[:,1])

<<<<<<< HEAD
        print('Pixel_MIN', np.amin(idxpixel[:,0]), np.amin(idxpixel[:,1]))
        print('Pixel_MIN', np.amax(idxpixel[:,0]), np.amax(idxpixel[:,1]))

<<<<<<< HEAD
<<<<<<< HEAD
        if len(I) < (x_len+1)*(y_len+1):
=======
        if len(I_flat) < (x_len+1)*(y_len+1):
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
=======
>>>>>>> d11dfe9... Solved pointing, multidetectors stacking and loading bugs
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

    def map_multidetectors_Ionly(self, crpix):
        print('Multi x2', self.pixelmap)

        Xmin = np.inf
        Xmax = -np.inf
        Ymin = np.inf
        Ymax = -np.inf

        for i in range(self.number):
<<<<<<< HEAD
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
=======
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
>>>>>>> db74452... Solved a bug on applying the offset
            else:
                idxpixel = self.pixelmap[i].copy()
            print('Pixel_MIN', np.amin(idxpixel[:,0]), np.amin(idxpixel[:,1]))
            print('Pixel_MIN', np.amax(idxpixel[:,0]), np.amax(idxpixel[:,1]))
            # mapvalues = self.map_singledetector_Ionly(crpix = crpix, value=self.data[i],noise=1/self.weight[i],\
            #                                           angle=self.polangle[i], idxpixel = idxpixel)

            value = self.map_param(crpix=crpix, idxpixel = idxpixel, value=self.data[i], noise=1/self.weight[i], angle=self.polangle[i])

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

<<<<<<< HEAD
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
=======
    def map_singledetector(self, crpix, value=None, sigma=None, angle=None, idxpixel=None):
>>>>>>> d11dfe9... Solved pointing, multidetectors stacking and loading bugs

        '''
        Function to reshape the previous array to create a 2D map for a single detector
        if also polarization maps are requested
        '''

        if idxpixel is None:
            idxpixel = self.pixelmap.copy()
        else:
            idxpixel = idxpixel

        (I_est_flat, Q_est_flat, U_est_flat, N_hits_flat, \
         c_flat, c2_flat, s_flat, s2_flat, m_flat, param) = self.map_param(crpix=crpix, idxpixel=idxpixel, value=value, \
                                                                           noise=1/self.weight**2,angle=self.polangle)

        Delta = (c_flat**2*(c2_flat-N_hits_flat)+2*s_flat*c_flat*m_flat-c2_flat*s_flat**2-\
                 N_hits_flat*(c2_flat**2+m_flat**2-c2_flat*N_hits_flat))
        A = -(c2_flat**2+m_flat**2-c2_flat*N_hits_flat)
        B = c_flat*(c2_flat-N_hits_flat)+s_flat*m_flat
        C = c_flat*m_flat-s_flat*c2_flat
        D = -((c2_flat-N_hits_flat)*N_hits_flat+s_flat**2)
        E = c_flat*s_flat-m_flat*N_hits_flat
        F = c2_flat*N_hits_flat-c_flat**2

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
=======

        x_len = np.amax(idxpixel[:,0])-np.amin(idxpixel[:,0])
        y_len = np.amax(idxpixel[:,1])-np.amin(idxpixel[:,1])
>>>>>>> d11dfe9... Solved pointing, multidetectors stacking and loading bugs

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
        #pol = np.sqrt(Q_pixel_flat**2+U_pixel_flat**2)

>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)
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
        
        finalmap_I_est = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_Q_est = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_U_est = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_N_hits = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_c = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_c2 = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_s = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_s2 = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_m = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_I = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_Q = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_U = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))

<<<<<<< HEAD
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
=======
        for i in range(self.number):
            if np.size(np.shape(self.pixelmap)) == 2:
                idxpixel = self.pixelmap.copy()
>>>>>>> d11dfe9... Solved pointing, multidetectors stacking and loading bugs
            else:
                idxpixel = self.pixelmap[i].copy()

            (I_est_flat, Q_est_flat, U_est_flat, N_hits_flat, \
             c_flat, c2_flat, s_flat, s2_flat, m_flat, param) = self.map_param(crpix=crpix, idxpixel=idxpixel, value=self.data[i], \
                                                                               noise=1/self.weight[i],angle=self.polangle[i])

            Xmin_map_temp, Xmax_map_temp = np.amin(idxpixel[:,0]), np.amax(idxpixel[:,0])
            Ymin_map_temp, Ymax_map_temp = np.amin(idxpixel[:,1]), np.amax(idxpixel[:,1])

            index1x = int(Xmin_map_temp-Xmin)
            index2x = int(index1x + np.abs(Xmax_map_temp-Xmin_map_temp))
            index1y = int(Ymin_map_temp-Ymin)
            index2y = int(index1y + np.abs(Ymax_map_temp-Ymin_map_temp))

            x_len = Xmax_map_temp-Xmin_map_temp
            y_len = Ymax_map_temp-Ymin_map_temp

            if len(I_est_flat) < (x_len+1)*(y_len+1):
                valmax = (x_len+1)*(y_len+1)
                pmax = np.amax(param)
                I_num_fin = 0.*np.arange(pmax+1, valmax)
                Q_num_fin = 0.*np.arange(pmax+1, valmax)
                U_num_fin = 0.*np.arange(pmax+1, valmax)
                N_hits_fin = 0.*np.arange(pmax+1, valmax)
                c_fin = 0.*np.arange(pmax+1, valmax)
                c2_fin = 0.*np.arange(pmax+1, valmax)
                s_fin = 0.*np.arange(pmax+1, valmax)
                s2_fin = 0.*np.arange(pmax+1, valmax)
                m_fin = 0.*np.arange(pmax+1, valmax)
                
                I_est_flat = np.append(I_est_flat, I_num_fin)
                Q_est_flat = np.append(Q_est_flat, Q_num_fin)
                U_est_flat = np.append(U_est_flat, U_num_fin)
                N_hits_flat = np.append(N_hits_flat, N_hits_fin)
                c_flat = np.append(c_flat, c_fin)
                c2_flat = np.append(c2_flat, c2_fin)
                s_flat = np.append(s_flat, s_fin)
                s2_flat = np.append(s2_flat, s2_fin)
                m_flat = np.append(m_flat, m_fin)

            I_est = np.reshape(I_est_flat, (y_len+1,x_len+1))
            Q_est = np.reshape(Q_est_flat, (y_len+1,x_len+1))
            U_est = np.reshape(U_est_flat, (y_len+1,x_len+1))
            N_hits_est = np.reshape(N_hits_flat, (y_len+1,x_len+1))
            c_est = np.reshape(c_flat, (y_len+1,x_len+1))
            c2_est = np.reshape(c2_flat, (y_len+1,x_len+1))
            s_est = np.reshape(s_flat, (y_len+1,x_len+1))
            s2_est = np.reshape(s2_flat, (y_len+1,x_len+1))
            m_est = np.reshape(m_flat, (y_len+1,x_len+1))


            finalmap_I_est[index1y:index2y+1,index1x:index2x+1] += I_est
            finalmap_Q_est[index1y:index2y+1,index1x:index2x+1] += Q_est
            finalmap_U_est[index1y:index2y+1,index1x:index2x+1] += U_est
            finalmap_N_hits[index1y:index2y+1,index1x:index2x+1] += N_hits_est
            finalmap_c[index1y:index2y+1,index1x:index2x+1] += c_est
            finalmap_c2[index1y:index2y+1,index1x:index2x+1] += c2_est
            finalmap_s[index1y:index2y+1,index1x:index2x+1] += s_est
            finalmap_s2[index1y:index2y+1,index1x:index2x+1] += s2_est
            finalmap_m[index1y:index2y+1,index1x:index2x+1] += m_est


        Delta = (c_flat**2*(c2_flat-N_hits_flat)+2*s_flat*c_flat*m_flat-c2_flat*s_flat**2-\
                 N_hits_flat*(c2_flat**2+m_flat**2-c2_flat*N_hits_flat))
        A = -(c2_flat**2+m_flat**2-c2_flat*N_hits_flat)
        B = c_flat*(c2_flat-N_hits_flat)+s_flat*m_flat
        C = c_flat*m_flat-s_flat*c2_flat
        D = -((c2_flat-N_hits_flat)*N_hits_flat+s_flat**2)
        E = c_flat*s_flat-m_flat*N_hits_flat
        F = c2_flat*N_hits_flat-c_flat**2

        index, = np.where(np.abs(Delta)>0.)
        print('INDEX', i, index, np.amin(Delta[index]), np.amax(Delta[index]))
        finalmap_I[index] = (A[index]*finalmap_I_est[index]+B[index]*finalmap_Q_est[index]+\
                             C[index]*finalmap_U_est[index])/Delta[index]
        finalmap_Q[index] = (B[index]*finalmap_I_est[index]+D[index]*finalmap_Q_est[index]+\
                             E[index]*finalmap_U_est[index])/Delta[index]
        finalmap_U[index] = (C[index]*finalmap_I_est[index]+E[index]*finalmap_Q_est[index]+\
                             F[index]*finalmap_U_est[index])/Delta[index]

        return finalmap_I, finalmap_Q, finalmap_U


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

    