import numpy as np
from astropy import wcs

class conversion(object):

    '''
    class to handle convesion between different coodinates sytem 
    '''

    def __init__(self, coord1, coord2, lst = None, lat = None):

        self.coord1 = coord1              #Array of coord 1 (if RA needs to be in hours)
        self.coord2 = np.radians(coord2)  #Array of coord 2 converted in radians   
        self.lst = lst                    #Local Sideral Time in hours
        self.lat = np.radians(lat)        #Latitude converted in radians

    def ra2ha(self):

        '''
        Return the hour angle given the lst and the ra
        ''' 

        return self.lst-self.coord1

    def ha2ra(self, hour_angle):

        return self.lst - hour_angle

    def radec2azel(self):

        '''
        Function to convert RA and DEC to AZ and EL
        '''

        hour_angle = np.radians(self.ra2ha()*15.)

        el = np.arcsin(np.sin(self.coord2)*np.sin(self.lat)+\
                       np.cos(self.lat)*np.cos(self.coord2)*np.cos(hour_angle))

        x = -np.sin(self.lat)*np.cos(self.coord2)*np.cos(hour_angle) + np.cos(self.lat)*np.sin(self.coord2)
        y = np.cos(self.coord2)*np.sin(hour_angle)

        az = -np.arctan2(x, y)
        if isinstance(az, np.ndarray):
            index, = np.where(az<0)
            az[index] += 2*np.pi
        else:
            if az <= 0:
                az += 2*np.pi
        return np.degrees(az), np.degrees(el)

    def azel2radec(self):

        '''
        Function to convert AZ and EL to RA and DEC
        '''

        dec = np.arcsin(np.sin(self.coord2)*np.sin(self.lat)-\
                        np.cos(self.lat)*np.cos(self.coord2)*np.cos(np.radians(self.coord1)))

        x = np.sin(self.lat)*np.cos(self.coord2)*np.cos(np.radians(self.coord1)) +\
            np.cos(self.lat)*np.sin(self.coord2)

        y = np.cos(self.coord2)*np.sin(np.radians(self.coord1))

        hour_angle = np.degrees(np.arctan2(x, y))/15.

        ra = self.ha2ra(hour_angle)*15.

        return ra, np.degrees(dec)

class apply_offset(object):

    '''
    Class to apply the offset to different coordinates
    '''

    def __init__(self, coord1, coord2, ctype, xsc_offset, det_offset = np.array([0.,0.]),\
                 lst = None, lat = None):

        self.coord1 = coord1                    #Array of coordinate 1
        self.coord2 = coord2                    #Array of coordinate 2
        self.ctype = ctype                      #Ctype of the map
        self.xsc_offset = xsc_offset            #Offset with respect to star cameras in xEL and EL
        self.det_offset = det_offset            #Offset with respect to the central detector in xEL and EL
        self.lst = lst                          #Local Sideral Time array
        self.lat = lat                          #Latitude array

    def correction(self):

        if self.ctype.lower() == 'ra and dec':

            conv2azel = conversion(self.coord1, self.coord2, self.lst, self.lat)

            az, el = conv2azel.radec2azel()

            xEL = az*np.cos(np.radians(el))

            xEL_corrected = xEL+self.xsc_offset[0]+self.det_offset[0]
            EL_corrected = el+self.xsc_offset[1]+self.det_offset[1]

            conv2radec = conversion(xEL_corrected/np.cos(np.radians(EL_corrected)), EL_corrected, \
                                    self.lst, self.lat)

            ra_corrected, dec_corrected = conv2radec.azel2radec()

            return ra_corrected, dec_corrected

        elif self.cype.lower() == 'az and el':

            
            el_corrected = self.coord2+self.xsc_offset[1]+self.det_offset[1]

            az_corrected = (self.coord1*np.cos(self.coord2)+self.xsc_offset[0]+\
                            self.det_offset[0])/np.cos(el_corrected)

            return az_corrected, el_corrected

        else:

            return (self.coord1+self.xsc_offset[0]+self.det_offset[0], \
                    self.coord2+self.xsc_offset[1]+self.det_offset[1])


class compute_offset(object):

    def __init__(self, coord1_ref, coord2_ref, map_data, \
                 pixel1_coord, pixel2_coord, wcs_trans, ctype, \
                 lst, lat):

        self.coord1_ref = coord1_ref           #Reference value of the map along the x axis in RA and DEC
        self.coord2_ref = coord2_ref           #Reference value of the map along the y axis in RA and DEC
        self.map_data = map_data               #Maps 
        self.pixel1_coord = pixel1_coord       #Array of the coordinates converted in pixel along the x axis
        self.pixel2_coord = pixel2_coord       #Array of the coordinates converted in pixel along the y axis
        self.wcs_trans = wcs_trans             #WCS transformation 
        self.ctype = ctype                     #Ctype of the map
        self.lst = lst                         #Local Sideral Time
        self.lat = lat                         #Latitude

    def centroid(self, threshold=0.275):

        '''
        For more information about centroid calculation see Shariff, PhD Thesis, 2016
        '''

        maxval = np.max(self.map_data)
        minval = np.min(self.map_data)
        y_max, x_max = np.where(self.map_data == maxval)

        lt_inds = np.where(self.map_data < threshold*maxval)
        gt_inds = np.where(self.map_data > threshold*maxval)

        weight = np.zeros((self.map_data.shape[0], self.map_data.shape[1]))
        weight[gt_inds] = 1.
        a = self.map_data[gt_inds]
        flux = np.sum(a)

        x_coord_max = np.floor(np.amax(self.pixel1_coord))+1
        x_coord_min = np.floor(np.amin(self.pixel1_coord))

        x_arr = np.arange(x_coord_min, x_coord_max)

        y_coord_max = np.floor(np.amax(self.pixel2_coord))+1
        y_coord_min = np.floor(np.amin(self.pixel2_coord))

        y_arr = np.arange(y_coord_min, y_coord_max)

        xx, yy = np.meshgrid(x_arr, y_arr)
        
        x_c = np.sum(xx*weight*self.map_data)/flux
        y_c = np.sum(yy*weight*self.map_data)/flux

        # import matplotlib.pyplot as plt
        # plt.imshow(self.map_data, origin = 'lower', extent=[x_coord_min, x_coord_max, y_coord_min, y_coord_max])
        # plt.plot(x_c, y_c, 'x')
        # plt.show()

        return np.rint(x_c), np.rint(y_c)
    
    def value(self):

        #Centroid of the map
        x_c, y_c = self.centroid()
        map_center = wcs.utils.pixel_to_skycoord(x_c, y_c, self.wcs_trans)
        x_map = map_center.ra.hour
        y_map = map_center.dec.degree
        
        print('b1', x_c, y_c, x_map*15., y_map, np.average(self.lst), np.average(self.lat))
        if self.ctype.lower() == 'ra and dec':
            centroid_conv = conversion(x_map, y_map, np.average(self.lst), np.average(self.lat))

            coord1_reference = self.coord1_ref/15.

            az_centr, el_centr = centroid_conv.radec2azel()
            xel_centr = az_centr*np.cos(np.radians(el_centr))

        else:
            coord1_reference = self.coord1_ref
            el_centr = y_map
            if self.cytpe.lower() == 'xel and el':
                xel_centr = x_map            
            else:
                xel_centr = x_map/np.cos(np.radians(el_centr))
            

        ref_conv = conversion(coord1_reference, self.coord2_ref, np.average(self.lst), \
                              np.average(self.lat))

        az_ref, el_ref = ref_conv.radec2azel()

        xel_ref = az_ref*np.cos(np.radians(el_ref))

        print('TEST_REF')

        return xel_centr-xel_ref, el_centr-el_ref




        








        

        
        


