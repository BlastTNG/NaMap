import pygetdata as gd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class data_value():
<<<<<<< HEAD
<<<<<<< HEAD
=======
    
    '''
    Class for reading the values from a DIRFILE
    '''
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72

=======
    
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)
    def __init__(self, det_path, det_name, coord_path, coord1_name, \
                 coord2_name, det_file_type, coord1_file_type, coord2_file_type, \
                 experiment):

<<<<<<< HEAD
        self.det_path = det_path
        self.det_name = det_name
        self.coord_path = coord_path
        self.coord1_name = coord1_name
        self.coord2_name = coord2_name
        self.det_file_type = det_file_type
        self.coord1_file_type = coord1_file_type
        self.coord2_file_type = coord2_file_type
        self.experiment = experiment

    def conversion_type(self, file_type):

=======
        self.det_path = det_path                    #Path of the detector dirfile
        self.det_name = det_name                    #Detector name to be analyzed
        self.coord_path = coord_path                #Path of the coordinates dirfile
        self.coord1_name = coord1_name              #Coordinates 1 name, e.g. RA or AZ
        self.coord2_name = coord2_name              #Coordinates 2 name
        self.det_file_type = det_file_type          #Detector DIRFILE datatype
        self.coord1_file_type = coord1_file_type    #Coordinate 1 DIRFILE datatype
        self.coord2_file_type = coord2_file_type    #Coordinate 2 DIRFILE datatype
        self.experiment = experiment                #Experiment to be analyzed

    def conversion_type(self, file_type):

        '''
        Function to define the different datatype conversions strings for pygetdata
        '''

>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
        if file_type == 'u16':
            gdtype = gd.UINT16
        elif file_type == 'u32':
            gdtype = gd.UINT32
        elif file_type == 's32':
            gdtype = gd.INT32
        elif file_type == 'float':
            gdtype = gd.FLOAT32

        return gdtype 

    def load(self, filepath, file, file_type):
<<<<<<< HEAD
        if np.size(file) == 1: 
            d = gd.dirfile(filepath, gd.RDONLY)
            vectors = d.field_list()
            for i in range (len(vectors)):
<<<<<<< HEAD
                if vectors[i] == file:
=======

        '''
        Return the values of the DIRFILE as a numpy array
        
        filepath: path of the DIRFILE to be read
        file: name of the value to be read from the dirfile, e.g. detector name or
              coordinate name
        file_type: data type conversion string for the DIRFILE data
        '''
        if np.size(file) == 1: 
            d = gd.dirfile(filepath, gd.RDONLY)
            vectors = d.field_list()
            for i in range (len(vectors)):
                if str(vectors[i])[2:-1] == file:
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
                if str(vectors[i])[2:-1] == file:
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)
                    gdtype = self.conversion_type(file_type)
                    if self.experiment.lower()=='blast-tng':
                        num = d.eof('MCP_1HZ_FRAMECOUNT')
                    else:
                        num = d.nframes
                    
<<<<<<< HEAD
<<<<<<< HEAD
                    values = d.getdata(file, num_frames = num-1, first_frame=0)
=======
                    values = d.getdata(file, gdtype, num_frames = num-1, first_frame=0)
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
                    values = d.getdata(file, gdtype, num_frames = num-1, first_frame=0)
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)
            return np.asarray(values)
        else:
            d = gd.dirfile(filepath[0], gd.RDWR|gd.UNENCODED)
            vectors = d.field_list()
            len_det = len(d.getdata(vectors[detlist[0]], gd.UINT16, num_frames = d.nframes))
            values = np.zeros((len(filepath), len_det))

            for i in range(len(filepath)):
                d = gd.dirfile(filepath[i], gd.RDWR|gd.UNENCODED)
                vectors = d.field_list()
                for j in range(len(vectors)):
                    if vectors[j] == file[i]:
                        values[i,:] = np.asarray(d.getdata(vectors[file[i]], \
                                                 gdtype_det,num_frames = d.nframes))
                
            return values

    def values(self):

<<<<<<< HEAD
        det_data = self.load(self.det_path, self.det_name, self.det_file_type)
        coord2_data = self.load(self.coord_path, self.coord2_name.lower(), self.coord2_file_type)

        print('det')
        det_data = self.load(self.det_path, self.det_name, self.det_file_type)
        print('coord2', self.coord2_name)
        coord2_data = self.load(self.coord_path, self.coord2_name.lower(), self.coord2_file_type)
        print('coord1', self.coord1_name)

        if self.coord1_name.lower() == 'cross-el':
            coord1_data = self.load(self.coord_path, 'az', self.coord1_file_type)
            coord1_data = coord1_data*np.cos(coord2_data)
        elif self.coord1_name.lower() == 'ra':
            coord1_data = self.load(self.coord_path, self.coord1_name.lower(), self.coord1_file_type)
            coord1_data = coord1_data*15.
        else:
            coord1_data = self.load(self.coord_path, self.coord1_name.lower(), self.coord1_file_type)
=======
        '''
        Function to return the timestreams for detector and coordinates
        '''

        det_data = self.load(self.det_path, self.det_name, self.det_file_type)
        coord2_data = self.load(self.coord_path, self.coord2_name.lower(), self.coord2_file_type)

        if self.coord1_name.lower() == 'ra':
            coord1_data = self.load(self.coord_path, self.coord1_name.lower(), self.coord1_file_type)
        else:
            coord1_data = self.load(self.coord_path, 'az', self.coord1_file_type)
            
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
        return det_data, coord1_data, coord2_data

class convert_dirfile():

<<<<<<< HEAD
    def __init__(self, data, param1, param2):

        self.data = data
        self.param1 = param1
        self.param2 = param2

    def conversion(self):

        return self.param1*self.data+self.param2

class frame_zoom_sync():

    def __init__(self, det_data, det_fs, det_sample_frame,\
                 coord1_data, coord2_data, coord_fs, coord_sample_frame, \
                 frame1, frame2, experiment, roach_number=None, roach_pps_path=None):

        self.det_data = det_data
        self.det_fs = float(det_fs)
        self.det_sample_frame = int(float(det_sample_frame))
        self.coord1_data = coord1_data
        self.coord_fs = float(coord_fs)
        self.coord_sample_frame = int(float(coord_sample_frame))
        self.coord2_data = coord2_data
        self.frame1 = int(float(frame1))
        self.frame2 = int(float(frame2))
        self.experiment = experiment
        if roach_number is not None:
            self.roach_number = int(float(roach_number))
        else:
            self.roach_number = roach_number
        self.roach_pps_path = roach_pps_path

    def frame_zoom(self, data, sample_frame, fs, fps):

=======
    '''
    Class for converting TODs from dirfile value to real value, 
    considering a linear conversion
    '''

    def __init__(self, data, param1, param2):

        self.data = data        #DIRFILE TOD
        self.param1 = param1    #gradient of the conversion
        self.param2 = param2    #intercept of the conversion

    def conversion(self):

        self.data = self.param1*self.data+self.param2

class frame_zoom_sync():

    '''
    This class is designed to extract the frames of interest from the complete timestream and 
    sync detector and coordinates timestream given a different sampling of the two
    '''

    def __init__(self, det_data, det_fs, det_sample_frame,\
                 coord1_data, coord2_data, coord_fs, coord_sample_frame, \
                 startframe, endframe, experiment, roach_number=None, roach_pps_path=None):

        self.det_data = det_data                                #Detector data timestream
        self.det_fs = float(det_fs)                             #Detector frequency sampling
        self.det_sample_frame = int(float(det_sample_frame))    #Detector samples in each frame of the timestream
        self.coord1_data = coord1_data                          #Coordinate 1 data timestream
        self.coord_fs = float(coord_fs)                         #Coordinates frequency sampling
        self.coord_sample_frame = int(float(coord_sample_frame))#Coordinates samples in each frame of the time stream
        self.coord2_data = coord2_data                          #Coordinate 2 data timestream
        self.startframe = int(float(startframe))                #Start frame
        self.endframe = int(float(endframe))                    #End frame
        self.experiment = experiment                            #Experiment to be analyzed, right now BLASTPol or BLAST-TNG
        if roach_number is not None:
            self.roach_number = int(float(roach_number))        #If BLAST-TNG is the experiment, this gives the number of the roach used to read the detector
        else:
            self.roach_number = roach_number
        self.roach_pps_path = roach_pps_path                    #Pulse per second of the roach used to sync the data

    def frame_zoom(self, data, sample_frame, fs, fps):

        '''
        Selecting the frames of interest and associate a timestamp for each value
        '''

>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
        frames = fps.copy()

        print('PARAM_ZOOM')
        print(sample_frame, fs, fps)
        print(len(data))

        frames[0] = fps[0]*sample_frame
        frames[1] = fps[1]*sample_frame+1

        print(frames)

        if len(np.shape(data)) == 1:
<<<<<<< HEAD
            time = np.arange(len(data))/np.floor(fs)
            time = time[frames[0]:frames[1]]

=======
            time = (np.arange(np.diff(frames))+frames[0])/np.floor(fs)
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
            return time, data[frames[0]:frames[1]]
        else:
            time = np.arange(len(data[0, :]))/np.floor(fs)
            time = time[frames[0]:frames[1]]
            return  time, data[:,frames[0]:frames[1]]

    def det_time(self):
<<<<<<< HEAD
=======

        '''
        This function is specific for BLAST-TNG only and creates a time array using the PPS data
        '''
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
        
        d = gd.dirfile(self.roach_pps_path, gd.RDONLY)
        string = 'pps_count_roach'+str(int(self.roach_number))
        
        data = d.getdata(string, gd.UINT32, num_frames = d.nframes-1)

        tmin = np.amin(data)
        tmax = np.amax(data)

        time = np.array([])
        j = 0
        t0 = float(tmin)
        
        index, = np.where(data==tmin)
        index_switch = index[0]
        
        if index_switch == 0:
            bins_temp = np.bincount(data)
            bins = bins_temp[int(tmin):].copy()
            for i in range(tmin, tmax+1, 1):           
                if bins[j] != 0:
                    temp = np.arange(t0, t0+1.,1/float(bins[j]))
                    time = np.append(time, temp)
                t0 += 1.
                j += 1
    
        else:
            bins_temp = np.bincount(data[:index_switch])
            index_bins, = np.where(bins_temp != 0)
            
            
            for i in range(len(index_bins)):
                temp = index_bins[i]+np.arange(0., bins_temp[index_bins[i]], \
                                               1.)/bins_temp[index_bins[i]]
                    
                time = np.append(time, temp)
            
            bins_temp = np.bincount(data[index_switch:])
            index_bins, = np.where(bins_temp != 0)
            
            for i in range(len(index_bins)):
<<<<<<< HEAD
<<<<<<< HEAD
                 temp = index_bins[i]+np.arange(0., bins_temp[index_bins[i]], \
                                                1.)/bins_temp[index_bins[i]] 
                    
                 time = np.append(time, temp)
=======
                temp = index_bins[i]+np.arange(0., bins_temp[index_bins[i]], \
                                               1.)/bins_temp[index_bins[i]] 
                    
                time = np.append(time, temp)
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
                temp = index_bins[i]+np.arange(0., bins_temp[index_bins[i]], \
                                               1.)/bins_temp[index_bins[i]] 
                    
                time = np.append(time, temp)
>>>>>>> 4ee3dbe... Fixed bug in selecting data

        return time

    def coord_int(self, coord1, coord2, time_acs, time_det):

<<<<<<< HEAD
=======
        '''
        Interpolates the coordinates values to compensate for the smaller frequency sampling
        '''

>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
        coord1_int = interp1d(time_acs, coord1, kind='linear')
        coord2_int = interp1d(time_acs, coord2, kind= 'linear')

        return coord1_int(time_det), coord2_int(time_det)

    def sync_data(self):
<<<<<<< HEAD
        if self.experiment.lower() == 'blast-tng':
            dettime = self.det_time()

            frames0 = self.frame1*self.det_sample_frame
            frames1 = self.frame2*self.det_sample_frame+1

            detTOD = self.det_data.copy()[frames[0]:frames[1]]
        elif self.experiment.lower() == 'blastpol':
            dettime, detTOD = self.frame_zoom(self.det_data, self.det_sample_frame, \
                                              self.det_fs, np.array([self.frame1,self.frame2]))

        # print(dettime)
        coord1time, coord1 = self.frame_zoom(self.coord1_data, self.coord_sample_frame, \
                                             self.coord_fs, np.array([self.frame1,self.frame2]))

        coord2time, coord2 = self.frame_zoom(self.coord2_data, self.coord_sample_frame, \
<<<<<<< HEAD
                                          self.coord_fs, np.array([self.frame1,self.frame2]))
=======

        '''
        Wrapper for the previous functions to return the slices of the detector and coordinates TODs,  
        and the associated time
        '''

        if self.experiment.lower() == 'blast-tng':
            
            sframe = self.startframe*self.det_sample_frame
            eframe = self.endframe*self.det_sample_frame+1
            
            dettime = self.det_time()[sframe:eframe]
            self.det_data = self.det_data[sframe:eframe]

        elif self.experiment.lower() == 'blastpol':
            dettime, self.det_data = self.frame_zoom(self.det_data, self.det_sample_frame, \
                                                     self.det_fs, np.array([self.startframe,self.endframe]))

        coord1time, coord1 = self.frame_zoom(self.coord1_data, self.coord_sample_frame, \
                                             self.coord_fs, np.array([self.startframe,self.endframe]))

        coord2time, coord2 = self.frame_zoom(self.coord2_data, self.coord_sample_frame, \
                                             self.coord_fs, np.array([self.startframe,self.endframe]))
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72

=======
                                             self.coord_fs, np.array([self.frame1,self.frame2]))
        # print('COORD')
        # print(self.coord1_data)
        # print(self.coord2_data)
        # print(coord1time)
>>>>>>> 4ee3dbe... Fixed bug in selecting data
        index1, = np.where(np.abs(dettime-coord1time[0]) == np.amin(np.abs(dettime-coord1time[0])))
        index2, = np.where(np.abs(dettime-coord1time[-1]) == np.amin(np.abs(dettime-coord1time[-1])))

        coord1_inter, coord2_inter = self.coord_int(coord1, coord2, \
                                                    coord1time, dettime[index1[0]+10:index2[0]-10])
        
<<<<<<< HEAD
<<<<<<< HEAD
        detTOD = self.det_data.copy()

        return dettime[index1[0]+10:index2[0]-10], detTOD[index1[0]+10:index2[0]-10], \
               coord1_inter, coord2_inter
=======
        del coord1time
        del coord2time
        del coord1
        del coord2

        return (dettime[index1[0]+10:index2[0]-10], self.det_data[index1[0]+10:index2[0]-10], \
                coord1_inter, coord2_inter)
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
        
        # print('INDICES')
        # print(index1[0],index2[0])
        # plt.plot(detTOD[index1[0]:index2[0]])
        # plt.show()
        return (dettime[index1[0]+10:index2[0]-10], detTOD[index1[0]+10:index2[0]-10], \
                coord1_inter, coord2_inter)
>>>>>>> 4ee3dbe... Fixed bug in selecting data

