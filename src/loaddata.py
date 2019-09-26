import pygetdata as gd
import numpy as np
from scipy.interpolate import interp1d
import os
import astropy.table as tb

class data_value():
<<<<<<< HEAD
<<<<<<< HEAD
=======
    
    '''
    Class for reading the values of the TODs (detectors and coordinates) from a DIRFILE
    '''
<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72

=======
    
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)
=======

>>>>>>> 651e1e6... Commented files
    def __init__(self, det_path, det_name, coord_path, coord1_name, \
                 coord2_name, det_file_type, coord1_file_type, coord2_file_type, \
<<<<<<< HEAD
<<<<<<< HEAD
                 experiment):
<<<<<<< HEAD

<<<<<<< HEAD
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

=======
=======
>>>>>>> 8b07277... IQ -> power libraries
=======
                 experiment, lst_file_type, lat_file_type):
>>>>>>> 3f224e8... Added pointing input dialogs and caluclation
=======
                 experiment, lst_file_type, lat_file_type, hwp_file_type):
>>>>>>> d11dfe9... Solved pointing, multidetectors stacking and loading bugs
        self.det_path = det_path                    #Path of the detector dirfile
        self.det_name = det_name                    #Detector name to be analyzed
        self.coord_path = coord_path                #Path of the coordinates dirfile
        self.coord1_name = coord1_name              #Coordinates 1 name, e.g. RA or AZ
        self.coord2_name = coord2_name              #Coordinates 2 name
        self.det_file_type = det_file_type          #Detector DIRFILE datatype
        self.coord1_file_type = coord1_file_type    #Coordinate 1 DIRFILE datatype
        self.coord2_file_type = coord2_file_type    #Coordinate 2 DIRFILE datatype
        self.experiment = experiment                #Experiment to be analyzed

        self.lst_file_type = lst_file_type          #LST DIRFILE datatype
        self.lat_file_type = lat_file_type          #LAT DIRFILE datatype

        self.hwp_file_type = hwp_file_type          #HWP DIRFILE datatype

    def conversion_type(self, file_type):

>>>>>>> 651e1e6... Commented files
        '''
        Function to define the different datatype conversions strings for pygetdata
        '''

<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> 651e1e6... Commented files
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
<<<<<<< HEAD
        if np.size(file) == 1: 
            d = gd.dirfile(filepath, gd.RDONLY)
            vectors = d.field_list()
            for i in range (len(vectors)):
<<<<<<< HEAD
                if vectors[i] == file:
=======
=======
>>>>>>> 651e1e6... Commented files

        '''
        Return the values of the DIRFILE as a numpy array
        
        filepath: path of the DIRFILE to be read
        file: name of the value to be read from the dirfile, e.g. detector name or
              coordinate name
        file_type: data type conversion string for the DIRFILE data
        '''
        if np.size(file) == 1: 
            d = gd.dirfile(filepath, gd.RDONLY)
<<<<<<< HEAD
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
=======
            gdtype = self.conversion_type(file_type)
            if self.experiment.lower()=='blast-tng':
                num = d.eof('MCP_1HZ_FRAMECOUNT')
            else:
                num = d.nframes
            if isinstance(file, str):
                values = d.getdata(file, gdtype, num_frames = num-1, first_frame=0)
            else:
                values = d.getdata(file[0], gdtype, num_frames = num-1, first_frame=0)
>>>>>>> db74452... Solved a bug on applying the offset
            return np.asarray(values)
        else:
            d = gd.dirfile(filepath, gd.RDWR|gd.UNENCODED)
            gdtype = self.conversion_type(file_type)
            values = np.array([])

            for i in range(len(file)):
                if i == 0:
                    values = d.getdata(file[i], gdtype,num_frames = d.nframes)
                else:
                    values = np.vstack((values, d.getdata(file[i], gdtype,num_frames = d.nframes)))
            return values

    def values(self):

<<<<<<< HEAD
<<<<<<< HEAD
        det_data = self.load(self.det_path, self.det_name, self.det_file_type)
        coord2_data = self.load(self.coord_path, self.coord2_name.lower(), self.coord2_file_type)
=======
        '''
        Function to return the timestreams for detector and coordinates
        '''
<<<<<<< HEAD
>>>>>>> 651e1e6... Commented files

        det_data = self.load(self.det_path, self.det_name, self.det_file_type)
=======
        if self.experiment.lower() == 'blast-tng':
            # Temporary function 
            det_data = np.loadtxt(self.det_path+self.det_name)
        else:
            det_data = self.load(self.det_path, self.det_name, self.det_file_type)
>>>>>>> 8b07277... IQ -> power libraries
        coord2_data = self.load(self.coord_path, self.coord2_name.lower(), self.coord2_file_type)

        if self.coord1_name.lower() == 'ra':
            coord1_data = self.load(self.coord_path, self.coord1_name.lower(), self.coord1_file_type)
        else:
<<<<<<< HEAD
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

        if self.hwp_file_type is not None:
            hwp_data = self.load(self.coord_path, 'pot_hwpr', self.hwp_file_type)
        else:
            hwp_data = 0.
        
        if self.lat_file_type is not None and self.lat_file_type is not None:
            
<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
            coord1_data = self.load(self.coord_path, 'az', self.coord1_file_type)
            
>>>>>>> 8989c24... Correct calculation of coordinates
        return det_data, coord1_data, coord2_data
=======
            lat = self.load(self.coord_path, 'lat', self.lat_file_type)
            lst = self.load(self.coord_path, 'lst', self.lst_file_type)

            return det_data, coord1_data, coord2_data, hwp_data, lst, lat
        
        else:
            return det_data, coord1_data, coord2_data, hwp_data

<<<<<<< HEAD
>>>>>>> 3f224e8... Added pointing input dialogs and caluclation

=======
>>>>>>> 63d0b03... Added pointing offset calculation
class convert_dirfile():

<<<<<<< HEAD
<<<<<<< HEAD
=======
    '''
    Class for converting TODs from dirfile value to real value, 
    considering a linear conversion
    '''

>>>>>>> 651e1e6... Commented files
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
                 startframe, endframe, experiment, \
                 lst_data, lat_data, lstlatfreq, lstlat_sample_frame, \
                 offset =None, roach_number=None, roach_pps_path=None, hwp_data=0., \
                 hwp_fs=None, hwp_sample_frame=None):

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
        self.lst_data = lst_data                                #LST timestream (if correction is required and coordinates are RA-DEC)
        self.lat_data = lat_data                                #LAT timestream (if correction is required and coordinates are RA-DEC)
        self.lstlatfreq = lstlatfreq                            #LST-LAT sampling frequency (if correction is required and coordinates are RA-DEC)
        self.lstlat_sample_frame = lstlat_sample_frame          #LST-LAT samples per frame (if correction is required and coordinates are RA-DEC)
        if roach_number is not None:
            self.roach_number = int(float(roach_number))        #If BLAST-TNG is the experiment, this gives the number of the roach used to read the detector
        else:
            self.roach_number = roach_number
        self.roach_pps_path = roach_pps_path                    #Pulse per second of the roach used to sync the data
        self.offset = offset                                    #Time offset between detector data and coordinates
        self.hwp_data = hwp_data
        self.hwp_fs = float(hwp_fs)
        self.hwp_sample_frame = float(hwp_sample_frame)

<<<<<<< HEAD
    def frame_zoom(self, data, sample_frame, fs, fps):
<<<<<<< HEAD
<<<<<<< HEAD

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
=======
>>>>>>> 651e1e6... Commented files
=======
    def frame_zoom(self, data, sample_frame, fs, fps, offset = None):
>>>>>>> 53b90cc... Correct sync with offset

        '''
        Selecting the frames of interest and associate a timestamp for each value
        '''
<<<<<<< HEAD

<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> fa75c3a... Reduced memory consumption
=======
>>>>>>> 651e1e6... Commented files
=======
>>>>>>> d11dfe9... Solved pointing, multidetectors stacking and loading bugs
        frames = fps.copy()

        frames[0] = fps[0]*sample_frame
        frames[1] = fps[1]*sample_frame+1

<<<<<<< HEAD
        if len(np.shape(data)) == 1:
<<<<<<< HEAD
<<<<<<< HEAD
            time = np.arange(len(data))/np.floor(fs)
            time = time[frames[0]:frames[1]]

=======
            time = (np.arange(np.diff(frames))+frames[0])/np.floor(fs)
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
            time = (np.arange(np.diff(frames))+frames[0])/np.floor(fs)
<<<<<<< HEAD
>>>>>>> fa75c3a... Reduced memory consumption
            return time, data[frames[0]:frames[1]]
=======
            if offset is not None:
=======
        if offset is not None:
>>>>>>> db74452... Solved a bug on applying the offset
                delay = offset*np.floor(fs)/1000.
                frames = frames.astype(float)+delay

        if len(np.shape(data)) == 1:
            time = (np.arange(np.diff(frames))+frames[0])/np.floor(fs)
            return time, data[int(frames[0]):int(frames[1])]
>>>>>>> 53b90cc... Correct sync with offset
        else:
            print('TEST', len(data[0, :]))
            time = np.arange(len(data[0, :]))/np.floor(fs)
            time = time[int(frames[0]):int(frames[1])]
            print('OK')
            return  time, data[:,int(frames[0]):int(frames[1])]

    def det_time(self):
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> 651e1e6... Commented files

        '''
        This function is specific for BLAST-TNG only and creates a time array using the PPS data
        '''
<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> 651e1e6... Commented files
        
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
<<<<<<< HEAD
=======
=======
>>>>>>> 651e1e6... Commented files
        '''
        Interpolates the coordinates values to compensate for the smaller frequency sampling
        '''

<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> 651e1e6... Commented files
        coord1_int = interp1d(time_acs, coord1, kind='linear')
        coord2_int = interp1d(time_acs, coord2, kind= 'linear')

        return coord1_int(time_det), coord2_int(time_det)

    def sync_data(self):
<<<<<<< HEAD
<<<<<<< HEAD
        if self.experiment.lower() == 'blast-tng':
            dettime = self.det_time()
=======
>>>>>>> 651e1e6... Commented files

        '''
        Wrapper for the previous functions to return the slices of the detector and coordinates TODs,  
        and the associated time
        '''

        if self.experiment.lower() == 'blast-tng':
            
            sframe = self.startframe*self.det_sample_frame
            eframe = self.endframe*self.det_sample_frame+1
            all_time = self.det_time().copy()

            dettime = all_time[sframe:eframe]
            self.det_data = self.det_data[sframe:eframe]

        elif self.experiment.lower() == 'blastpol':
            print('data', self.det_data)
            dettime, self.det_data = self.frame_zoom(self.det_data, self.det_sample_frame, \
                                                     self.det_fs, np.array([self.startframe,self.endframe]), \
                                                     self.offset)

        print('TEST_2')
        coord1time, coord1 = self.frame_zoom(self.coord1_data, self.coord_sample_frame, \
                                             self.coord_fs, np.array([self.startframe,self.endframe]))

        coord2time, coord2 = self.frame_zoom(self.coord2_data, self.coord_sample_frame, \
<<<<<<< HEAD
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

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
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
=======
                                             self.coord_fs, np.array([self.startframe,self.endframe]))

>>>>>>> 651e1e6... Commented files
=======
        if self.offset is not None:
            dettime = dettime - self.offset/1000.
=======
        # if self.offset is not None:
        #     print('OFFSET')
        #     dettime = dettime - self.offset/1000.
        #     print(np.diff(dettime))
>>>>>>> 53b90cc... Correct sync with offset

<<<<<<< HEAD
>>>>>>> 77760bc... Add TOD timing offset
=======
        print('COORD Time',coord1time-coord1time[0])
        print('DETTIME', dettime-dettime[0])
        print('length dettime', len(dettime))
        print('len coordtime', len(coord1time))
        dettime = dettime-dettime[0]
        coord1time = coord1time-coord1time[0]
>>>>>>> 8b07277... IQ -> power libraries
=======
        if self.offset is not None:
            dettime = dettime - self.offset/1000.

=======
>>>>>>> db74452... Solved a bug on applying the offset
        dettime = dettime-dettime[0]
        coord1time = coord1time-coord1time[0]
>>>>>>> b96fc79... Added wcs projection to maps
        index1, = np.where(np.abs(dettime-coord1time[0]) == np.amin(np.abs(dettime-coord1time[0])))
        index2, = np.where(np.abs(dettime-coord1time[-1]) == np.amin(np.abs(dettime-coord1time[-1])))

        coord1_inter, coord2_inter = self.coord_int(coord1, coord2, \
                                                    coord1time, dettime[index1[0]+10:index2[0]-10])
<<<<<<< HEAD
<<<<<<< HEAD
        
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        detTOD = self.det_data.copy()

        return dettime[index1[0]+10:index2[0]-10], detTOD[index1[0]+10:index2[0]-10], \
               coord1_inter, coord2_inter
=======
=======
>>>>>>> 6acdf4e... Solved a memory leak when trying to replot with different parameters
=======

>>>>>>> 3f224e8... Added pointing input dialogs and caluclation
=======

        if isinstance(self.hwp_data, np.ndarray):
            print('HWP_data', self.hwp_data)
            print(self.hwp_sample_frame, type(self.hwp_sample_frame))
            print(self.hwp_fs, type(self.hwp_fs))
            hwptime, hwp = self.frame_zoom(self.hwp_data, self.hwp_sample_frame, \
                                           self.hwp_fs, np.array([self.startframe,self.endframe]))
            
            hwptime = hwptime - hwptime[0]
            index1, = np.where(np.abs(dettime-hwptime[0]) == np.amin(np.abs(dettime-hwptime[0])))
            index2, = np.where(np.abs(dettime-hwptime[-1]) == np.amin(np.abs(dettime-hwptime[-1])))

            hwp_interpolation = interp1d(hwptime, hwp, kind='linear')
            hwp_inter = hwp_interpolation(dettime[index1[0]+10:index2[0]-10])

            del hwptime
            del hwp

        else:

            hwp_inter = np.zeros_like(coord1_inter)


>>>>>>> d11dfe9... Solved pointing, multidetectors stacking and loading bugs
        del coord1time
        del coord2time
        del coord1
        del coord2
<<<<<<< HEAD
<<<<<<< HEAD

<<<<<<< HEAD
<<<<<<< HEAD
        return (dettime[index1[0]+10:index2[0]-10], self.det_data[index1[0]+10:index2[0]-10], \
                coord1_inter, coord2_inter)
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
=======

        if self.lat_data is not None and self.lat_data is not None:
            print('LST/LAT')
            lsttime, lst = self.frame_zoom(self.lst_data, self.lstlat_sample_frame, \
                                           self.lstlatfreq, np.array([self.startframe,self.endframe]))

            lattime, lat = self.frame_zoom(self.lat_data, self.lstlat_sample_frame, \
                                           self.lstlatfreq, np.array([self.startframe,self.endframe]))

            lsttime = lsttime-lsttime[0]
            index1, = np.where(np.abs(dettime-lsttime[0]) == np.amin(np.abs(dettime-lsttime[0])))
            index2, = np.where(np.abs(dettime-lsttime[-1]) == np.amin(np.abs(dettime-lsttime[-1])))

            lst_inter, lat_inter = self.coord_int(lst, lat, \
                                                  lsttime, dettime[index1[0]+10:index2[0]-10])

            del lst
            del lat

            if np.size(np.shape(self.det_data)) > 1:
                return (dettime[index1[0]+10:index2[0]-10], self.det_data[:,index1[0]+10:index2[0]-10], \
                        coord1_inter, coord2_inter, hwp_inter, lst_inter, lat_inter)
            else:
                return (dettime[index1[0]+10:index2[0]-10], self.det_data[index1[0]+10:index2[0]-10], \
                        coord1_inter, coord2_inter,  hwp_inter, lst_inter, lat_inter)
        
        else:
<<<<<<< HEAD
            return (dettime[index1[0]+10:index2[0]-10], self.det_data[index1[0]+10:index2[0]-10], \
                    coord1_inter, coord2_inter)
<<<<<<< HEAD
>>>>>>> 3f224e8... Added pointing input dialogs and caluclation
        
=======
>>>>>>> 6acdf4e... Solved a memory leak when trying to replot with different parameters
        # print('INDICES')
        # print(index1[0],index2[0])
        # plt.plot(detTOD[index1[0]:index2[0]])
        # plt.show()
=======
>>>>>>> 651e1e6... Commented files
        return (dettime[index1[0]+10:index2[0]-10], self.det_data[index1[0]+10:index2[0]-10], \
                coord1_inter, coord2_inter)
>>>>>>> 4ee3dbe... Fixed bug in selecting data
=======
        return (dettime[index1[0]+10:index2[0]-10], self.det_data[index1[0]+10:index2[0]-10], \
                coord1_inter, coord2_inter)
        
<<<<<<< HEAD

>>>>>>> 8b07277... IQ -> power libraries

=======
>>>>>>> cab17d8... update repo
=======
=======
            if np.size(np.shape(self.det_data)) > 1:
                return (dettime[index1[0]+10:index2[0]-10], self.det_data[:,index1[0]+10:index2[0]-10], \
                        coord1_inter, coord2_inter, hwp_inter)
            else:
                return (dettime[index1[0]+10:index2[0]-10], self.det_data[index1[0]+10:index2[0]-10], \
<<<<<<< HEAD
                        coord1_inter, coord2_inter)
>>>>>>> db74452... Solved a bug on applying the offset
=======
                        coord1_inter, coord2_inter, hwp_inter)
>>>>>>> d11dfe9... Solved pointing, multidetectors stacking and loading bugs

class xsc_offset():
    
    '''
    class to read star camera offset files
    '''

    def __init__(self, xsc, frame1, frame2):

        self.xsc = xsc #Star Camera number
        self.frame1 = frame1 #Starting frame
        self.frame2 = frame2 #Ending frame

    def read_file(self):

        '''
        Function to read a star camera offset file and return the coordinates 
        offset
        '''

        path = os.getcwd()+'/xsc_'+str(int(self.xsc))+'.txt'

        xsc_file = np.loadtxt(path, skiprows = 2)

        index, = np.where((xsc_file[0]>=float(self.frame1)) & (xsc_file[1]<float(self.frame2)))

        if np.size(index) > 1:
            index = index[0]

        return xsc_file[2], xsc_file[3]

<<<<<<< HEAD
>>>>>>> 63d0b03... Added pointing offset calculation
=======
class det_table():

    '''
    Class to read detector tables. For BLASTPol can convert also detector names using another table
    '''

    def __init__(self, name, experiment, pathtable):

        self.name = name
        self.experiment = experiment
        self.pathtable = pathtable

    def loadtable(self):
        det_off = np.zeros((np.size(self.name), 2))
        noise = np.ones(np.size(self.name))
        grid_angle = np.zeros(np.size(self.name))
        pol_angle_offset = np.zeros(np.size(self.name))
        resp = np.zeros(np.size(self.name))

        if self.experiment.lower() == 'blastpol':

            for i in range(np.shape(det_off)[0]):
                if self.name[i][0].lower() == 'n':            
                    path = self.pathtable+'bolo_names.txt'
                    name_table = np.loadtxt(path, skiprows = 1, dtype = str)

                    index, = np.where(self.name[i].upper() == name_table[:,1])
                    real_name = name_table[index, 0]
                else:
                    real_name = self.name

                path = self.pathtable+'bolotable.tsv'
                btable = tb.Table.read(path, format='ascii.basic')
                index, = np.where(btable['Name'] == real_name[0].upper())
                det_off[i, 1] = btable['EL'][index]/3600.       #Conversion from arcsec to degrees
                det_off[i, 0] = btable['XEL'][index]/3600.     #Conversion from arcsec to degrees          

                noise[i] = btable['WhiteNoise'][index]
                grid_angle[i] = btable['Angle'][index]
                pol_angle_offset[i] = btable['Chi'][index]
                resp[i] = btable['Resp.'][index]*-1.


            return det_off, noise, grid_angle, pol_angle_offset, resp

>>>>>>> db74452... Solved a bug on applying the offset
