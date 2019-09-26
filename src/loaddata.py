import pygetdata as gd
import numpy as np
from scipy.interpolate import interp1d
import os
import astropy.table as tb

class data_value():
    
    '''
    Class for reading the values of the TODs (detectors and coordinates) from a DIRFILE
    '''

    def __init__(self, det_path, det_name, coord_path, coord1_name, \
                 coord2_name, det_file_type, coord1_file_type, coord2_file_type, \
                 experiment, lst_file_type, lat_file_type, hwp_file_type):
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

        '''
        Function to define the different datatype conversions strings for pygetdata
        '''

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

        '''
        Return the values of the DIRFILE as a numpy array
        
        filepath: path of the DIRFILE to be read
        file: name of the value to be read from the dirfile, e.g. detector name or
              coordinate name
        file_type: data type conversion string for the DIRFILE data
        '''
        if np.size(file) == 1: 
            d = gd.dirfile(filepath, gd.RDONLY)
            gdtype = self.conversion_type(file_type)
            if self.experiment.lower()=='blast-tng':
                num = d.eof('MCP_1HZ_FRAMECOUNT')
            else:
                num = d.nframes
            if isinstance(file, str):
                values = d.getdata(file, gdtype, num_frames = num-1, first_frame=0)
            else:
                values = d.getdata(file[0], gdtype, num_frames = num-1, first_frame=0)
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

        '''
        Function to return the timestreams for detector and coordinates
        '''
        if self.experiment.lower() == 'blast-tng':
            # Temporary function 
            det_data = np.loadtxt(self.det_path+self.det_name)
        else:
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
            
            lat = self.load(self.coord_path, 'lat', self.lat_file_type)
            lst = self.load(self.coord_path, 'lst', self.lst_file_type)

            return det_data, coord1_data, coord2_data, hwp_data, lst, lat
        
        else:
            return det_data, coord1_data, coord2_data, hwp_data

class convert_dirfile():

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

    def frame_zoom(self, data, sample_frame, fs, fps, offset = None):

        '''
        Selecting the frames of interest and associate a timestamp for each value
        '''
        frames = fps.copy()

        frames[0] = fps[0]*sample_frame
        frames[1] = fps[1]*sample_frame+1

        if offset is not None:
                delay = offset*np.floor(fs)/1000.
                frames = frames.astype(float)+delay

        if len(np.shape(data)) == 1:
            time = (np.arange(np.diff(frames))+frames[0])/np.floor(fs)
            return time, data[int(frames[0]):int(frames[1])]
        else:
            print('TEST', len(data[0, :]))
            time = np.arange(len(data[0, :]))/np.floor(fs)
            time = time[int(frames[0]):int(frames[1])]
            print('OK')
            return  time, data[:,int(frames[0]):int(frames[1])]

    def det_time(self):

        '''
        This function is specific for BLAST-TNG only and creates a time array using the PPS data
        '''
        
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
                temp = index_bins[i]+np.arange(0., bins_temp[index_bins[i]], \
                                               1.)/bins_temp[index_bins[i]] 
                    
                time = np.append(time, temp)

        return time

    def coord_int(self, coord1, coord2, time_acs, time_det):

        '''
        Interpolates the coordinates values to compensate for the smaller frequency sampling
        '''

        coord1_int = interp1d(time_acs, coord1, kind='linear')
        coord2_int = interp1d(time_acs, coord2, kind= 'linear')

        return coord1_int(time_det), coord2_int(time_det)

    def sync_data(self):

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
                                             self.coord_fs, np.array([self.startframe,self.endframe]))

        dettime = dettime-dettime[0]
        coord1time = coord1time-coord1time[0]
        index1, = np.where(np.abs(dettime-coord1time[0]) == np.amin(np.abs(dettime-coord1time[0])))
        index2, = np.where(np.abs(dettime-coord1time[-1]) == np.amin(np.abs(dettime-coord1time[-1])))

        coord1_inter, coord2_inter = self.coord_int(coord1, coord2, \
                                                    coord1time, dettime[index1[0]+10:index2[0]-10])

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


        del coord1time
        del coord2time
        del coord1
        del coord2

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
            if np.size(np.shape(self.det_data)) > 1:
                return (dettime[index1[0]+10:index2[0]-10], self.det_data[:,index1[0]+10:index2[0]-10], \
                        coord1_inter, coord2_inter, hwp_inter)
            else:
                return (dettime[index1[0]+10:index2[0]-10], self.det_data[index1[0]+10:index2[0]-10], \
                        coord1_inter, coord2_inter, hwp_inter)

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

