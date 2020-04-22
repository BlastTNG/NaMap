import pygetdata as gd
import numpy as np
from scipy.interpolate import interp1d
import os
import astropy.table as tb

import src.detector as det

class data_value():
    
    '''
    Class for reading the values of the TODs (detectors and coordinates) from a DIRFILE
    '''

    def __init__(self, det_path, det_name, coord_path, coord1_name, \
                 coord2_name, det_file_type, coord1_file_type, coord2_file_type, \
                 experiment, lst_file_type, lat_file_type, hwp_file_type, startframe,\
                 endframe):

        '''
        For BLAST-TNG the detector name is given as kid_# where # is 1,2,3,4,5
        The number is then converted to the equivalent letters that are coming from 
        the telemetry name
        '''

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

        self.startframe = int(startframe)           #Starting frame to be analyzed
        self.endframe = int(endframe)               #Ending frame to be analyzed

        if self.startframe < 100:
            self.bufferframe = int(0)                      #Buffer frames to be loaded before and after the starting and ending frame
        else:
            self.bufferframe = int(100)

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
            if file_type is not None:
                gdtype = self.conversion_type(file_type)
            else:
                gdtype = gd.FLOAT64
            if self.experiment.lower()=='blast-tng':
                num = self.endframe-self.startframe+2*self.bufferframe
                first_frame = self.startframe-self.bufferframe
            else:
                num = d.nframes
                first_frame = 0
            if isinstance(file, str):
                values = d.getdata(file, gdtype, num_frames = num, first_frame=first_frame)
            else:
                values = d.getdata(file[0], gdtype, num_frames = num, first_frame=first_frame)
            return np.asarray(values)
        else:
            d = gd.dirfile(filepath, gd.RDONLY)
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
 
            list_conv = [['A', 'B'], ['D', 'E'], ['G', 'H'], ['K', 'I'], ['M', 'N']]
            kid_num  = int(self.det_name[-1])

            try:
                det_I_string = 'kid'+list_conv[kid_num-1][0]+'_roachN'
                det_Q_string = 'kid'+list_conv[kid_num-1][1]+'_roachN'
                I_data = self.load(self.det_path, det_I_string, self.det_file_type)
                Q_data = self.load(self.det_path, det_Q_string, self.det_file_type)
            except:
                val = str(kid_num)
                det_I_string = 'i_kid000'+val+'_roach3'
                det_Q_string = 'q_kid000'+val+'_roach3'
                I_data = self.load(self.det_path, det_I_string, self.det_file_type)
                Q_data = self.load(self.det_path, det_Q_string, self.det_file_type)

            kidutils = det.kidsutils()

            det_data = kidutils.KIDmag(I_data, Q_data)

        else:
            det_data = self.load(self.det_path, self.det_name, self.det_file_type)
        
        print('COORDINATES', self.coord1_name.lower(), self.coord2_name.lower())

        if self.coord2_name.lower() == 'dec':
            if self.experiment.lower()=='blast-tng':
                coord2 = 'DEC'
                filetype = None
            else:
                coord2 = 'dec'
                filetype = self.coord2_file_type
            coord2_data = self.load(self.coord_path, coord2, filetype)
        elif self.coord2_name.lower() == 'y':
            coord2_data = self.load(self.coord_path, 'y_stage', self.coord2_file_type)
        else:
            if self.experiment.lower()=='blast-tng':
                coord2 = 'EL'
                filetype = None
            else:
                coord2 = 'el'
                filetype = self.coord2_file_type
            coord2_data = self.load(self.coord_path, coord2, filetype)

        if self.coord1_name.lower() == 'ra':
            if self.experiment.lower()=='blast-tng':
                coord1 = 'RA'
                filetype = None
            else:
                coord1 = 'ra'
                filetype = self.coord1_file_type
            coord1_data = self.load(self.coord_path, coord1, filetype)
        elif self.coord1_name.lower() == 'x':
            coord1_data = self.load(self.coord_path, 'x_stage', self.coord1_file_type)
        else:
            if self.experiment.lower()=='blast-tng':
                coord1 = 'AZ'
                filetype = None
            else:
                coord1 = 'az'
                filetype = self.coord1_file_type
            coord1_data = self.load(self.coord_path, coord1, filetype)

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
                 hwp_fs=None, hwp_sample_frame=None, xystage=False):

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
        if hwp_fs is not None:
            self.hwp_fs = float(hwp_fs)
        else:
            self.hwp_fs = hwp_fs
        if hwp_sample_frame is not None:
            self.hwp_sample_frame = float(hwp_sample_frame)
        else:
            self.hwp_sample_frame = hwp_sample_frame

        self.xystage=xystage                                   #Flag to check if the coordinates data are coming from an xy stage scan

        if self.startframe < 100:
            self.bufferframe = int(0)
        else:
            self.bufferframe = int(100)

    def frame_zoom(self, data, sample_frame, fs, fps, offset = None):

        '''
        Selecting the frames of interest and associate a timestamp for each value.
        '''

        frames = fps.copy()

        frames[0] = fps[0]*sample_frame
        if fps[1] == -1:
            frames[1] = len(data)*sample_frame
        else:
            frames[1] = fps[1]*sample_frame+1

        if offset is not None:
            delay = offset*np.floor(fs)/1000.
            frames = frames.astype(float)+delay

        if len(np.shape(data)) == 1:
            time = (np.arange(np.diff(frames))+frames[0])/np.floor(fs)
            return time, data[int(frames[0]):int(frames[1])]
        else:
            time = np.arange(len(data[0, :]))/np.floor(fs)
            time = time[int(frames[0]):int(frames[1])]
            return  time, data[:,int(frames[0]):int(frames[1])]

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
            d = gd.dirfile(self.roach_pps_path)
            
            first_frame = self.startframe-self.bufferframe
            num_frames = self.endframe-self.startframe+2*self.bufferframe
            interval = self.endframe-self.startframe
            print('FRAMES', first_frame, num_frames, interval)
            ctime_mcp = d.getdata('time', first_frame=first_frame, num_frames=num_frames)
            ctime_usec = d.getdata('time_usec', first_frame=first_frame, num_frames=num_frames)
            framecount_100hz = d.getdata('mcp_100hz_framecount', first_frame=first_frame, num_frames=num_frames)
            print('LEN', len(ctime_mcp), len(ctime_usec), len(framecount_100hz))
            if self.xystage is True:
                #frequency_ctime = 100
                sample_ctime = 100
            else:
                #frequency_ctime = self.coord_fs
                sample_ctime = self.coord_sample_frame
            #ctime_start_temp = ctime_mcp[0]+ctime_usec[0]/1e6+0.2
            #ctime_mcp = ctime_start_temp + (framecount_100hz-framecount_100hz[0])/frequency_ctime
            ctime_start = ctime_mcp+ctime_usec/1e6+0.2
            ctime_mcp = ctime_mcp[self.bufferframe*sample_ctime:self.bufferframe*sample_ctime+interval*sample_ctime]
 
            if self.offset is not None:
                ctime_mcp += self.offset/1000.

            ctime_start = ctime_mcp[0]
            ctime_end = ctime_mcp[-1]
            
            coord1 = self.coord1_data[self.bufferframe*self.coord_sample_frame:self.bufferframe*self.coord_sample_frame+\
                                      interval*self.coord_sample_frame]
            coord2 = self.coord2_data[self.bufferframe*self.coord_sample_frame:self.bufferframe*self.coord_sample_frame+\
                                      interval*self.coord_sample_frame]
                                    
            if self.xystage is True:
                freq_array = np.append(0, np.cumsum(np.repeat(1/self.coord_sample_frame, self.coord_sample_frame*interval-1)))
                coord1time = ctime_start+freq_array
                coord2time = coord1time.copy()
            else:
                if self.coord_sample_frame != 100:
                    freq_array = np.append(0, np.cumsum(np.repeat(1/self.coord_sample_frame, self.coord_sample_frame*interval-1)))
                    coord1time = ctime_start+freq_array
                    coord2time = coord1time.copy()
                else:
                    coord1time = ctime_mcp.copy()
                    coord2time = ctime_mcp.copy()

            print(coord1, len(self.coord1_data), len(coord1), len(coord1time))

            kidutils = det.kidsutils()
            
            start_det_frame = self.startframe-self.bufferframe
            end_det_frame = self.endframe+self.bufferframe

            frames = np.array([start_det_frame, end_det_frame], dtype='int')

            dettime, pps_bins = kidutils.det_time(self.roach_pps_path, self.roach_number, frames, \
                                                  ctime_start, ctime_mcp[-1], self.det_fs)

            coord1int = interp1d(coord1time, coord1, kind='linear')
            coord2int = interp1d(coord2time, coord2, kind= 'linear')

            idx_roach_start, = np.where(np.abs(dettime-ctime_start) == np.amin(np.abs(dettime-ctime_start)))
            idx_roach_end, = np.where(np.abs(dettime-ctime_end) == np.amin(np.abs(dettime-ctime_end)))

            if len(np.shape(self.det_data)) == 1:
                self.det_data = kidutils.interpolation_roach(self.det_data, pps_bins[pps_bins>350], self.det_fs)
            else:
                for i in range(len(self.det_data)):
                    self.det_data[i] = kidutils.interpolation_roach(self.det_data[i], pps_bins[pps_bins>350], self.det_fs)
            
            dettime = dettime[idx_roach_start[0]:idx_roach_end[0]]
            self.det_data = self.det_data[idx_roach_start[0]:idx_roach_end[0]]

            index1, = np.where(np.abs(dettime-coord1time[0]) == np.amin(np.abs(dettime-coord1time[0])))
            index2, = np.where(np.abs(dettime-coord1time[-1]) == np.amin(np.abs(dettime-coord1time[-1])))

            coord1_inter = coord1int(dettime[index1[0]+200:index2[0]-200])
            coord2_inter = coord2int(dettime[index1[0]+200:index2[0]-200])
            dettime = dettime[index1[0]+200:index2[0]-200]

            print('COORDINATES', np.amin(coord2_inter), np.amax(coord2_inter), len(coord2_inter))

            if len(np.shape(self.det_data)) == 1:
                self.det_data = self.det_data[index1[0]+200:index2[0]-200]
            else:
                for i in range(len(self.det_data)):
                    self.det_data[i] = self.det_data[i, index1[0]+200:index2[0]-200]

        elif self.experiment.lower() == 'blastpol':
            dettime, self.det_data = self.frame_zoom(self.det_data, self.det_sample_frame, \
                                                     self.det_fs, np.array([self.startframe,self.endframe]), \
                                                     self.offset)
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

            dettime = dettime[index1[0]+10:index2[0]-10]
            self.det_data = self.det_data[:,index1[0]+10:index2[0]-10]

        if isinstance(self.hwp_data, np.ndarray):

            if self.experiment.lower() == 'blastpol':
                hwptime, hwp = self.frame_zoom(self.hwp_data, self.hwp_sample_frame, \
                                               self.hwp_fs, np.array([self.startframe,self.endframe]))
                
                hwptime = hwptime - hwptime[0]
                index1, = np.where(np.abs(dettime-hwptime[0]) == np.amin(np.abs(dettime-hwptime[0])))
                index2, = np.where(np.abs(dettime-hwptime[-1]) == np.amin(np.abs(dettime-hwptime[-1])))

                hwp_interpolation = interp1d(hwptime, hwp, kind='linear')
                hwp_inter = hwp_interpolation(dettime[index1[0]+10:index2[0]-10])

            else:
                
                hwp = self.hwp_data[self.bufferframe*self.coord_sample_frame:self.bufferframe*self.coord_sample_frame+\
                                    interval*self.coord_sample_frame]

                freq_array = np.append(0, np.cumsum(np.repeat(1/self.hwp_sample_frame, self.hwp_sample_frame*interval-1)))
                hwptime = ctime_start+freq_array

                hwp_interpolation = interp1d(hwptime, hwp, kind='linear')
                hwp_inter = hwp_interpolation(dettime)
 
            del hwptime
            del hwp

        else:

            hwp_inter = np.zeros_like(coord1_inter)

        del coord1time
        del coord2time
        del coord1
        del coord2

        if self.lat_data is not None and self.lat_data is not None:

            if self.experiment.lower() == 'blastpol':
                lsttime, lst = self.frame_zoom(self.lst_data, self.lstlat_sample_frame, \
                                               self.lstlatfreq, np.array([self.startframe,self.endframe]))

                lattime, lat = self.frame_zoom(self.lat_data, self.lstlat_sample_frame, \
                                               self.lstlatfreq, np.array([self.startframe,self.endframe]))

                lsttime = lsttime-lsttime[0]
                index1, = np.where(np.abs(dettime-lsttime[0]) == np.amin(np.abs(dettime-lsttime[0])))
                index2, = np.where(np.abs(dettime-lsttime[-1]) == np.amin(np.abs(dettime-lsttime[-1])))

                lst_inter, lat_inter = self.coord_int(lst, lat, \
                                                      lsttime, dettime[index1[0]+10:index2[0]-10])

            else:
                lst = self.lst_data[self.bufferframe*self.coord_sample_frame:self.bufferframe*self.coord_sample_frame+\
                                    interval*self.coord_sample_frame]
                lat = self.lat_data[self.bufferframe*self.coord_sample_frame:self.bufferframe*self.coord_sample_frame+\
                                    interval*self.coord_sample_frame]

                lsttime = ctime_mcp.copy()
                lattime = ctime_mcp.copy()

                lstint = interp1d(lsttime, lst, kind='linear')
                latint = interp1d(lattime, lat, kind= 'linear')

                lst_inter = lstint(dettime)
                lat_inter = latint(dettime)

            del lst
            del lat

            if np.size(np.shape(self.det_data)) > 1:
                return (dettime, self.det_data, \
                        coord1_inter, coord2_inter, hwp_inter, lst_inter, lat_inter)
            else:
                return (dettime, self.det_data, \
                        coord1_inter, coord2_inter,  hwp_inter, lst_inter, lat_inter)
        
        else:
            if np.size(np.shape(self.det_data)) > 1:
                return (dettime, self.det_data, \
                        coord1_inter, coord2_inter, hwp_inter)
            else:
                return (dettime, self.det_data, \
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


