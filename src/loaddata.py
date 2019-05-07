import pygetdata as gd
import numpy as np
from scipy.interpolate import interp1d

class data_value():

    def __init__(self, det_path, det_name, coord_path, coord1_name, \
                 coord2_name, det_file_type, coord1_file_type, coord2_file_type, \
                 experiment):

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
        if np.size(file) == 1: 
            d = gd.dirfile(filepath, gd.RDONLY)
            vectors = d.field_list()
            print(vectors)

            for i in range (len(vectors)):
                if vectors[i] == file:
                    gdtype = self.conversion_type(file_type)
                    if self.experiment.lower()=='blast-tng':
                        num = d.eof('MCP_1HZ_FRAMECOUNT')
                    else:
                        num = d.nframes
                    
                    values = d.getdata(file, num_frames = num-1, first_frame=0)
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
        return det_data, coord1_data, coord2_data

class convert_dirfile():

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
        if roach_number != None:
            self.roach_number = int(float(roach_number))
        else:
            self.roach_number = roach_number
        self.roach_pps_path = roach_pps_path

    def frame_zoom(self, data, sample_frame, fs, fps):

        frames = fps.copy()

        frames[0] = fps[0]*sample_frame
        frames[1] = fps[1]*sample_frame+1

        if len(np.shape(data)) == 1:
            time = np.arange(len(data))/np.floor(fs)
            time = time[frames[0]:frames[1]]

            return time, data[frames[0]:frames[1]]
        else:
            time = np.arange(len(data[0, :]))/np.floor(fs)
            time = time[frames[0]:frames[1]]
            return  time, data[:,frames[0]:frames[1]]

    def det_time(self):
        
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

        coord1_int = interp1d(time_acs, coord1, kind='linear')
        coord2_int = interp1d(time_acs, coord2, kind= 'linear')

        return coord1_int(time_det), coord2_int(time_det)

    def sync_data(self):
        if self.experiment.lower() == 'blast-tng':
            dettime = self.det_time()
        elif self.experiment.lower() == 'blastpol':
            dettime, detTOD = self.frame_zoom(self.det_data, self.det_sample_frame, \
                                          self.det_fs, np.array([self.frame1,self.frame2]))

        coord1time, coord1 = self.frame_zoom(self.coord1_data, self.coord_sample_frame, \
                                          self.coord_fs, np.array([self.frame1,self.frame2]))

        coord2time, coord2 = self.frame_zoom(self.coord2_data, self.coord_sample_frame, \
                                          self.coord_fs, np.array([self.frame1,self.frame2]))

        index1, = np.where(np.abs(dettime-coord1time[0]) == np.amin(np.abs(dettime-coord1time[0])))
        index2, = np.where(np.abs(dettime-coord1time[-1]) == np.amin(np.abs(dettime-coord1time[-1])))

        coord1_inter, coord2_inter = self.coord_int(coord1, coord2, \
                                                    coord1time, dettime[index1[0]+10:index2[0]-10])
        
        detTOD = self.det_data.copy()

        return dettime[index1[0]+10:index2[0]-10], detTOD[index1[0]+10:index2[0]-10], \
               coord1_inter, coord2_inter

