import numpy as np
import scipy.signal as sgn
import pygetdata as gd

class data_cleaned():

    '''
    Class to clean the detector TOD using the functions in 
    the next classes. Check them for more explanations
    '''

    def __init__(self, data, fs, cutoff, detlist, polynomialorder, despike, sigma, prominence):

        self.data = data                #detector TOD
        self.fs = float(fs)             #frequency sampling of the detector
        self.cutoff = float(cutoff)     #cutoff frequency of the highpass filter
        self.detlist = detlist          #detector name list
        self.polynomialorder = polynomialorder #polynomial order for fitting
        self.sigma = sigma                  #height in std value to look for spikes
        self.prominence = prominence        #prominence in std value to look for spikes
        self.despike = despike              #if True despikes the data 
        

    def data_clean(self):

        '''
        Function to return the cleaned TOD as numpy array
        '''
        cleaned_data = np.zeros_like(self.data)


        if np.size(self.detlist) == 1:
            det_data = detector_trend(self.data)
            if self.polynomialorder != 0:
                residual_data = det_data.fit_residual(order=self.polynomialorder)
            else:
                residual_data = self.data

            if self.despike is True:
                desp = despike(residual_data)
                data_despiked = desp.replace_peak(hthres=self.sigma, pthres=self.prominence)
            else:
                data_despiked = residual_data.copy()

            if self.cutoff != 0:
                filterdat = filterdata(data_despiked, self.cutoff, self.fs)
                cleaned_data = filterdat.ifft_filter(window=True)
            else:
                cleaned_data = data_despiked

            return cleaned_data

        else:
            for i in range(np.size(self.detlist)):
                det_data = detector_trend(self.data[i,:])
                residual_data = det_data.fit_residual()

                desp = despike(residual_data)
                data_despiked = desp.replace_peak()

                filterdat = filterdata(data_despiked, self.cutoff, self.fs)
                cleaned_data[i,:] = filterdat.ifft_filter(window=True)

            return cleaned_data
        
class despike():

    '''
    Class to despike the TOD
    '''

    def __init__(self, data):

        self.data = data

    def findpeak(self, hthres=5, pthres=0):

        '''
        This function finds the peak in the TOD.
        hthresh and pthres are measured in how many std the height (or the prominence) 
        of the peak is computed. The height of the peak is computed with respect to 
        the mean of the signal        
        '''

        index = np.ones(1)
        # ledge = np.array([], dtype = 'int')
        # redge = np.array([], dtype = 'int')

        y_std = np.std(self.data)
        y_mean = np.mean(self.data)

        print('MEAN and STD', y_mean, y_std)

        if np.amin(self.data) > 0:
            data_to_despike = self.data-y_mean
        else:
            data_to_despike = self.data.copy()

        # plt.plot(np.abs(data_to_despike))
        # plt.show()

        if hthres != 0 and pthres == 0:
            index, param = sgn.find_peaks(np.abs(data_to_despike), height = hthres*y_std, distance=100)
        elif pthres != 0 and hthres == 0:
            index, param = sgn.find_peaks(np.abs(data_to_despike), prominence = pthres*y_std)
        elif hthres != 0 and pthres != 0:
            index, param = sgn.find_peaks(np.abs(data_to_despike), height = hthres*y_std, \
                                          prominence = pthres*y_std)

        # ledget = sgn.peak_widths(np.abs(data_to_despike),index)[2]
        # redget = sgn.peak_widths(np.abs(data_to_despike),index)[3]

        # ledge = np.append(ledge, np.floor(ledget).astype(int))
        # redge = np.append(redge, np.ceil(redget).astype(int))

        print('INDEX', index)

        return index

    def peak_width(self, peaks, hthres=5, pthres=0, window = 100):


        '''
        Function to estimate the width of the peaks.
        Window is the parameter used by the algorith to find the minimum 
        left and right of the peak. The minimum at left and right is used
        to compute the width of the peak
        '''
        
        #peaks = self.findpeak(hthres=hthres, pthres=pthres)
        y_mean = np.mean(self.data)
        if np.amin(self.data) > 0:
            data_to_despike = self.data-y_mean
        else:
            data_to_despike = self.data.copy()
        param = sgn.peak_widths(np.abs(data_to_despike),peaks, rel_height = 1.0)

        ledge = np.array([], dtype='int')
        redge = np.array([], dtype='int')

        for i in range(len(peaks)):
            left_edge, = np.where(np.abs(data_to_despike[peaks[i]-window:peaks[i]]) == \
                                  np.amin(np.abs(data_to_despike[peaks[i]-window:peaks[i]])))
            right_edge, = np.where(np.abs(data_to_despike[peaks[i]:peaks[i]+window]) == \
                                   np.amin(np.abs(data_to_despike[peaks[i]:peaks[i]+window])))

            left_edge += (peaks[i]-window)
            right_edge += peaks[i]

            ledge = np.append(ledge, left_edge[-1])
            redge = np.append(redge, right_edge[-1])
            print('INDEX', i, peaks[i], left_edge, right_edge)
            print('PEAKS', left_edge, right_edge, peaks[i])
        print(len(peaks), len(ledge), len(redge))
        return param[0].copy(), ledge, redge

    def replace_peak(self, hthres=5, pthres = 5, peaks = np.array([]), widths = np.array([])):

        '''
        This function replaces the spikes data with noise realization. Noise can be gaussian
        or poissonian based on the statistic of the data
        '''

        x_inter = np.array([], dtype = 'int')

        ledge = np.array([], 'int')
        redge = np.array([], 'int')
        replaced = self.data.copy()

        if np.size(peaks) == 0:
            peaks = self.findpeak(hthres=hthres, pthres=pthres)
        if np.size(widths) == 0:
            widths = self.peak_width(peaks=peaks, hthres=hthres, pthres=pthres)

        for i in range(0, len(peaks)):
            # width = int(np.ceil(widths[0][i]))
            # # if width <= 13:
            # #     interval = 25
            # # elif width > 13 and width < 40:
            # #     interval = width*2
            # # else:
            # #     interval = width*3

            left_edge = int(np.floor(widths[1][i]))
            right_edge = int(np.ceil(widths[2][i]))
            ledge = np.append(ledge, left_edge)
            redge = np.append(redge, right_edge)

            x_inter = np.append(x_inter, np.arange(left_edge, right_edge))
            replaced[left_edge:right_edge] = (replaced[left_edge]+\
                                              replaced[right_edge])/2.

        final_mean = np.mean(replaced)
        final_std = np.std(replaced)
        final_var = np.var(replaced)

        p_stat = np.abs(final_mean/final_var-1.)
        print('CHAR', p_stat, final_mean, final_std, final_var)
        if p_stat <=1e-2:
            '''
            This means that the variance and the mean are approximately the 
            same, so the distribution is Poissonian.
            '''
            mu = (final_mean+final_var)/2.
            y_sub = np.random.poisson(mu, len(x_inter))
        else:
            y_sub = np.random.normal(final_mean, final_std, len(x_inter))

        if np.size(y_sub) > 0:
            replaced[x_inter] = y_sub
        #print(left_edge, right_edge)
        #print('TEST', x_inter, ledge, redge)

        return replaced

class filterdata():

    '''
    class for filter the detector TOD
    '''

    def __init__(self, data, cutoff, fs):
        
        '''
        See data_cleaned for parameters explanantion
        '''

        self.data = data
        self.cutoff = cutoff
        self.fs = fs
    
    def highpass(self, order):

        '''
        Highpass butterworth filter.
        order parameter is the order of the butterworth filter
        '''
        
        nyq = 0.5*self.fs
        normal_cutoff = self.cutoff / nyq
        b, a = sgn.butter(order, normal_cutoff, btype='highpass', analog=False)
        return b, a

    def butter_highpass_filter(self, order=5):

        '''
        Data filtered with a butterworth filter 
        order parameter is the order of the butterworth filter
        '''
        b, a = self.highpass(order)
        filterdata = sgn.lfilter(b, a, self.data)
        return filterdata

    def cosine_filter(self, f):

        '''
        Highpass cosine filter
        '''

        if f < .5*self.cutoff:
            return 0
        elif 0.5*self.cutoff <= f  and f <= self.cutoff:
            return 0.5-0.5*np.cos(np.pi*(f-0.5*self.cutoff)*(self.cutoff-0.5*self.cutoff)**-1)
        elif f > self.cutoff:
            return 1
    
    def fft_filter(self, window):

        '''
        Return an fft of the despiked data using the cosine filter.
        Window is a parameter that can be true if the FFT is computed 
        using a Hanning window
        '''

        if window is True:
            window_data = np.hanning(len(self.data))

            fft_data = np.fft.rfft(self.data*window_data)
        else:
            fft_data = np.fft.rfft(self.data)

        fft_frequency = np.fft.rfftfreq(np.size(self.data), 1/self.fs)

        vect = np.vectorize(self.cosine_filter)

        filtereddata = vect(fft_frequency)*fft_data

        return filtereddata

    def ifft_filter(self, window):

        '''
        Inverse FFT of cleaned FFT data calculated in the previous function.
        '''

        ifft_data = np.fft.irfft(self.fft_filter(window=window), len(self.data))

        return ifft_data

class detector_trend():

    '''
    Class to detrend a TOD
    '''

    def __init__(self, data):

        self.data = data

    def polyfit(self, edge = 0, delay=0, order=6):

        '''
        Function to fit a trend line to a TOD
        '''

        x = np.arange(len(self.data))

        y_fin = np.array([])
        index_exclude = np.array([], dtype=int)

        if np.size(edge) == 1:
            p = np.polyfit(x, self.data, order)
            poly = np.poly1d(p)
            y_fin = poly(x)
        else:
            if np.size(delay) == 1:
                delay = np.ones(np.size(edge)+1)*delay
                delay[0] = 0
            else:
                delay = delay 
            for i in range(np.size(edge)+1):
                index1 = int(i*edge+delay[i])  
                index2 = int((i+1)*edge)
                
                p = np.polyfit(x[index1:index2], \
                               self.data[index1:index2], order)

                poly = np.poly1d(p)
                y = poly(x[index1:index2])
                y_fin = np.append(y_fin, y)

                if i != np.size(edge):
                    if delay[i+1] > 0:
                        zeros = np.zeros(int(delay[i+1]))
                        y_fin = np.append(y_fin, zeros)
                        index_exclude = np.append(index_exclude, np.arange(delay[i+1])+edge)

        return y_fin, index_exclude.astype(int)
    
    def fit_residual(self, edge = 0, delay=0, order=6):

        '''
        Function to remove the trend polynomial from the TOD
        '''

        polyres = self.polyfit(edge=edge, delay=delay, order=order)
        fitteddata = polyres[0]
        index = polyres[1]

        zero_data = self.data.copy()
        zero_data[index] = 0.

        return -fitteddata+zero_data

class kidsutils():

    '''
    Class containing useful functions for KIDs
    '''

    def rotatePhase(self, I, Q):

        '''
        Rotate phase for a KID
        '''

        X = I+1j*Q
        phi_avg = np.arctan2(np.mean(Q),np.mean(I))
        E = X*np.exp(-1j*phi_avg)
        I = E.real
        Q = E.imag

        return I, Q

    def KIDphase(self, I, Q):

        '''
        Compute the phase of a KID. This is proportional to power, in particular
        Power = Phase/Responsivity
        '''

        phibar = np.arctan2(np.mean(Q),np.mean(I))
        I_rot, Q_rot = self.rotatePhase(I, Q)
        phi = np.arctan2(Q,I)

        return phi-phibar

    def KIDmag(self, I, Q):

        ''' 
        Compute the magnitude response of a KID
        '''

        return np.sqrt(I**2+Q**2 )

    def interpolation_roach(self, data, bins, sampling):

        '''
        data: values that need to be interpolated
        bins: bins of data coming from the pps signal
        sampling: frequency sampling of the detectors 
        '''

        start = np.append(0, np.cumsum(bins[:-1]))
        end = np.cumsum(bins)
        ln = np.linspace(start, end-1, int(sampling))
        idx = np.reshape(np.transpose(ln), np.size(ln))
        idx_plus = np.append(idx[:-1]+1, idx[-1])
        
        return (data[idx_plus.astype(int)]-data[idx.astype(int)])*(idx-idx.astype(int))+data[idx.astype(int)]
     
    def det_time(self, path, roach_number, frames, ctime_start, ctime_end, sampling):

        roach_string_ctime = 'ctime_packet_roach'+str(int(roach_number))
        pps_roach_string = 'pps_count_roach'+str(int(roach_number))

        d = gd.dirfile(path)
        num_frames= int(frames[1]-frames[0])
        ctime_roach = d.getdata(roach_string_ctime, first_frame=frames[0], num_frames=num_frames)
        pps = d.getdata(pps_roach_string, first_frame=frames[0], num_frames=num_frames)
        ctime_roach = np.ones_like(pps)*ctime_roach[0]

        bn = np.bincount(pps)
        bins = bn[bn>0]
        
        if bins[0] < 350:
            pps = pps[bins[0]:]
            ctime_roach = ctime_roach[bins[0]:]
        if bins[-1] < 350:
            pps = pps[:-bins[-1]]
            ctime_roach = ctime_roach[:-bins[-1]]

        #idx_roach_start, = np.where(np.abs(ctime_roach-ctime_start) == np.amin(np.abs(ctime_roach-ctime_start)))
        #idx_roach_end, = np.where(np.abs(ctime_roach-ctime_end) == np.amin(np.abs(ctime_roach-ctime_end)))
        ctime_roach = ctime_roach*1e-2
        ctime_roach += 1570000000
        pps_duration = pps[-1]-pps[0]+1
        pps_final = pps[0]+np.arange(0, pps_duration, 1/488)

        ctime_roach = self.interpolation_roach(ctime_roach, bins[bins>350], sampling)
        ctime_roach += pps_final

        return ctime_roach, bins