import numpy as np
import os
import scipy.signal as sgn
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

class data_cleaned():

    def __init__(self, data, fs, cutoff):

        self.data = data
        self.fs = float(fs)
        self.cutoff = float(cutoff)

    def data_clean(self):
        
        det_data = detector(self.data, 0, 0)
        residual_data = det_data.fit_residual()

        desp = despike(residual_data)
        data_despiked = desp.replace_peak()

        filterdat = filterdata(data_despiked, self.cutoff, self.fs)
        cleaned_data = filterdat.ifft_filter(window=True)

        return cleaned_data
    
class despike():

    def __init__(self, data):

        self.data = data

    def findpeak(self, hthres=0, pthres=0):

        '''
        hthresh and pthres are measured in how many std the height (or the prominence) 
        of the peak is computed. The height of the peak is computed with respect to 
        the mean of the signal        
        '''
        index = np.ones(1)
        index_final = np.array([], dtype = 'int')
        ledge = np.array([], dtype = 'int')
        redge = np.array([], dtype = 'int')
        count = 0

        y_std = np.std(self.data)
        y_mean = np.mean(self.data)

        if hthres != 0 and pthres == 0:
            index, param = sgn.find_peaks(np.abs(self.data), height = y_mean + hthres*y_std, distance=100)
        elif pthres != 0 and hthres == 0:
            index, param = sgn.find_peaks(np.abs(self.data), prominence = pthres*y_std)
        elif hthres != 0 and pthres != 0:
            index, param = sgn.find_peaks(np.abs(self.data), height = y_mean + hthres*y_std, \
                                            prominence = pthres*y_std)

        ledget = sgn.peak_widths(np.abs(self.data),index)[2]
        redget = sgn.peak_widths(np.abs(self.data),index)[3]

        ledge = np.append(ledge, np.floor(ledget).astype(int))
        redge = np.append(redge, np.ceil(redget).astype(int))

        return index

    def peak_width(self, hthres=5, pthres=0, window = 100):

        y_mean = np.mean(self.data)
        
        peaks = self.findpeak(hthres=hthres, pthres=pthres)
        param = sgn.peak_widths(np.abs(self.data),peaks, rel_height = 1.0)

        height = self.data[peaks]-y_mean

        ledge = np.array([], dtype='int')
        redge = np.array([], dtype='int')

        for i in range(len(peaks)):
            left_edge, = np.where(self.data[peaks[i]-window:peaks[i]] == \
                                  np.amin(self.data[peaks[i]-window:peaks[i]]))
            right_edge, = np.where(self.data[peaks[i]:peaks[i]+window] == \
                                  np.amin(self.data[peaks[i]:peaks[i]+window]))

            left_edge += (peaks[i]-window)
            right_edge += peaks[i]

            ledge = np.append(ledge, left_edge)
            redge = np.append(redge, right_edge)

        return param[0].copy(), ledge, redge

    def replace_peak(self, hthres=5, pthres = 0, peaks = np.array([]), widths = np.array([])):

        x_inter = np.array([], dtype = 'int')

        if np.size(peaks) == 0:
            peaks = self.findpeak(hthres=hthres, pthres=pthres)
        if np.size(widths) == 0:
            widths = self.peak_width(hthres=hthres, pthres=pthres)

        replaced = self.data.copy()
        for i in range(0, len(peaks)):
            width = int(np.ceil(widths[0][i]))
            if width <= 13:
                interval = 25
            elif width > 13 and width < 40:
                interval = width*2
            else:
                interval = width*3

            left_edge = int(np.floor(widths[1][i]))
            right_edge = int(np.ceil(widths[2][i]))

            # plt.plot(self.data)
            # plt.plot(peaks, self.data[peaks], 'x')
            # plt.plot(left_edge, self.data[left_edge], 'x')
            # plt.plot(right_edge, self.data[right_edge], 'x')
            # plt.show()

            x_inter = np.append(x_inter, np.arange(left_edge, right_edge))
            replaced[left_edge:right_edge] = (replaced[left_edge]+\
                                              replaced[right_edge])/2.

        final_mean = np.mean(replaced)
        final_std = np.std(replaced)
        final_var = np.var(replaced)

        p_stat = np.abs(final_mean/final_var-1.)

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

        return replaced

class filterdata():

    def __init__(self, data, cutoff, fs):
        
        '''
        fs: sample frequency
        cutoff: cutoff frequency
        '''

        self.data = data
        self.cutoff = cutoff
        self.fs = fs
    
    def highpass(self, order):
        
        nyq = 0.5*self.fs
        normal_cutoff = self.cutoff / nyq
        b, a = sgn.butter(order, normal_cutoff, btype='highpass', analog=False)
        return b, a

    def butter_highpass_filter(self, order=5):
        b, a = self.highpass(order)
        filterdata = sgn.lfilter(b, a, self.data)
        return filterdata

    def cosine_filter(self, f):
        if f < .5*self.cutoff:
            return 0
        elif 0.5*self.cutoff <= f  and f <= self.cutoff:
            return 0.5-0.5*np.cos(np.pi*(f-0.5*self.cutoff)*(self.cutoff-0.5*self.cutoff)**-1)
        elif f > self.cutoff:
            return 1
    
    def fft_filter(self, window):
        if window == True:
            window_data = np.hanning(len(self.data))

            fft_data = np.fft.rfft(self.data*window_data)
        else:
            fft_data = np.fft.rfft(self.data)

        fft_frequency = np.fft.rfftfreq(np.size(self.data), 1/self.fs)

        vect = np.vectorize(self.cosine_filter)

        filtereddata = vect(fft_frequency)*fft_data

        return filtereddata

    def ifft_filter(self, window):

        ifft_data = np.fft.irfft(self.fft_filter(window=window), len(self.data))

        return ifft_data

class detector():

    def __init__(self, data, responsivity, grid):

        self.data = data
        self.responsivity = responsivity
        self.grid = grid

    def calibrate(self):

        return self.data*self.responsivity

    def polangle(self, roll, hwp_angle):

        return self.grid-2*hwp_angle+roll

    def exponentialfit(self, edge, delay=0):
        x = np.arange(len(self.data))
        y_fin = np.array([])
        index_exclude = np.array([], dtype=int)

        if np.size(delay) == 1:
            delay = np.ones(np.size(edge)+1)*delay
            delay[0] = 0
        else:
            delay = delay 

        for i in range(np.size(edge)+1):
            index1 = int(i*edge+delay[i])  
            index2 = int((i+1)*edge)

            popt, pcov = curve_fit(exp, x, self.data)
            y = exp(x[index1:index2], *popt)
            y_fin = np.append(y_fin, y)

            if i != np.size(edge):
                if delay[i+1] > 0:
                    zeros = np.zeros(int(delay[i+1]))
                    y_fin = np.append(y_fin, zeros)
                    index_exclude = np.append(index_exclude, np.arange(delay[i+1])+edge)

        return y_fin, index_exclude.astype(int)

    def polyfit(self, edge = 0, delay=0, order=6):

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

        polyres = self.polyfit(edge=edge, delay=delay, order=order)
        fitteddata = polyres[0]
        index = polyres[1]

        zero_data = self.data.copy()
        zero_data[index] = 0.

        return fitteddata-zero_data