import pygetdata as gd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

kidstable = '../kidstable.txt'

kidsarray = np.loadtxt(kidstable,skiprows=1)

responsivity_pos = 4
det_pos = 2

def detToR(det_num):
    r = kidsarray[kidsarray[:,det_pos]==det_num,responsivity_pos][0]
    return r

def rotatePhase(i_ts,q_ts):
    X = i_ts+1j*q_ts
    phi_avg = np.arctan2(np.mean(q_ts),np.mean(i_ts))
    E = X*np.exp(-1j*phi_avg)
    i_out = E.real 
    q_out = E.imag
    return i_out, q_out

def dirfileToUseful(file_name,data_type):
    nom_path = '../roach_data'
    q = gd.dirfile(nom_path,gd.RDONLY)
    values = q.getdata(file_name,data_type)
    return values


def phasetopower(ifile,qfile,det_num,startframe,endframe):
    r = detToR(det_num)
    i = dirfileToUseful(ifile,gd.FLOAT32)
    q = dirfileToUseful(qfile,gd.FLOAT32)
    i = i[startframe:endframe]
    q = q[startframe:endframe]
    i_rot, q_rot = rotatePhase(i,q)
    phi = np.arctan2(q_rot,i_rot)
    phibar = np.arctan2(np.mean(q),np.mean(i))
    delphi = phi-phibar
    power = delphi/r
    power = power-np.mean(power[len(power)-1001:len(power)-1])
    return power

p = phasetopower('kidA_roachN','kidB_roachN',2,6738000,6743000)

plt.plot(p)
plt.show()


