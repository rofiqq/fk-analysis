# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 17:59:41 2018

@author: ainurrofiq
"""

from obspy import read, UTCDateTime
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial import distance
from scipy.interpolate import griddata

#---------------------------
# ---------- Edit ----------
#---------------------------
recordformat= 'mseed'
coordfile = 'example.txt'
smax = 0.3
fmin, fmax = 0.01, 9
tmin = UTCDateTime("2004-12-26T01:7:10.5")
tmax = tmin+60*2
# --------------------------

# Import record files
def stream(unique, akhir):
    RecordFiles=[a for a in os.listdir(os.getcwd()) if unique in a and a.endswith(akhir)]
    st=read()    
    st.clear()
    for files in RecordFiles:
        st += read(files)
    return st

# Import coordinate files and change to relative coordinate
def coordinates(filename):
    floc=open(filename,'r')
    fline=floc.readlines()
    coords = []
    for i in range(len(fline)):
        fline[i]=fline[i].split()
        coords.append([float(fline[i][1]),float(fline[i][2]),float(fline[i][3])])
    xmin, xmax = min(np.array(coords)[:,0]), max(np.array(coords)[:,0])
    ymin, ymax = min(np.array(coords)[:,1]), max(np.array(coords)[:,1])
    closest_pt_index = distance.cdist([((xmin+xmax)/2, (ymin+ymax)/2)], np.array(coords)[:,0:2]).argmin()
    LocDict={}
    for CoorPoint in fline:
        LocDict[CoorPoint[0]]=(
                [(float(CoorPoint[1])-coords[closest_pt_index][0])/1000,
                 (float(CoorPoint[2])-coords[closest_pt_index][1])/1000,
                 float(CoorPoint[3])])
    return LocDict
    
def f_k(recordformat, coordfile, smax, fmin, fmax, tmin, tmax):
    LocDict = coordinates(coordfile)
    st = stream('', recordformat)
    # Apply coordinate to stream
    coordinate = []
    for trace in st :
        if LocDict.has_key(trace.stats.station)==True:
            trace.stats.location=LocDict[trace.stats.station]
        coordinate.append(trace.stats.location)
    coordinate = np.array(coordinate)  
    
    # Trim all signal
    st = st.copy().trim(starttime=tmin, endtime=tmax)    
       
    # Time step and number of stations to be stacked
    delta = st[0].stats.delta
    nbeam = len(st)
    
    # Pre-process, and filter to frequency window
    st.detrend()
    st.taper(type='cosine', max_percentage=0.05)
    
    st = st.copy().filter("bandpass", freqmin=fmin, freqmax=fmax)   
    npts = st[0].stats.npts
    
    # Computer Fourier transforms for each trace
    fft_st = np.zeros((nbeam, (npts / 2) + 1), dtype=complex) # Length of real FFT is only half that of time series data
    for i, tr in enumerate(st):
        fft_st[i, :] = np.fft.rfft(tr.data) # Only need positive frequencies, so use rfft
    
    freqs = np.fft.fftfreq(npts, delta)[0:(npts / 2) + 1]
    
    # Slowness increment
    sinc = 2*nbeam
    
    # Shange max slowness amplitude
    smaxnew = smax/(np.sin(np.radians(45)))
    
    # Make grid from slowness and backazimuth
    slownessVector = np.linspace(0, smaxnew, sinc)
    backazimVector = np.arange(0, 360.1, 2)
    
    # Array geometry
    x, y = np.split(coordinate[:, :2], 2, axis=1)
    
    # Calculate the F-K spectrum
    # make matrix for slowness (r), backazimuth (theta), power (fk)
    fk = np.zeros((len(backazimVector), len(slownessVector)))
    theta = np.zeros((len(backazimVector), len(slownessVector)))
    r = np.zeros((len(backazimVector), len(slownessVector)))
    for ii in range(len(backazimVector)):
        for jj in range(len(slownessVector)):
            # change to cartesian (minus degree because clockwise)
            slow_x = slownessVector[jj]*np.sin(np.radians(backazimVector[ii]))
            slow_y = slownessVector[jj]*np.cos(np.radians(backazimVector[ii]))
            func = 0
            for kk in range(len(x)):
                dt = slow_x * x[kk] + slow_y * y[kk]
                func += (np.exp(-1j * 2 * np.pi * dt * freqs))*(fft_st[kk])**2
            fk[ii, jj] = np.sum(abs(func)/nbeam)
            theta[ii, jj] = backazimVector[ii]
            r[ii, jj] = slownessVector[jj]
    
    # Average power for all signal
    tracepower = np.sum(np.sum(abs(fft_st**2), axis=1))/nbeam
    
    # Relative power in dB
    fk = 10*np.log10(fk/tracepower)
    
    # convert to cartesian
    Sx = r*np.sin(np.radians(theta))
    Sy = r*np.cos(np.radians(theta))
    
    # calculate backazimuth and slowness more detail by interpolation
    SxInt, SyInt = np.meshgrid(np.linspace(-smax, smax, 100),
                               np.linspace(-smax, smax, 100))
    
    fkInt = griddata((Sx.reshape(-1), Sy.reshape(-1)), 
                     fk.reshape(-1), 
                     (SxInt, SyInt),method='linear')
    
    # Find loc max power
    fkIntmax = np.unravel_index(np.nanargmax(fkInt), fkInt.shape)
    
    # Find Sx and Sy interpolate when power is maximum
    SxInt_max = SxInt[fkIntmax]
    SyInt_max = SyInt[fkIntmax]
    
    slowness = np.hypot(SxInt_max, SyInt_max)
    backazimuth = np.degrees(np.arctan2(SxInt_max, SyInt_max))
    if backazimuth < 0:
        backazimuth += 360.
    
    # Plot in Cartesian
    fig = plt.figure(figsize=(10, 8), tight_layout = True)
    fig.add_axes()
    plt.contourf(Sx, Sy, fk, cmap=plt.cm.jet,levels=np.arange(-10,0.05,0.1))
    plt.grid('on', color='lightgray',linestyle='--')
    plt.xlabel('Sx (s/km)')
    plt.ylabel('Sy (s/km)')
    plt.colorbar(ticks=np.arange(-10,0.1,1), label = 'Relative Power (dB)')
    plt.clim(-10,0)
    plt.xlim(-smax, smax);
    plt.ylim(-smax, smax);  
    plt.title("FK Analysis, slowness= " + '%.4f' % slowness + " s/km,  backazimuth= " + '%.1f' % backazimuth + " deg")
    plt.show()
    fig.savefig('F-K Analysis_cartesian.png',
                    bbox_inches="tight",dpi=fig.dpi)
    # Plot in polar
    fig2, ax = plt.subplots(figsize=(10,8),tight_layout=True, subplot_kw=dict(projection='polar'))
    plt.rc('grid', color='lightgray', linestyle='--')
    ax.grid('on', color='lightgray',linestyle='--')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    plt.ylim(ymax= smax)
    cax = ax.contourf(np.radians(theta), r, fk,
                      cmap=plt.cm.jet,
                      levels=np.arange(-10,0.05,0.1))
    r_label = []
    ax.set_yticklabels([])
    for i in ax.get_yticks():
        if i < smax:
            r_label.append(str(i)+' s/km')
            ax.annotate(str(i),
                    xy=(np.radians(90),i), xycoords='data',
                    horizontalalignment='center',
                    verticalalignment='top'
                    )
    ax.annotate('Slowness (s/km)',
            xy=(np.radians(87.5),smax/2), xycoords='data',
            xytext=(0, 0.01), textcoords='offset points',
            horizontalalignment='center',
            verticalalignment='botto'
            )
    plt.title("FK Analysis, slowness= " + '%.4f' % slowness + " s/km,  backazimuth= " + '%.1f' % backazimuth + " deg")
    cb = fig2.colorbar(cax,ticks=np.arange(-10,0.1,1), pad=0.1, extend='both',
                      shrink=0.5, orientation='horizontal') 
    cb.ax.invert_xaxis()
    cb.set_clim(-10,0)
    cb.ax.set_title('Relative Power (dB)')
    fig2.savefig('F-K Analysis_polar.png',
                    bbox_inches="tight",dpi=fig.dpi)
    return

def main():
    f_k(recordformat, coordfile, smax, fmin, fmax, tmin, tmax)
if __name__ == "__main__":
  main()

    