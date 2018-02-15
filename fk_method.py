
from obspy import read, UTCDateTime
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial import distance

#---------------------------
# ---------- Edit ----------
#---------------------------
recordformat= 'mseed'
coordfile = 'example.txt'
smax = 0.2
fmin, fmax = 0.01, 9
tmin = UTCDateTime("2004-12-26T01:7:10.5")
tmax = tmin+120
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
                 float(CoorPoint[3])]
                )
    return LocDict
    
def fk(recordformat, coordfile, smax, fmin, fmax, tmin, tmax):
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
    # Slowness grid
    sinc = 2*nbeam
    slow_x = np.linspace(-smax, smax, sinc)
    slow_y = np.linspace(-smax, smax, sinc)
    
    # Array geometry
    x, y = np.split(coordinate[:, :2], 2, axis=1)  
    
    # Calculate the F-K spectrum
    fk = np.zeros((sinc, sinc))
    
    for ii in range(sinc):
        for jj in range(sinc):
            func = 0
            for kk in range(len(x)):
                dt = slow_x[jj] * x[kk] + slow_y[ii] * y[kk]
                func += (np.exp(-1j * 2 * np.pi * dt * freqs))*(fft_st[kk])**2
            beam = func/nbeam
            fk[ii, jj] = np.vdot(beam, beam).real
            
    tracepower = np.vdot(fft_st**2, fft_st**2).real
    fk = nbeam * fk / tracepower
    # Find maximum
    fkmax = np.unravel_index(np.argmax(fk), (sinc, sinc)) 
    
    slow_x_max = slow_x[fkmax[1]]
    slow_y_max = slow_y[fkmax[0]]
    
    slowness = np.hypot(slow_x_max, slow_y_max)
    backazimuth = np.degrees(np.arctan2(slow_x_max, slow_y_max))
    
    if backazimuth < 0:
        backazimuth += 360.
    
    fig = plt.figure(figsize=(8, 6), tight_layout = True)
    fig.add_axes()
    plt.contourf(slow_x, slow_y, fk,32)
    plt.grid('on', linestyle='-')
    plt.xlabel('slowness east (s/km)')
    plt.ylabel('slowness north (s/km)')
    plt.colorbar(ticks=np.arange(0,1,0.1))
    plt.xlim(-smax, smax);
    plt.ylim(-smax, smax);  
    plt.title("FK Analysis, slowness= " + '%.4f' % slowness + " s/km,  backazimuth= " + '%.1f' % backazimuth + " deg")
    plt.show()
    fig.savefig('F-K Analysis.png',
                    bbox_inches="tight",dpi=fig.dpi)
    return

def main():
    fk(recordformat, coordfile, smax, fmin, fmax, tmin, tmax)

if __name__ == "__main__":
  main()
