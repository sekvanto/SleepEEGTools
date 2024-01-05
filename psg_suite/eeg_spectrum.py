'''
Contains class representing EEG Spectral data
'''
import numpy as np
import collections
from  spectrum import *
import matplotlib.pyplot as plt
from . import plotting_util
import pickle
from timeit import default_timer as timer

class EEGSpectralData():
    """
    Handles computation, manipulation and analysis of 
    EEG spectral data
    """
    timestamps = None
    sapmlestamps = None
    frequencystamps = None
    window = None
    step = None
    n_electrodes = None
    sampling_rate = None
    data = None

    def __init__(self, eegdata=None, n_electrodes=None, window=None, step=None, desired_len=3000, desired_overlap=1.6, downsample=None, desired_fstep=.1):
        """
        Uses raw EEG data to create frequency power distribution histogram

        Args:
            eegdata: instance of EEGData, only supply none if you intend to load from file
            n_electrodes: N. of electrodes to be used for computation
            window: N. of samples in sliding window used to estiamte the power at given time point
            step: N. of samples between individual power distribution estimations
            downsample: Downsampling power in frequency dimension (1=no downsampling)
        """
        if eegdata is None:
            return
        if n_electrodes is None:
            n_electrodes = eegdata.n_electrodes
        if step is None:
            step = eegdata.data.shape[0] // desired_len
        if window is None:
            window = 1
            while window < step * desired_overlap:
                window *= 2
        self.n_electrodes = n_electrodes
        self.sampling_rate = eegdata.sampling_rate
        self.step =step
        self.frequencystamps = np.arange(int(window/2)+1)/(int(window/2)) * self.sampling_rate/2
        if downsample is None:
            downsample = max(1, int((len(self.frequencystamps) - 1) * desired_fstep / (self.frequencystamps[-1] - self.frequencystamps[0])))
        self.frequencystamps = self.frequencystamps[:len(self.frequencystamps) // downsample * downsample:downsample]
        self.data = np.zeros((n_electrodes,len(range(window,eegdata.data.shape[0],step)),len(self.frequencystamps)),dtype=np.float64)
        for el in range(n_electrodes):
            n = 0;
            for d in range(window,eegdata.data.shape[0],step):
                w = eegdata.data[(d-window):d,el]
                w = w.flatten()
                p = speriodogram(w, detrend=False, sampling=eegdata.sampling_rate)
                p = p[:len(p) - len(p) % downsample]
                for i in range(downsample):
                    self.data[el,n,:] += p[i::downsample]
                n+=1
        self.samplestamps = np.arange(window,eegdata.data.shape[0],step);
        self.timestamps = self.samplestamps/self.sampling_rate


    def frequency_cutoff(self, cutoff=45, low=1):
        """
        Reduces the histogram data by cutting off frequencies higher than specifed frequency
        
        Args:
            cutoff: freqency above which the histogram data will be removed
            low   : freqency below which the histogram data will be removed
        """
        ci = np.argmin(self.frequencystamps<=cutoff)
        li = np.argmax(self.frequencystamps>=low)
        self.data = self.data[:,:,li:ci+1]
        self.frequencystamps = self.frequencystamps[li:ci+1]
        
    def plot(self, elid=0, colormap="cold_black_hot_symmetry_green", vmin=None, vmax=None, xlabels=True, axes=None, title="EEG Spectrogram", figsize=(15,7), blocking=False, normalize=True, interpolation='hanning'):
        """
        Plots sperctrogram into an axes provided for desired electrode.

        Args:
            elid: Index of the electrode for which spectrogram will be plotted
            colormap: plot colormap (parula by default)
            vmin: Value of log hist which is used for the lowest color, default is np.min(dt)+0.43*np.ptp(dt)
            vmax: Value of log hist which is used for the higest color, default is np.max(dt)-0.03*np.ptp(dt)
            xlabels: True to display x axis label, false to hide
            axes: matplotlib.axes.Axes object, None = Make new plot window
            title: Title of the figure when plotting standalone (axes=None)
            figsize: Size of the figure when plotting standalone (axes=None)
            blocking: True to block program execution, false to continue when plotting standalone (axes=None)
        """
        dt = self.data[elid]
        if normalize:
            x = dt[:,np.argmax(self.frequencystamps>=1):np.argmin(self.frequencystamps<=17)].flatten()
            x.sort()
            vmin = 0
            vmax = len(x) - 1
            dt = np.interp(dt, x, np.arange(len(x)))
        else:
            #Log histogram for better visual interpretation
            dt = np.log(dt)
            #Determine vmin/vmax values
            if vmin is None:
                vmin = np.min(dt)+0.43*np.ptp(dt)
            if vmax is None:
                vmax = np.max(dt)-0.03*np.ptp(dt)

        if axes is None:
            fig=plt.figure(figsize=figsize)
            plt.title(title)
            axes = fig.axes[0]

        step = np.argmax(self.frequencystamps>5)
        if step <= 1:  # Check if the step is less than or equal to 1
            step = 2   # Set a default step value (you can adjust this as needed)


        #Calculate Y axis labels
        yticks = np.arange(0,len(self.frequencystamps), step-1)
        yticklabels = ["{:6.2f}".format(i) for i in self.frequencystamps[yticks]]

        #Calculate X axis labels
        sleep_dur = self.timestamps[-1]
        xtickspacing = 300;
        if len(np.arange(0,sleep_dur,300)) > 20:
            xtickspacing = 600;
        if len(np.arange(0,sleep_dur,600)) > 20:
            xtickspacing = 1200;
        if len(np.arange(0,sleep_dur,1200)) > 20:
            xtickspacing = 1800;
        if len(np.arange(0,sleep_dur,1800)) > 20:
            xtickspacing = 3600;
        xticks = np.arange(0,sleep_dur,xtickspacing)
        xticklabels = [str(int(i/60)) for i in xticks]
        xticks = xticks/(self.step/self.sampling_rate)

        #Plot the histogram
        axes.set_yticks(yticks)
        axes.set_yticklabels(yticklabels)
        axes.set_xticks(xticks)
        axes.set_xticklabels(xticklabels)
        axes.imshow(np.transpose(dt), origin="lower", aspect="auto", 
                cmap=plotting_util.colormap(colormap), interpolation=interpolation,vmin=vmin,vmax=vmax)
        axes.set_ylabel("Frequency (Hz)")
        if xlabels:
            axes.set_xlabel("Time (min)")

        if axes is None:
            if blocking:
                plt.show()
            else:
                plt.draw()


    def index_to_time(self, index):
        """
        Transforms index to time in seconds.

        Args:
            index: histogram array x index

        Returns:
            time: Time associated with provided index in seconds
        """
        return index*self.step/self.sampling_rate

    def load_pkl(self, fname):
        """
        Loads EEG spectral data data from pickle file

        Args:
            fname: Path to file to be loaded
        """
        start = timer()
        with open(fname,'rb') as f:
            ld = pickle.load(f)
            self.timestamps = ld.timestamps
            self.samplestamps = ld.samplestamps
            self.frequencystmps = ld.frequencystamps
            self.window = ld.window
            self.step = ld.step
            self.n_electrodes = ld.n_electrodes
            self.sampling_rate = ld.sampling_rate
            self.data = ld.data
        end = timer()
        print(fname + " unpickled in " + str(end - start))

    def save_pkl(self, fname):
        """
        Saves EEG spectral data data to pickle file

        Args:
            fname: Path to file to be saved
        """
        with open(fname,'wb') as f:
            pickle.dump(self,f)


