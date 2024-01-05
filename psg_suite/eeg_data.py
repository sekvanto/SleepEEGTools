'''
Class for storing EEG Sleep data
'''
import numpy as np
import csv
import warnings
import pickle
import re
from timeit import default_timer as timer
from scipy import signal



'''
Class for storing EEG Sleep data
'''
import numpy as np
class EEGData():
    bitrate = None #Signal in range 0 to 2^bitrate
    n_electrodes = None #Number of electrodes
    sampling_rate = None #Sampling rate of the signal
    origin = None #Number around which the signal is centered, usually 0 or 2^(bitrate-1)
    standartized = None #False = signal range 0 to 2^bitrate, True = s. range -1 to 1
    data = None #Electrodes data
    def __init__(self):
        pass

    def load_raw(self, fname, n_electrodes=2, samp_rate=256,
                 bitrate=10, origin=512, standartized=False):
        """
        Loads EEG data from a raw, space separated format.

        Args:
            fname: Path to file to be loaded
            n_electrodes:  Number of electrode traces to be loaded
            samp_rate: Sampling rate of the recording device
            bitrate: Bitrate of the recording
            origin: Origin around which the signal is centered, usually 0 or 2^(bitrate-1)
            standartized: False = signal range 0 to 2^bitrate, True = s. range -1 to 1
        """
        try:
            with open(fname) as f:
                lines = f.read().splitlines()
                self.data = np.zeros((len(lines),n_electrodes),dtype=np.float64)
                n = 0
                for ln in lines:
                    strdata = ln.split(' ')
                    for elid in range(n_electrodes):
                        self.data[n][elid] = float(strdata[elid])
                    n += 1
        except Exception:
            self.data = None
            raise
        self.bitrate = bitrate
        self.n_electrodes = n_electrodes
        self.sampling_rate = samp_rate
        self.origin = origin
        self.standartized = standartized

    def load_openvibe(self, fname, n_electrodes=2, bitrate=10,
                      origin=512, standartized=False, delim=';'):
        """
        Loads EEG data from an OpenVibe file

        Args:
            fname: Path to file to be loaded
            n_electrodes:  Number of electrode traces to be loaded
            bitrate: Bitrate of the recording
            origin: Origin around which the signal is centered, usually 0 or 2^(bitrate-1)
            standartized: False = signal range 0 to 2^bitrate, True = s. range -1 to 1
            delim: Delimiter used to separate entries in the file
        """
        try:
            with open(fname) as f:
                for i, l in enumerate(f):
                    pass
            nlines = i
            with open(fname) as f:
                if nlines+1 < 2:
                    raise IOError("Invalid file format")
                self.data = np.zeros((nlines,n_electrodes),dtype=np.float64)
                n = 0;
                for ln in f:
                    ln=ln.rstrip()
                    if n == 0:
                        #print(ln)
                        header = ln.split(delim)
                        if header[0] != "Time (s)":
                            raise IOError("Invalid file format. First column should be Time.")
                        if header[-1] != "Sampling Rate":
                            raise IOError("Invalid file format. Last column "+
                                          "should be Sampling Rate.")
                        for i in range(1, len(header)-1):
                            if header[i] != "Channel " + str(i):
                                raise IOError("Invalid file format. Column " + str(i+1) +
                                              " should be Channel " + str(i))
                        if len(header) - 2 < n_electrodes:
                            n_electrodes = len(header) - 2
                            warnings.warn("Not enough electrode channels in the file. Only "
                                          + str(n_electrodes) + " will be read")
                        n+=1
                        continue
                    strdata = ln.split(delim)
                    if n == 1:
                        self.sampling_rate = int(strdata[-1])
                    for elid in range(n_electrodes):
                        self.data[n-1][elid] = float(strdata[elid+1])-512
                    n+=1
        except Exception:
            self.data = None
            self.sampling_rate = None
            raise
        self.bitrate = bitrate
        self.n_electrodes = n_electrodes
        self.origin = origin
        self.standartized = standartized

    def load_openbci(self, fname):
        """
        Loads EEG data from an OpenBCI file
        """
        try:
            self.load_pkl(fname + '.pickle')
            return
        except:
            pass
        try:
            with open(fname) as f:
                n_lines = 0
                channels = 4
                used_channels = [False for _ in range(16)]
                self.sampling_rate = 250.0
                self.origin = 0.0
                self.standartized = False
                self.bitrate = 'microvolts'
                for l in f:
                    l = l.strip().lower()
                    if l[0] == '%':
                        if 'channels' in l:
                            m = re.search(r'\b\d+\b', l)
                            if m is not None:
                                channels = int(m.group(0))
                        elif 'rate' in l:
                            m = re.search(r'\b\d+\.\d+\b', l)
                            if m is not None:
                                self.sampling_rate = float(m.group(0))
                    elif l != '':
                        n_lines += 1
                        for i, v in enumerate(l.split(',')[1: channels + 1]):
                            v = float(v.strip())
                            if abs(v) > 1e-3:
                                used_channels[i] = True
                used_channels = [i + 1 for i, v in enumerate(used_channels) if v]
                channels = len(used_channels)
                self.n_electrodes = channels
            with open(fname) as f:
                data = np.zeros((n_lines,channels),dtype=np.float64)
                n = 0;
                for l in f:
                    l = l.strip()
                    if l[0] == '%':
                        continue
                    l = l.split(',')
                    for i, j in enumerate(used_channels):
                       data[n][i] = float(l[j].strip())
                    n += 1
            self.data = data
        except Exception:
            self.data = None
            self.sampling_rate = None
            self.bitrate = None
            self.n_electrodes = None
            self.origin = None
            self.standartized = None
            raise
        try:
            self.save_pkl(fname + '.pickle')
        except:
            pass

    def load_pkl(self, fname):
        """
        Loads EEG data from pickle file

        Args:
            fname: Path to file to be loaded
        """
        start = timer()
        with open(fname,'rb') as f:
            ld = pickle.load(f)
            self.bitrate = ld.bitrate
            self.n_electrodes = ld.n_electrodes
            self.sampling_rate = ld.sampling_rate
            self.origin = ld.origin
            self.standartized = ld.standartized
            self.data = ld.data
        end = timer()
        print(fname + " unpickled in " + str(end - start))

    def save_pkl(self, fname):
        """
        Saves EEG data to pickle file

        Args:
            fname: Path to file to be saved
        """
        with open(fname,'wb') as f:
            pickle.dump(self,f)


    def standartize(self):
        """
        Standartizes the data by setting origin to 0 and range to -1 to 1
        """

        if self.standartized:
            return
        self.data = self.data - self.origin
        origin = 0
        self.data = self.data / np.power(2,self.bitrate - 1)
        self.standartized = True

    def sleep_duration(self):
        """
        Returns sleep duration in seconds
        """
        return self.data.shape[0]/self.sampling_rate

    def filter_notch(self, freq=50, width=1):
        b, a = signal.iirnotch(freq * 2 / self.sampling_rate, freq / width)
        self.data = signal.filtfilt(b, a, self.data, axis=0)

    def filter_freq(self, invert, order, flo, fhi=None):
        if fhi is None:
            b, a = signal.butter(order, flo * 2 / self.sampling_rate,
                                 'lowpass' if invert else 'highpass')
        else:
            b, a = signal.butter(order,
                                 tuple(i * 2 / self.sampling_rate for i in (flo, fhi)),
                                 'bandstop' if invert else 'bandpass')
        self.data = signal.filtfilt(b, a, self.data, axis=0)

    def filter_bandpass(self, flo=.1, fhi=40, order=3):
        self.filter_freq(False, order, flo, fhi)

    def filter_highpass(self, freq=.1, order=3):
        self.filter_freq(False, order, freq)

    def plot(self, elid=None, colors=None, legend=None, axes=None, title="EEG Waveform", figsize=(15,7), blocking=False):
        """
        Plots waveform into an axes provided for specified electrodes.

        Args:
            elid: Index of the electrode for which waveform will be plotted. Accepts iterables. None is "all".
            colors: plot colormap, None for default, otherwise an iterable with one color per electrode.
            legend: True to display x axis labels, false to hide, default true if more than one electrode.
            axes: matplotlib.axes.Axes object, None = Make new plot window
            title: Title of the figure when plotting standalone (axes=None)
            figsize: Size of the figure when plotting standalone (axes=None)
            blocking: True to block program execution, false to continue when plotting standalone (axes=None)
        """
        if elid is None:
            elid = tuple(range(self.n_electrodes))
        elif isinstance(elid, int):
            elid = (elid,)
        else:
            elid = tuple(elid)
        if legend is None:
            legend = len(elid) > 1

        if axes is None:
            fig=plt.figure(figsize=figsize)
            plt.title(title)
            axes = fig.axes[0]

        x_points = np.arange(len(self.data)) / self.sampling_rate / 60  # minutes
        axes.set(xlim=(0, x_points[-1]))

        #Plot the waveforms
        for i in elid:
            axes.plot(x_points, self.data[:,i], label="Ch{}".format(i))
        
        if legend:
            axes.legend(loc='upper right', bbox_to_anchor=(0, 1))

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
        return index / self.sampling_rate

