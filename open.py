#!/usr/bin/env python3
import numpy as np
import csv
import math
import tkinter
import tkinter.filedialog
import sys
import os

from psg_suite.eeg_data import EEGData
from psg_suite.eeg_spectrum import EEGSpectralData
from psg_suite.sleep_stage_label import SleepStageLabel

def mf():
    ftype = None  # Initialize ftype to None at the start of the function

    
    if len(sys.argv) > 3:
        print("Usage: {} [[format] filename]".format(sys.argv[0]))
        return
    if len(sys.argv) == 3:
        ftype = sys.argv[1].lower()
        fname = sys.argv[2]
        if ftype not in ('openvibe', 'raw', 'openbci'):
            print("Unknown format {}. Supported: OpenVIBE, RAW, OpenBCI.".format(ftype))
            return
    elif len(sys.argv) == 2:
        ftype = None
        fname = sys.argv[1]
    else:
        root_window = tkinter.Tk()
        root_window.withdraw()
        fname = tkinter.filedialog.askopenfilename(filetypes=[('All Supported Files (*.CSV, *.OVIBE, *.DAT, *.TXT)',('.csv','.ovibe','.dat','.txt')),('OpenVIBE CSV (*.CSV, *.OPENVIBE)',('.csv','.openvibe')),('Raw (*.DAT)','.dat'),('OpenBCI CSV (*.TXT)','.txt'),('All Files (*.*)','.*')])
        root_window.destroy()
        if fname == "":
            print("No file was selected - aborting")
            return
    if ftype is None:
        ftype = fname.lower().rsplit('.', 1)[-1]
        if ftype in ("csv", "ovibe"):
            print("Assuming OpenVIBE format")
            ftype = 'openvibe'
        elif ftype in ('dat',):
            print("Assuming raw format")
            ftype = 'raw'
        elif ftype in ('txt',):
            print("Assuming OpenBCI format")
            ftype = 'openbci'
        else:
            print("Could not guess format from file extension. Specify it on command line or rename your file.")
            return
    
    print("---------------------------------------------------------")
    data = EEGData()
    if ftype == 'openvibe':
        print("Loading OpenVIBE capture data from: {} ...".format(fname))
        data.load_openvibe(fname)
    elif ftype == 'raw':
        print("Loading raw capture data from: {} ...".format(fname))
        data.load_raw(fname)
    elif ftype == 'openbci':
        print("Loading OpenBCI capture data from: {} ...".format(fname))
        data.load_openbci(fname)
    else:
        print("Unknown capture format!")
        return
    print(data.data)
    length = len(data.data)
    print("---------------------------------------------------------")
    minutes = int(math.ceil(data.sleep_duration()/60))
    print("data size: " + str(length) + " (" + str(minutes) + " minutes)")
    print("data shape: " + str(np.shape(data)))
    #eeg_data_visual.plot_eeg_data(data)
    data.filter_notch(50)
    data.filter_highpass(1/60)
    #data.filter_bandpass(1, 30, 6)
    #data.lowpass(14)
    spectrum = EEGSpectralData(data)
    print("hist shape: " + str(np.shape(spectrum.data)))
    print("freqs shape: " + str(np.shape(spectrum.frequencystamps)))
    print("max: " + str(np.max(np.log(spectrum.data))))
    print("min: " + str(np.min(np.log(spectrum.data))))
    print("ptp: " + str(np.ptp(np.log(spectrum.data))))
    print("---------------------------------------------------------")
    print("Displaying spectrograms ...")
    figwidth = 7 if minutes < 40 else 16 # adjust figure size based on number of minutes in data set so that naps get a smaller display thats easier to read
    title = os.path.basename(fname).rsplit('.', 1)[0]
    spectrum.frequency_cutoff(30, 0)
    sleep_labels = SleepStageLabel(title,"","",data.sleep_duration())
    stages_file = fname + '.stages'
    if os.path.isfile(stages_file):
        print("Loading existing stage data ...")
        sleep_labels.load_txt(stages_file)
    else:
        print("No existing stage data found.")
    sleep_labels.label_manual(((data,{}),) + tuple((spectrum,{"elid":i,"xlabels":False}) for i in range(spectrum.n_electrodes)),title=title,figsize=(figwidth, 9))
    if sleep_labels.saving:
        print("Saving stage data ...")
        sleep_labels.save_txt(stages_file)
    else:
        print("Stage data will not be saved")
    print("Done")
if __name__ == '__main__':
    mf()
