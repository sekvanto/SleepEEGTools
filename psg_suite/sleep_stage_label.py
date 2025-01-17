"""
Contains class for determing and storing sleep stage label data
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.table import Table
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import csv

class SleepStageLabel():
    """
    Class for determing and storing sleep stage label data
    """
    name = None
    sleep_block = None
    sleep_length = None
    date = None
    loaded_stage_times = None
    loaded_stage_labels = None
    stage_times = None
    stage_labels = None
    saving = False

    def __init__(self, name, date, sleep_block, sleep_length):
        """
        Initializes the stage label

        args:
            name: Name of the recording (usually filename)
            date: Date of the recording
            sleep_block: Identifier for the sleepblock in the recoring (use c1,c2,... for cores  and n1,n2,... for naps)
            sleep_length: Length of the sleep block in seconds
        """
        self.name = name
        self.date = date
        self.sleep_block = sleep_block
        self.sleep_length = sleep_length

    def label_manual(self, display_elems, figsize=(15, 7.5), title="Sleep Stages"):
        """
        Displays dialog for manual stage labeling

        args:
            displa_elems: list of lists of parameters used for display in format 
                          ((data_struct,{'parname1':parval1,'parname2':parval2,...}),...)
                          where data_struct is of class containing plot method (such 
                          as EEGSpectralData) and 'parname1':parval1,... are
                          name-value pairs supplied to the function.
            figsize: Size of the figure
            block: Blocks code execution until label dialog is closed
        """
        self.saving = False
        sleep_stage_labels = ['NREM3','NREM2','REM','NREM1','WAKE','MASK OFF','???']

        height_ratios = np.ones(len(display_elems))*3;
        height_ratios = np.append(height_ratios,1)
        fig=plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(len(display_elems)+1, 1, height_ratios=height_ratios)
        ax_transforms = {}
        def format_time_period(val):
            m, s = divmod(int(round(val)), 60)
            h, m = divmod(m, 60)
            return "%d:%02d:%02d" % (h, m, s)
        def reset_labels(event=None):
            self.stage_times = [0]
            self.stage_labels = [5]
            if event is not None:
                redraw_labels(event)
        def reload_labels(event=None):
            self.stage_times = np.append([], self.loaded_stage_times)
            self.stage_labels = np.append([], self.loaded_stage_labels)
            if self.stage_times[0] is None or self.stage_labels[0] is None or self.stage_times.size != self.stage_labels.size:
                print("data is bad, resetting labels")
                reset_labels(event)
            else:
                print("stage_times size is " + str(self.stage_times.size))
                print("stage_labels size is " + str(self.stage_labels.size))
            if event is not None:
                redraw_labels(event)

        def redraw_labels(event=None):
            # Debug: Print stage_times and stage_labels
            print("Debug - stage_times:", self.stage_times)
            print("Debug - stage_labels:", self.stage_labels)
            
            # Update line data for stages
            line1.set_xdata(np.concatenate((self.stage_times, [self.sleep_length])))
            line1.set_ydata(np.concatenate((self.stage_labels, [self.stage_labels[-1]])))
            
            # Clear previous filled areas to avoid overlap
            for coll in ax1.collections[:]:  # Iterate over a copy of the list
                coll.remove()

            # Define the colors for each stage
            stage_colors = {
                'MASK OFF': 'red',
                'WAKE': 'red',
                'NREM1': 'lightpink',
                'NREM2': 'lightblue',
                'NREM3': 'blue',
                'REM': 'green',
                '???': 'gray'
            }

            # Create a mask array of zeros with the same length as the stage_times array
            mask = np.zeros_like(self.stage_times)

            # Loop through each stage and fill the area below the line
            for stage_idx, stage in enumerate(sleep_stage_labels):
                # Fill areas where the stage is present
                mask[:] = np.array(self.stage_labels) == stage_idx
                if np.any(mask):
                    ax1.fill_between(
                        np.concatenate((self.stage_times, [self.sleep_length])),
                        0,  # Start filling from 0
                        np.concatenate((mask, [mask[-1]])) * (stage_idx + 1),  # Multiply mask by stage index to shift up
                        step='post',
                        color=stage_colors[stage]
                    )

            # Initialize an array to store the duration of each stage
            stage_durations = [0] * len(sleep_stage_labels)
            excluded_stages = {'WAKE', '???', 'MASK OFF'}  # Define stages to exclude from total sleep time
            
            # Calculate durations for each stage
            for i in range(len(self.stage_times) - 1):
                duration = self.stage_times[i + 1] - self.stage_times[i]
                stage_durations[int(self.stage_labels[i])] += duration

            # Calculate the total sleep time excluding the durations of WAKE, ???, and MASK OFF
            total_sleep_time = self.sleep_length - sum(stage_durations[sleep_stage_labels.index(stage)] for stage in excluded_stages if stage in sleep_stage_labels)

            # Update the table with the calculated durations
            for i, duration in enumerate(stage_durations):
                formatted_duration = format_time_period(duration)
                print(f"Debug - Updating table cell with label {sleep_stage_labels[i]} with duration {formatted_duration}")
                table.get_celld()[(len(sleep_stage_labels) - 1 - i, 1)].get_text().set_text(formatted_duration)

            # Update the total sleep time cell
            print(f"Debug - Adjusted Total Sleep Time: {format_time_period(total_sleep_time)}")
            table.get_celld()[(len(stage_durations), 1)].get_text().set_text(format_time_period(total_sleep_time))

            # Redraw the canvas after making changes
            fig.canvas.draw()

        def on_pick(figure, mouseevent):
            #print(ax_transforms)
            #print(mouseevent)
            #print(figure)
            #print(figure.get_axes())
            if not mouseevent.inaxes in ax_transforms:
                return False, {}
            eegdata = ax_transforms[mouseevent.inaxes]
            xmouse, ymouse = mouseevent.xdata, mouseevent.ydata
            xmouse = eegdata.index_to_time(xmouse)
            #print('x, y of mouse: {:.2f},{:.2f}'.format(xmouse, ymouse))
            larger=[x[0] for x in enumerate(self.stage_times) if x[1] > xmouse]
            if len(larger)>0:
                idx = larger[0]
                if self.stage_labels[idx-1] != self.stage_label:
                    self.stage_times[idx] = xmouse
                    self.stage_labels[idx] = self.stage_label
                else:
                    self.stage_times = np.delete(self.stage_times, idx)
                    self.stage_labels = np.delete(self.stage_labels, idx)
            else:
                if self.stage_labels[-1] != self.stage_label:
                    self.stage_times = np.append(self.stage_times, xmouse)
                    self.stage_labels = np.append(self.stage_labels, self.stage_label)
            #print(stage_times)
            #print(stage_labels)
            for i in range(1, len(self.stage_labels) - 1):
                if self.stage_labels[i] == self.stage_labels[i-1]:
                    self.stage_labels = np.delete(self.stage_labels, i)
                    self.stage_times = np.delete(self.stage_times, i)
            redraw_labels()
            return True, {}

        
        for did in range(len(display_elems)):
            delem = display_elems[did]
            ax = plt.subplot(gs[did])
            ax_transforms[ax] = delem[0]
            params = delem[1]
            params['axes']=ax
            delem[0].plot(**params)
            if did == 0:
                plt.title(title)
            #TODO
            ax.set_picker(True)
            

            fig.canvas.mpl_connect('pick_event', on_pick)
            fig.set_picker(on_pick)

        xtickspacing = 300;
        if len(np.arange(0,self.sleep_length,300)) > 20:
            xtickspacing = 600;
        if len(np.arange(0,self.sleep_length,600)) > 20:
            xtickspacing = 1200;
        if len(np.arange(0,self.sleep_length,1200)) > 20:
            xtickspacing = 1800;
        if len(np.arange(0,self.sleep_length,1800)) > 20:
            xtickspacing = 3600;
        xticks = np.arange(0,self.sleep_length,xtickspacing)
        xticklabels = [str(int(i/60)) for i in xticks]

        reload_labels()
        ax1 = plt.subplot(gs[-1])
        line1,=ax1.plot(np.concatenate((self.stage_times,[self.sleep_length])), 
                        np.concatenate((self.stage_labels,[self.stage_labels[-1]])),drawstyle="steps-post")
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("Sleep Stage")
        ax1.set_xlim(0,self.sleep_length)
        ax1.set_yticks(np.arange(7))
        ax1.set_yticklabels(sleep_stage_labels)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xticklabels)

        self.stage_label = 6
        rax = plt.axes([0.0, 0.0, 0.2, 0.16], facecolor='lightgoldenrodyellow')
        radio = RadioButtons(rax, sleep_stage_labels[::-1], active=0)
        def stagepicker(label):
            self.stage_label = sleep_stage_labels.index(label)
            #print(plot_eeg_log_hist.stage_label)
            #fig.canvas.draw_idle()
        def done(event):
            self.saving = True
            plt.close()
        if self.loaded_stage_labels is not None:
            axreload = plt.axes([0.7, 0.0, 0.1, 0.075])
            breload = Button(axreload, 'Reload')
            breload.on_clicked(reload_labels)
        axreset = plt.axes([0.8, 0.0, 0.1, 0.075])
        breset = Button(axreset, 'Reset')
        breset.on_clicked(reset_labels)
        axdone = plt.axes([0.9, 0.0, 0.1, 0.075])
        bdone = Button(axdone, 'Save &\n Quit')
        bdone.on_clicked(done)
        radio.on_clicked(stagepicker)
        tableax = plt.axes([0.2, 0.0, 0.25, 0.16], facecolor='lightblue')
        tableax.get_yaxis().set_visible(False)
        table = Table(tableax, bbox=[0,0,1,1])
        height = table._approx_text_height()
        lidx = 0
        for label in sleep_stage_labels[::-1]:
            table.add_cell(lidx, 0, width=0.6, height=height, text=label)
            table.add_cell(lidx, 1, width=0.4, height=height, text='')
            lidx = lidx + 1
        table.add_cell(lidx, 0, width=0.6, height=height, text='Total Sleep Time')
        table.add_cell(lidx, 1, width=0.4, height=height, text='')
        tableax.add_table(table)
        
        fig.canvas.callbacks.connect('pick_event', on_pick)
        ##fig.canvas.set_window_title('EEG Spectrogram Analysis')

        plt.subplots_adjust(left=0.15 if figsize[0] < 10 else 0.075, bottom=0.2, right=0.99, top=0.97)
        redraw_labels()
        plt.show()
        self.stage_times = np.array(self.stage_times)
        self.stage_times = np.concatenate((self.stage_times, [self.sleep_length]))
        self.stage_labels = np.concatenate((self.stage_labels, [6]))

    
    def load_txt(self, fname):
        """
        Loads sleep label data in text fromat from provided file

        Args:
            fname: Path to the file to be loaded
        """
        with open(fname) as f:
            self.name = f.readline().rstrip('\n')
            self.sleep_block = f.readline().rstrip('\n')
            self.sleep_length = float(f.readline().rstrip('\n'))
            self.date = f.readline().rstrip('\n')

            reader = csv.reader(f)
            data = np.asfarray(np.array(list(reader)),float)
            self.loaded_stage_times = self.stage_times = data[:,0]
            self.loaded_stage_labels = self.stage_labels = data[:,1].astype(int)
 
    def save_txt(self,fname):
        """
        Saves sleep label data in text fromat to provided file

        Args:
            fname: Path to the file to be saved
        """
        with open(fname,'w') as wrf:
            wrf.write(str(self.name)+"\n")
            wrf.write(str(self.sleep_block)+"\n")
            wrf.write(str(self.sleep_length)+"\n")
            wrf.write(str(self.date)+"\n")

            wr = csv.writer(wrf, delimiter=',', lineterminator='\n')
            for i in range(len(self.stage_times)):
                wr.writerow([self.stage_times[i],self.stage_labels[i]])
