import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from scipy.signal import butter, filtfilt
from scipy.stats import linregress, zscore
from scipy.optimize import curve_fit, minimize
from datetime import datetime, timedelta
import os
import matplotlib as mpl
import time

#FIXME list
#DONE save_path generation and usage should be refactored
#Events handling is confusing and probably unnecessary complicated (see issue #4 on repo)
    #the current example notebook and batch notebook does not use extract_events
    #remove extract_events function as not needed
    #remove events from create_basic as not needed? waiting for Hilde's input 


class preprocess:
    def __init__(self, path, sensors):
        """
        Initializes the preprocess class with the given path and sensors.
        Parameters:
        path (str): The path to the root_data directory (contains all acquired data)
        sensors (list): List of sensors used in the experiment.
        """
        self.path = os.path.join(path, '')
        
         # Create save path one level up with _processedData appended
        parent_dir = os.path.dirname(os.path.dirname(self.path))
        folder_name = os.path.basename(os.path.dirname(self.path)) + "_processedData/photometry"
        self.save_path = os.path.join(parent_dir, folder_name)
        # Create save directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)
        
        # Update path to point to photometry data folder
        self.path = os.path.join(path, 'photometry', '')
        self.sensors = sensors
        self.colors = ['mediumseagreen', 'indianred', 'mediumpurple', 'steelblue']


    def show_structure(self):
        """
        Display readable structure of photometry object
        Useful for debugging and understanding the object's contents
        """
        # Get attributes without dunders
        attributes = [attr for attr in dir(self) if not attr.startswith('__')]
        
        # Organize by type
        dataframes = []
        methods = []
        other = []
        
        for attr in attributes:
            if isinstance(getattr(self, attr), pd.DataFrame):
                dataframes.append(attr)
            elif callable(getattr(self, attr)):
                methods.append(attr)
            else:
                other.append(attr)
                
        # Print organized structure
        print("\n=== DataFrames ===")
        for df in sorted(dataframes):
            shape = getattr(self, df).shape
            print(f"{df}: {shape[0]} rows × {shape[1]} columns")
            
        print("\n=== Properties ===")
        for prop in sorted(other):
            print(f"{prop}: {type(getattr(self, prop)).__name__}")
            
        print("\n=== Methods ===")
        print(", ".join(sorted(methods)))
    
    
    def get_info(self):
        """
        Reads the Fluorescence.csv file and extracts experiment information.
        Returns:
        dict: A dictionary containing experiment information.
        """
        with open(os.path.join(self.path, 'Fluorescence.csv'), newline='') as f:
            reader = csv.reader(f)
            row1 = next(reader)

        info_str = str(row1[0]).replace(";", ",").replace('true', 'True').replace("false", "False")
        info = eval(info_str)

        return info


    def create_basic(self, cutend = False, cutstart = False, target_area = 'X'):
        '''
        The code assumes 470 nm channel always has data and does the following.
        0) loads Events.csv and Fluorescence-unaligned.csv
        1) aligns events and all fluorescence to the 470 nm channel in one dataframe
        2) if cutstart = True, removes first 15 seconds
        3) creates a new TimeStamp column in seconds rather than ms 
        4) creates a df with the data from all recorded wavelengths
        5) use the path to make a new path to the output data from the preprocessing.
        6) adds mousename, target_area and experiment type to info
        :param: cutend: default set to False, removes last 300 seconds, e.g. if you know that you forgot to turn off the equipment before removing fiber
        :param: cutstart: default set to False,removes first 15 seconds
        :param: target_area: default set to 'X', can be changed to the target area of the experiment
        :return: rawdata, data (cut at start and end if needed), data_seconds (timestamp), signals (main output df), save_path
        FIXME unclear why the events file is even read as it is not saved at the end
        '''
        
        info = self.info
        event_path = self.path + 'Events.csv'  # events with precise timestamps
        fluorescence_path = self.path + 'Fluorescence-unaligned.csv'  # fluorescence with precise timestamps

        # get parameters from path name
        mousename = self.path.split('/')[-3][:7]
        session = self.path.split('/')[-3][8:]
        if '&' in session:
            session = session.replace('&', '-') 
        experiments = self.path.split('/')[-4][:]
               
        #Adding mousename, target_area and experiment type to info  
        if 'mousename' not in self.info:
            self.info['mousename'] = ()
        self.info['mousename']= mousename
        if 'target_area' not in self.info:
            self.info['target_area'] = ()
        self.info['target_area'] = target_area
        if 'experiment_type' not in self.info:
            self.info['experiment_type'] = ()
        self.info['experiment_type'] = experiments
        
        #Adding mousename in self for plotting and naming (could use self.info['mousename'] instead)
        self.mousename = mousename
        
        #print the mousename and session to output, epsecially usefull for batch processing 
        print(f'\n\033[1mPreprocessing data for {mousename} in session {session}...\033[0m\n')
        
        #reading the csv files into pandas dataframes
        events = pd.read_csv(event_path) 
        fluorescence = pd.read_csv(fluorescence_path)
        
        #rename the 'Name' column in events to 'Event'
        events.rename(columns={'Name': 'Event'}, inplace=True)
        
        # Create and fill separate dataframes for each wavelength
        df_470 = fluorescence[fluorescence['Lights'] == 470][['TimeStamp', 'Channel1']].rename(columns={'Channel1': '470'})
        df_410 = fluorescence[fluorescence['Lights'] == 410][['TimeStamp', 'Channel1']].rename(columns={'Channel1': '410'})
        df_560 = fluorescence[fluorescence['Lights'] == 560][['TimeStamp', 'Channel1']].rename(columns={'Channel1': '560'})

        # Merge the '410' and '560' dataframes with the '470' dataframe based on nearest timestamps 
        # All will be shifted to match the 470 nm signal
        df_final = pd.merge_asof(df_470, df_410, on='TimeStamp', direction='backward')
        df_final = pd.merge_asof(df_final, df_560, on='TimeStamp', direction='backward')

        # Fill nan values or handle missing data
        # first forward filling and then backwards filling in case of nans at the end of the columns
        df_final['410'] = df_final['410'].ffill().bfill()
        df_final['560'] = df_final['560'].ffill().bfill()

        #merge events and fluorescence
        #is merged to match the 470 signal timestamps as closely as possible,
        # meaning that the event timestamps may be slightly shifted
        # FIXME in practice, Events.csv saved at the end is exactly the same as the original Events.csv file, so unsure if pd.merge.asof is necessary
        if len(events) < 1:
            print("WARNING: No events were found. Check for missing sync signal.")
            rawdata = df_final
            rawdata = rawdata.loc[:, ~rawdata.columns.str.contains('^Unnamed')]
            #data = rawdata[rawdata["TimeStamp"] > 30] FIXME if this line is needed, weird magic number plus there should always be events 
            data = rawdata
        else:
            rawdata = pd.merge_asof(df_final, events[['TimeStamp', 'Event', 'State']], on='TimeStamp', direction='backward') #FIXME this is not saved in the Events.csv file
            rawdata = rawdata.loc[:,~rawdata.columns.str.contains('^Unnamed')]  # sometimes an Unnamed column has appeared...
            # removing first 15 seconds because of bleaching
            if cutstart:
                # Remove initial bleaching period (first 15 seconds)
                data = rawdata[rawdata["TimeStamp"] > 15000]  
            else:
                data = rawdata
        if cutend == True:
            data = data.drop(data.tail(300).index) #This can be done if fiber was by mistake removed before NOTE why 300?
        
        data_seconds = pd.DataFrame([second for second in data['TimeStamp'] / 1000], columns=['TimeStamp'])
        signals = pd.DataFrame()
        if info['Light']['Led470Enable'] == True:
            signals['470'] = data['470']
        if info['Light']['Led560Enable'] == True:
            signals['560'] = data['560']
        if info['Light']['Led410Enable'] == True:
            signals['410'] = data['410']

        #Adding events as booleans: Create a new column for each unique event in the 'Name' column
        unique_events = events['Event'].unique()
        data = data.copy()
        for event in unique_events:
            # Initialize the event-specific column with False, using loc to avoid SettingWithCopyWarning
            data.loc[:, f"{event}_event"] = False
            
            # Filter the events for this specific event name
            event_rows = events[events['Event'] == event]
            
            #Get the transitions from 0 to 1
            transitions = event_rows[(event_rows['State'].shift(1) == 0) & (event_rows['State'] == 1)]

            #Get the timestamps for both 0 and 1 states
            start_timestamps = event_rows[event_rows['State'] == 0].loc[event_rows['State'].shift(-1) == 1, 'TimeStamp'].values
            end_timestamps = event_rows[event_rows['State'] == 1].loc[event_rows['State'].shift(1) == 0, 'TimeStamp'].values

            # For each time range, modify the corresponding values in another DataFrame (e.g., 'other_df')
            for start, end in zip(start_timestamps, end_timestamps):
                mask = (data['TimeStamp'] >= start) & (data['TimeStamp'] <= end)
                data.loc[mask, f"{event}_event"] = True 
 
        return(rawdata, data, data_seconds, signals)
        
 
    def extract_events(self): #FIXME any use for this?    
        """
        Assigns a boolean data column for each unique event type indicating 
        whether the event occurred at each time point (True/False). 
        Removes 'Event' and 'State' columns after processing.
        Saves the updated data DataFrame to self.data.
        
        Returns:
            DataFrame with 'Event' and 'State' columns (original values) before removal.
        """
        # Copy the data for processing
        data = self.data.copy()
    
        # List to hold unique event names for reference (optional)
        events = []
    
        # Check if the Event column exists and is not empty
        if 'Event' not in data.columns or data['Event'].isna().all():
            print("There are no recorded events.")

        else:
            events = pd.DataFrame()
            for col in data.columns:
                if 'event' in col:
                    events[col]= data[col]
            
        return events#data[[event for event in events]]


    def low_pass_filt(self, method = "auto", plot=False, x_start=None, x_end=None, savefig = False):
        """
        Apply low-pass filter to signals.
        
        Parameters:
        plot (bool): Whether to plot the filtered signals
        method (str): Method to determine cutoff frequency. Options: 'auto', 'sensor'
        x_start (float, optional): Start time for x-axis in seconds. If None, uses minimum time.
        x_end (float, optional): End time for x-axis in seconds. If None, uses maximum time.
        """
        start_time = time.time()
        
        signals = self.signals
        sensors = self.sensors
        filtered = pd.DataFrame(index=self.signals.index)
        
        # Set x-axis limits
        if x_start is None:
            x_start = self.data_seconds['TimeStamp'].min()
        if x_end is None:
            x_end = self.data_seconds['TimeStamp'].max()
        
        # The RWD software records the Fps for all recorded channels combined
        sample_rate = self.info['Fps'] / len(sensors) #get sample_rate per channel
        
        # Add keys if they don't exist
        if 'filtering_Wn' not in self.info:
            self.info['filtering_Wn'] = {}
        if 'filtering_method' not in self.info:
            self.info['filtering_method'] = {}
        
        for signal in signals:
            sensor = sensors[signal]
            
            # Determine cutoff frequency and filter order
            if method == 'auto':
                Wn = np.floor(sample_rate / 2.01)  # ~Nyquist frequency
                self.info['filtering_method'][signal] = 'auto'
            elif method == 'sensor': 
                if sensor == 'G8m':
                    Wn = 10
                elif sensor == 'rG1':
                    Wn = 10 
                elif sensor == 'g5-HT3':
                    Wn = 5
                elif isinstance(sensor, int):
                    Wn = 100 / (sensor / 3)
                else:
                    Wn = 1000 / (int(input("Enter sensor half decay time: ")) / 3)
                self.info['filtering_method'][signal] = 'sensor_specific'
                
            print(f'Filtering {signal} with method {method} at {Wn} Hz')
            
            # Store the Wn value for this signal in 'filtering_Wn'
            self.info['filtering_Wn'][signal] = [Wn]
            
            try:
                # Design the filter
                b, a = butter(2, Wn, btype='low', fs=sample_rate)
                # Apply the filter
                filtered[f'filtered_{signal}'] = filtfilt(b, a, signals[signal])
                self.info['filtering_Wn'][signal].append(True)
            
            except ValueError:
                print(f'Wn is set to {Wn} for the sensor {sensor}, but samplig rate {sample_rate} Hz is less than 2 * {Wn}. No filtering done.')
                self.info['filtering_Wn'][signal].append(False)
                # Copy the unfiltered signal to the filtered DataFrame
                filtered[f'filtered_{signal}'] = signals[signal]
        
        if plot:
            num_signals = len(signals.columns)
            fig, axes = plt.subplots(num_signals, 1, figsize=(12, 8), sharex=True)
            
            # If there's only one signal, `axes` won't be a list
            if num_signals == 1:
                axes = [axes]
            
            for ax, signal, color in zip(axes, signals, self.colors):
                # Create mask using data index
                mask = (self.data_seconds.index >= self.data_seconds[self.data_seconds['TimeStamp'] >= x_start].index[0]) & \
                    (self.data_seconds.index <= self.data_seconds[self.data_seconds['TimeStamp'] <= x_end].index[-1])
                
                # Plot data
                ax.plot(self.data_seconds['TimeStamp'], signals[signal], label='Original', color=color, alpha=1, linewidth=1)
                ax.plot(self.data_seconds['TimeStamp'], filtered[f'filtered_{signal}'], label='Filtered', color="black", linewidth=0.5, alpha=0.5)
                
                # Set x limits
                ax.set_xlim([x_start, x_end])
                
                # Set y limits based on visible range
                visible_orig = signals[signal][mask]
                visible_filt = filtered[f'filtered_{signal}'][mask]
                y_min = min(visible_orig.min(), visible_filt.min())
                y_max = max(visible_orig.max(), visible_filt.max())
                ax.set_ylim([y_min, y_max])
                
                ax.set_title(f'Signal: {signal}')
                ax.set_ylabel('fluorescence')
                ax.legend()
            
            axes[-1].set_xlabel('Seconds')
            fig.suptitle(f'Low-pass Filtered {self.mousename} with method: {method}')
            plt.tight_layout()
            
             # Save plot to file
            if savefig:
                fig.savefig(self.save_path + f'/low-pass_filtered_{self.mousename}.png', dpi=150)  # Lower DPI to save time
            
            plt.show()
            plt.close(fig)
            
        return filtered    


    def detrend(self, plot=False, method='divisive', savefig = False):
        try:
            traces = self.filtered
        except:
            print('No filtered signal was found')
            traces = self.signals
            
        data_detrended = pd.DataFrame(index=traces.index)  # Initialize with proper index
        exp_fits = pd.DataFrame()
        
        if 'detrend_params' not in self.info:
            self.info['detrend_params'] = {}   
        if 'detrend_method' not in self.info:
            self.info['detrend_method'] = {}
        self.info['detrend_method'] = method
            
        def double_exponential(time, amp_const, amp_fast, amp_slow, tau_slow, tau_multiplier):
            '''
            Based on: https://github.com/ThomasAkam/photometry_preprocessing
            Compute a double exponential function with constant offset.
            Parameters:
            t       : Time vector in seconds.
            const   : Amplitude of the constant offset.
            amp_fast: Amplitude of the fast component.
            amp_slow: Amplitude of the slow component.
            tau_slow: Time constant of slow component in seconds.
            tau_multiplier: Time constant of fast component relative to slow.
            '''
            tau_fast = tau_slow * tau_multiplier
            return amp_const + amp_slow * np.exp(-time / tau_slow) + amp_fast * np.exp(-time / tau_fast)

        # Fit curve to signal.
        for trace in traces:
            max_sig = np.max(traces[trace])
            min_sig = np.min(traces[trace])
            inital_params = [max_sig*0.5, max_sig*0.1, min_sig*0.8, 1, 0.1]
            bounds = ([0, 0, 0, 0, 0],
                  [max_sig, max_sig, max_sig, 36000, 1])
            try:
                signal_parms, _ = curve_fit(double_exponential, self.data_seconds['TimeStamp'], traces[trace], p0=inital_params, bounds=bounds, maxfev=1000) # parm_cov is not used
            except RuntimeError:
                print('Could not fit exponential fit for: \n', self.path)
                pass
            signal_expfit = double_exponential(self.data_seconds['TimeStamp'], *signal_parms)
            print(f'Detrending {trace} with params: [{", ".join(f"{x:.3g}" for x in signal_parms)}]') # prints to 3 significant digits 
            self.info['detrend_params'][trace]=signal_parms
            if method == "subtractive":
                signal_detrended = traces[trace].reset_index(drop=True) - signal_expfit.reset_index(drop=True)
            if method == "divisive":
                signal_detrended = traces[trace].reset_index(drop=True) / signal_expfit.reset_index(drop=True)
            # Add detrended signal to DataFrame
            data_detrended[f'detrend_{trace}'] = signal_detrended
            exp_fits[f'expfit{trace[-4:]}'] = signal_expfit

        #Plotting the detrended data
        if plot and len(data_detrended.columns) > 0:  # Only plot if there's data

            # Plotting the filtered data with the exponential fit
            fig, axs = plt.subplots(len(traces.columns), figsize=(15, 10), sharex=True)
            color_count = 0
            for trace, exp, ax in zip(traces.columns, exp_fits.columns, axs):
                # Plot original data
                line1 = ax.plot(self.data_seconds, traces[trace], c=self.colors[color_count], label=trace, alpha=0.7)
                color_count += 1
                
                # Create twin axis and plot exp fit
                ax2 = ax.twinx()
                line2 = ax2.plot(self.data_seconds, exp_fits[exp], c='black', label=exp)
                
                # Get combined y limits
                y_min = min(traces[trace].min(), exp_fits[exp].min())
                y_max = max(traces[trace].max(), exp_fits[exp].max())
                
                # Set same limits for both axes
                ax.set_ylim([y_min, y_max])
                ax2.set_ylim([y_min, y_max])
                
                ax.set(ylabel='fluorescence')
                ax2.set(ylabel='exponential fit')
                lns = line1 + line2
                labs = [l.get_label() for l in lns]
                ax.legend(lns, labs, loc=0)
            
            axs[-1].set(xlabel='seconds')
            fig.suptitle(f'exponential fit {self.mousename}')
            if savefig:
                plt.savefig(self.save_path + f'/exp-fit_{self.mousename}.png', dpi=300)
            
            fig, axs = plt.subplots(len(data_detrended.columns), figsize = (15, 10), sharex=True)
            color_count = 0
            for column, ax in zip(data_detrended.columns, axs):
                ax.plot(self.data_seconds, data_detrended[column], c=self.colors[color_count], label=column)
                if method == "subtractive":
                    ax.set(ylabel='data detrended (raw A.U.)')
                if method == "divisive":
                    ax.set(ylabel='data detrended (dF/F)')
                color_count += 1
                ax.legend()
            axs[-1].set(xlabel='seconds')
            fig.suptitle(f'detrended data {self.mousename}, with method: {method}')
            if savefig:
                plt.savefig(self.save_path + f'/Detrended_data_{self.mousename}.png', dpi=300)

        return data_detrended, exp_fits


    def movement_correct(self, plot = False):
        '''
        NOTE not used at the time (2025 Jan, Cohort0 and Cohort1)
        Uses detrended data from 410 and 470 signal to fit a linear regression that is then subtracted from the 470 data 
        only if the correlation is postive.
        Can change to always performing motion correction or changing to 560 if a red sensor is used for motion correction.
        Returns empty list if data could not be motion corrected.
        '''
         # Check if we have detrended data
        if not hasattr(self, 'data_detrended'):
            print('No detrended data found. Run detrend() method first.')
            return []
        
        data = self.data_detrended
        
        # Check if required columns exist
        required_columns = ['detrend_filtered_410', 'detrend_filtered_470']
        if not all(col in data.columns for col in required_columns):
            print('Required columns (detrend_filtered_410, detrend_filtered_470) not found in detrended data.')
            return []
        
        try:
            slope, intercept, r_value, p_value, std_err = linregress(x=data['detrend_filtered_410'], y=data['detrend_filtered_470'])
            print(f'For 410/470 motion correction, slope = {slope:.3f} and R-squared = {r_value**2:.3f}')
            if plot:
                fig, ax = plt.subplots(figsize=(15, 10))
                plt.scatter(data['detrend_filtered_410'][::5], data['detrend_filtered_470'][::5], alpha=0.1, marker='.')
                x = np.array(ax.get_xlim())
                ax.plot(x, intercept + slope * x)
                ax.set_xlabel('410')
                ax.set_ylabel('470')
                ax.set_title(f'410 nm - 470 nm correlation {self.mousename}.')
                xlim = ax.get_xlim()[1]
                ylim = ax.get_ylim()[0]
                plt.rcParams.update({'font.size': 18})
                plt.savefig(self.save_path + f'/motion_correlation_{self.mousename}.png', dpi=300)
                
            if slope > 0:
                control_corr = intercept + slope * data['detrend_filtered_410']
                signal_corrected = data['detrend_filtered_470'] - control_corr
                return signal_corrected
            else:
                print('signal could not be motion corrected')
                print(intercept, slope)
                return []
        except KeyError:
            print('signal could not be motion corrected, the original data is returned')
            print(intercept, slope)
            return []


    def z_score(self, motion = False, plot = False, savefig = False):
        '''
        Z-scoring of signal traces
        Gets relative values of signal
        Calculates median signal value and standard deviation
        Gives the signal strenght in terms of standard deviation units
        Does not take into account signal reduction due to bleaching
        '''
        zscored_data = pd.DataFrame()

        if motion == True:
            signal = self.motion_corr
            zscored_data[f'z_470'] = (signal - np.median(signal)) / np.std(signal)
            print('Motion correction APPLIED to z-score')
        if motion == False:
            signals = self.data_detrended
            for signal in signals:
                    signal_corrected = signals[signal]
                    zscored_data[f'z_{signal[-3:]}'] = (signal_corrected - np.median(signal_corrected)) / np.std(signal_corrected)
            print('Motion correction NOT APPLIED to z-score')

        zscored_data = zscored_data.reset_index(drop = True)
        
        if plot:
            fig, axs = plt.subplots(len(zscored_data.columns),figsize = (15, 10), sharex=True)
            color_count = 0
            if len(zscored_data.columns) > 1:
                for column, ax in zip(zscored_data.columns, axs):
                    ax.plot(self.data_seconds, zscored_data[column], c = self.colors[color_count], label = column)
                    ax.set(xlabel='seconds', ylabel='z-scored dF/F or raw')
                    ax.legend()
                    color_count += 1
            else:
                axs.plot(self.data_seconds, zscored_data, c=self.colors[2])
                axs.set(xlabel='seconds', ylabel='z-scored dF/F or raw')
                axs.legend()
    
            method_label = "raw" if self.info['detrend_method'] == "subtractive" else "dF/F"
            fig.suptitle(f'Z-scored {method_label} data {self.mousename}')
            
            if savefig:
                plt.savefig(self.save_path+f'/zscored_figure_{self.mousename}.png', dpi = 300)

        return zscored_data


    def get_deltaF_F(self, motion = False, plot = False, savefig = False):
        '''
        Input:
        - the detrended signal as delta F
        - The decay curve estimate as baseline fluorescence
        Calculates 100 * deltaF / F (ie. % change)
        :returns
        - dF/F (signals relative to baseline)
        '''
        if self.info['detrend_method'] == 'subtractive':
            print('The method used for detrending was subtractive, deltaF/F is calculated now.')    
            dF_F = pd.DataFrame()
            if motion == False:
                main_data = self.data_detrended
                for signal, fit in zip(main_data, self.exp_fits):
                    F = self.exp_fits[fit]
                    deltaF = main_data[signal]
                    signal_dF_F = 100 * deltaF / F
                    dF_F[f'{signal[-3:]}_dfF'] = signal_dF_F
            if motion == True:
                deltaF = self.motion_corr
                F = self.exp_fits['expfit_470']
                signal_dF_F = 100 * deltaF / F
                dF_F['470_dfF'] = signal_dF_F
                main_data = self.data_detrended
                for signal, fit in zip(main_data, self.exp_fits):
                    if '470' not in signal:
                        F = self.exp_fits[fit]
                        deltaF = main_data[signal]
                        signal_dF_F = 100 * deltaF / F
                        dF_F[f'{signal[-3:]}_dfF'] = signal_dF_F

            if plot:
                fig, axs = plt.subplots(len(dF_F.columns),figsize = (15, 10), sharex=True)
                color_count = 0
                if len(dF_F.columns) >1:
                    for column, ax in zip(dF_F.columns, axs):
                        ax.plot(self.data_seconds, dF_F[column], c = self.colors[color_count], label = column)
                        ax.set(xlabel='seconds', ylabel='dF/F')
                        ax.legend()
                        color_count += 1
                else:
                    axs.plot(self.data_seconds, dF_F, c=self.colors[2])
                    axs.set(xlabel='seconds', ylabel='dF/F')
                fig.suptitle(f'Delta F / F to exponential fit {self.mousename}')
                
                if savefig:
                    plt.savefig(self.save_path+f'/deltaf-F_figure_{self.mousename}.png', dpi = 300)
                
            return dF_F
        if self.info['detrend_method'] == 'divisive': 
            print('The method used for detrending was divisive, deltaF/F has been already calculated')
            dF_F = pd.DataFrame()
            
            if motion == False:
                # Copy all detrended signals to dF_F
                for signal in self.data_detrended.columns:
                    dF_F[f'{signal[-3:]}_dfF'] = self.data_detrended[signal]
            else:
                # Add motion corrected 470 signal
                dF_F['470_dfF'] = self.motion_corr
                # Add other signals if they exist
                for signal in self.data_detrended.columns:
                    if '470' not in signal:
                        dF_F[f'{signal[-3:]}_dfF'] = self.data_detrended[signal]
            
            if plot:
                fig, axs = plt.subplots(len(dF_F.columns),figsize = (15, 10), sharex=True)
                color_count = 0
                if len(dF_F.columns) >1:
                    for column, ax in zip(dF_F.columns, axs):
                        ax.plot(self.data_seconds, dF_F[column], c = self.colors[color_count], label = column)
                        ax.set(xlabel='seconds', ylabel='dF/F')
                        ax.legend()
                        color_count += 1
                else:
                    axs.plot(self.data_seconds, dF_F, c=self.colors[2])
                    axs.set(xlabel='seconds', ylabel='dF/F')
                fig.suptitle(f'Delta F/F {self.mousename}')
                
                if savefig:
                    plt.savefig(self.save_path+f'/deltaf-F_figure_{self.mousename}.png', dpi = 300)
                
            return dF_F


    def write_info_csv(self):
        '''
        Writes the available info into a csv file that can be read into a dictionary
        # To read it back:
        with open(f'{filename}_info.csv') as csv_file:
            reader = csv.reader(csv_file)
            info = dict(reader)
        '''
        path = self.save_path
        info = self.info
        with open(f'{path}/info.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in info.items():
                writer.writerow([key, value])
        print('Info.csv saved')

    # def add_crucial_info(self, additional = None):
    #     '''
    #     :param additional: optional, can take dict where keys are set as column names and
    #      items will repeat through rows. This must be necessary info for later slicing if data is pooled !!NOT IMPLEMENTED
    #     :return: a df with mouse ID, timedelta, recording date, and potential additional info
    #     *IMPORTANT: This must be according to a path where the folder name two levels up from the
    #      fluorencence.csv file is such that the 4 first letters gives mouse ID
    #     '''
    #     # concat dataframes but make sure to add columns specifying the mouse ID, the data, and so on
    #     # will need to backtrack mouse ID and recording data etc. though file names
    #     info_columns = pd.DataFrame()
    #     date_format = '%Y_%m_%d-%H_%M_%S'
 
    #     date_obj = datetime.strptime(self.path.split('/')[-2][:], date_format) #This line requires that the saving structure of the photometry software is kept

    #     actualtime = [date_obj + timedelta(0, time) for time in self.data_seconds['TimeStamp']]
    #     info_columns['Time'] = actualtime
    #     mouseID = self.path.split('/')[-3]#[:4] # if mouse ID annotation is changed to include more or less letters, the number 4 must be changed
    #     print(f'Please ensure that {mouseID} is the correct mouse ID \n '
    #           f'If not, changes must be made to either add_crucial_info fucntion or file naming')
    #     info_columns['mouseID'] = [mouseID for i in range(len(info_columns))]
    #     print(f'Mouse {mouseID}')
    #     Area = input('Add the location of fluorescent protein: ')
    #     info_columns['Area'] = [Area for i in range(len(info_columns))]
    #     Sex = input('Add the sex of the mouse: ')
    #     info_columns['Sex'] = [Sex for i in range(len(info_columns))]
    #     self.info['mouse_info'] = {'sensors': self.sensors, 'target_area': Area,'sex': Sex}
        
    #     print(f'info added for {self.mousename }\n')
    #     return info_columns


    def write_preprocessed_csv(self, Onix_align = True):
        """
        Writes the processed traces into a CSV file containing:
        - Corrected and z-scored traces
        - Optionally motion-corrected signal
        - If self.events exists, it also writes the events DataFrame to a separate file
    
        :param motion_correct: Default False, set to True if a motion-corrected signal should be added. ?FIXME? 
        """
        # Prepare the base filename
        filename = self.path.split('/')[-2][:]
    
        # Combine the base data #add or remove signals to save.
        final_df = pd.concat([self.data_seconds.reset_index(drop=True),
                              #self.filtered.reset_index(drop=True),FIXME delete as not needed, mousename and target area already in info.csv 
                              self.deltaF_F.reset_index(drop=True),
                              self.zscored.reset_index(drop=True),], axis=1)
                              #self.crucial_info.reset_index(drop=True)], axis=1) FIXME delete as not needed, mousename and target area already in info.csv 
        
        final_df = final_df.loc[:, ~final_df.columns.str.contains('^Unnamed')] #removed unwanted extra column
    
        # Save the main fluorescence file
        final_df.to_csv(self.save_path + '/Processed_fluorescence.csv', index=False)
        print('Processed_fluorescence.csv saved')
        
        # Handle events if self.events exists
        if hasattr(self, 'events') and isinstance(self.events, pd.DataFrame) and (Onix_align ==False):
            # Save the events DataFrame separately
            self.events.to_csv(self.save_path + '/Events.csv', index=False)
            print('Events detected and saved.')
            
        if Onix_align ==True:
            print('Original Events.csv saved to Events.csv to be used for ONIX alingment')
            event_path = self.path + 'Events.csv'  # events with precise timestamps
            events = pd.read_csv(event_path)
            events.to_csv(self.save_path + '/Events.csv', index = False)

        mpl.pyplot.close()
        
        
    def plot_all_signals(self):
        """Generate comprehensive figure of signal processing steps on A4 page"""
        # A4 dimensions and setup
        A4_WIDTH = 8.27
        A4_HEIGHT = 11.69
        SMALL_SIZE = 6
        MEDIUM_SIZE = 8
        
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
        plt.rc('axes', labelsize=SMALL_SIZE)     # fontsize of the x and y labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        
        n_signals = len(self.signals.columns)
        n_rows = n_signals * 3
        
        # Create figure
        fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT))
        gs = fig.add_gridspec(n_rows, 1, hspace=0.5)
        
        for idx, signal in enumerate(self.signals.columns):
            # 1. Exponential fit
            ax1 = fig.add_subplot(gs[idx*3])
            ax1.plot(self.data_seconds['TimeStamp'], self.filtered[f'filtered_{signal}'],
                    color=self.colors[idx], alpha=1, label=f'Filtered {signal}', linewidth=0.5)
            # Fix exponential fit column reference
            ax1.plot(self.data_seconds['TimeStamp'], self.exp_fits[f'expfit_{signal[-3:]}'],
                color='black', alpha=1, label='Exponential fit', linewidth=1)
            ax1.set_title(f'Exponential Fit - {signal}')
            ax1.legend(loc='upper right')
            
            # 2. Z-scored signal
            ax2 = fig.add_subplot(gs[idx*3 + 1])
            ax2.plot(self.data_seconds['TimeStamp'], self.zscored[f'z_{signal[-3:]}'],
                    color=self.colors[idx], linewidth=0.2)
            ax2.set_title(f'Z-scored Signal - {signal}')
            
            # 3. dF/F signal
            ax3 = fig.add_subplot(gs[idx*3 + 2])
            ax3.plot(self.data_seconds['TimeStamp'], self.deltaF_F[f'{signal[-3:]}_dfF'],
                    color=self.colors[idx],linewidth=0.2)
            ax3.set_title(f'ΔF/F Signal - {signal}')
        
        # Final formatting
        plt.xlabel('Time (s)')
        filter_method = self.info['filtering_method'][signal]
        detrend_method = self.info['detrend_method']
        fig.suptitle(f'Signal Processing Steps - {self.mousename}\nFiltering: {filter_method}, Detrending: {detrend_method}', 
                    y=0.95, fontsize=MEDIUM_SIZE)
        plt.tight_layout()
        
        # Save figures in multiple formats
        base_path = f'{self.save_path}/all_signals_{self.mousename}'
        plt.savefig(f'{base_path}.png', bbox_inches='tight', dpi=300)
        plt.savefig(f'{base_path}.eps', bbox_inches='tight', format='eps')
        plt.close()