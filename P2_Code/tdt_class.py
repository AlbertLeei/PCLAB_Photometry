import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import tdt

class TDTData:
    def __init__(self, tdt_data):
        self.streams = {}
        self.behaviors = tdt_data.epocs  # renamed to behaviors

        # Assume all streams have the same sampling frequency and length
        self.fs = tdt_data.streams['_465A'].fs
        self.timestamps = np.arange(len(tdt_data.streams['_465A'].data)) / self.fs

        # Streams
        self.DA = 'DA'  # 465
        self.ISOS = 'ISOS'  # 405
        self.streams['DA'] = tdt_data.streams['_465A'].data
        self.streams['ISOS'] = tdt_data.streams['_405A'].data
        
        self.dFF = None
        self.std_dFF = None
        self.zscore = None

    '''********************************** FILTERING **********************************'''
    def smooth_signal(self, filter_window=101):
        '''
        Smooths the signal using a moving average filter.
        
        Parameters:
        filter_window (int): The window size for the moving average filter.
        '''
        for stream_name in ['DA', 'ISOS']:
            if stream_name in self.streams:
                data = self.streams[stream_name]
                if filter_window > 1:
                    b = np.ones(filter_window) / filter_window
                    a = 1
                    smoothed_data = ss.filtfilt(b, a, data)
                    self.streams[stream_name] = smoothed_data
                elif filter_window == 0:
                    self.streams[stream_name] = data
                else:
                    raise ValueError("filter_window must be greater than 0")
                
    def downsample_data(self, N=10):
        downsampled_timestamps = self.timestamps[::N]
        for stream_name in ['DA', 'ISOS']:
            if stream_name in self.streams:
                data = self.streams[stream_name]
                downsampled_data = [np.mean(data[i:i + N]) for i in range(0, len(data), N)]
                self.streams[stream_name] = downsampled_data
        self.timestamps = downsampled_timestamps

    def remove_time(self, start_time, end_time):
        start_index = np.where(self.timestamps >= start_time)[0][0]
        end_index = np.where(self.timestamps <= end_time)[0][-1]
        keep_indices = np.ones_like(self.timestamps, dtype=bool)
        keep_indices[start_index:end_index+1] = False
        for stream_name in ['DA', 'ISOS']:
            if stream_name in self.streams:
                self.streams[stream_name] = self.streams[stream_name][keep_indices]
        self.timestamps = self.timestamps[keep_indices]

    def remove_initial_LED_artifact(self, t=10):
        '''
        This function removes the initial artifact caused by the onset of LEDs turning on.
        The artifact is assumed to occur within the first 't' seconds of the data.
        '''
        ind = np.where(self.timestamps > t)[0][0]
        for stream_name in ['DA', 'ISOS']:
            if stream_name in self.streams:
                self.streams[stream_name] = self.streams[stream_name][ind:]
        self.timestamps = self.timestamps[ind:]


    '''********************************** DFF AND ZSCORE **********************************'''
    def compute_dff(self):
        if 'DA' in self.streams and 'ISOS' in self.streams:
            x = np.array(self.streams['ISOS'])
            y = np.array(self.streams['DA'])
            bls = np.polyfit(x, y, 1)
            Y_fit_all = np.multiply(bls[0], x) + bls[1]
            Y_dF_all = y - Y_fit_all
            self.dFF = np.multiply(100, np.divide(Y_dF_all, Y_fit_all))
            self.std_dFF = np.std(self.dFF)
        else:
            self.dFF = None
            self.std_dFF = None

    def compute_zscore(self, method='standard', baseline_start=None, baseline_end=None):
        """
        Computes the z-score of the delta F/F (dFF) signal and saves it as a class variable.

        Parameters:
        method (str): The method used to compute the z-score. Options are:
            'standard' - Computes the z-score using the standard method (z = (x - mean) / std).
            'baseline' - Computes the z-score using a baseline period. Requires baseline_start and baseline_end.
            'modified' - Computes the z-score using a modified z-score method (z = 0.6745 * (x - median) / MAD).
        baseline_start (float): The start time of the baseline period for baseline z-score computation.
        baseline_end (float): The end time of the baseline period for baseline z-score computation.
        """
        if self.dFF is None:
            self.compute_dff()
        
        dff = np.array(self.dFF)
        
        if method == 'standard':
            self.zscore = (dff - np.nanmean(dff)) / np.nanstd(dff)
        
        elif method == 'baseline':
            if baseline_start is None or baseline_end is None:
                raise ValueError("Baseline start and end times must be provided for baseline z-score computation.")
            
            baseline_indices = np.where((self.timestamps >= baseline_start) & (self.timestamps <= baseline_end))[0]
            if len(baseline_indices) == 0:
                raise ValueError("No baseline data found within the specified baseline period.")
            
            baseline_mean = np.nanmean(dff[baseline_indices])
            baseline_std = np.nanstd(dff[baseline_indices])
            self.zscore = (dff - baseline_mean) / baseline_std
        
        elif method == 'modified':
            median = np.nanmedian(dff)
            mad = np.nanmedian(np.abs(dff - median))
            self.zscore = 0.6745 * (dff - median) / mad
        
        else:
            raise ValueError("Invalid zscore_method. Choose from 'standard', 'baseline', or 'modified'.")

    '''********************************** BEHAVIORS **********************************'''
    def extract_single_behavior(self, behavior_name, bout_aggregated_df):
        '''
        This function extracts single behavior events from the DataFrame and adds them to the TDT recording.

        Parameters:
        behavior_name: The name of the behavior to extract.
        bout_aggregated_df: The DataFrame containing the behavior data.
        '''
        # Filter the DataFrame to get rows corresponding to the specific behavior
        behavior_df = bout_aggregated_df[bout_aggregated_df['Behavior'] == behavior_name]
        
        # Extract onset and offset times as lists
        onset_times = behavior_df['Start (s)'].values.tolist()
        offset_times = behavior_df['Stop (s)'].values.tolist()
        
        # Define the event name by appending '_event' to the behavior name
        event_name = behavior_name + '_event'
        
        # Create a data array filled with 1s, with the same length as onset_times
        data_arr = [1] * len(onset_times)

        # Create a dictionary to hold the event data
        EVENT_DICT = {
            "name": event_name,
            "onset": onset_times,
            "offset": offset_times,
            "type_str": self.behaviors.Cam1.type_str,  # Copy type_str from an existing epoc
            "data": data_arr
        }

        # Assign the new event dictionary to the behaviors structure
        self.behaviors[event_name] = tdt.StructType(EVENT_DICT)

    def extract_manual_annotation_behaviors(self, bout_aggregated_csv_path):
        '''
        This function processes all behaviors of type 'STATE' in the CSV file and extracts them
        into the TDT recording.

        Parameters:
        bout_aggregated_csv_path: The file path to the CSV containing the behavior data.
        '''
        # Load the CSV file into a DataFrame
        bout_aggregated_df = pd.read_csv(bout_aggregated_csv_path)

        # Filter the DataFrame to include only rows where the behavior type is 'STATE'
        state_behaviors_df = bout_aggregated_df[bout_aggregated_df['Behavior type'] == 'STATE']
        
        # Get a list of unique behaviors in the filtered DataFrame
        unique_behaviors = state_behaviors_df['Behavior'].unique()

        # Iterate over each unique behavior
        for behavior in unique_behaviors:
            # Filter the DataFrame for the current behavior
            behavior_df = state_behaviors_df[state_behaviors_df['Behavior'] == behavior]
            
            # Call the helper function to extract and add the behavior events
            self.extract_single_behavior(behavior, behavior_df)

    def combine_consecutive_behaviors(self, behavior_name, bout_time_threshold=10, min_occurrences=1):
        '''
        Combine consecutive behavior events based on a specified time threshold 
        and minimum number of occurrences.

        Parameters:
        behavior_name (str): The name of the behavior to process.
        bout_time_threshold (int): The minimum time between events to consider them separate bouts.
        min_occurrences (int): The minimum number of occurrences to consider a valid bout.
        '''
        behavior_event = behavior_name + '_event'

        behavior_onsets = np.array(self.behaviors[behavior_event].onset)
        behavior_offsets = np.array(self.behaviors[behavior_event].offset)

        # Calculate the differences between consecutive onsets
        on_diff = np.diff(behavior_onsets)
        bout_indices = np.where(on_diff >= bout_time_threshold)[0]

        # Initialize lists for combined onsets and offsets
        combined_onsets = []
        combined_offsets = []

        # Combine consecutive behaviors based on the threshold
        start_idx = 0
        for ind in bout_indices:
            combined_onsets.append(behavior_onsets[start_idx])
            combined_offsets.append(behavior_offsets[ind])
            start_idx = ind + 1

        # Handle the last bout
        combined_onsets.append(behavior_onsets[bout_indices[-1] + 1])
        combined_offsets.append(behavior_offsets[-1])

        # Filter out bouts with fewer than the minimum occurrences
        valid_onsets = []
        valid_offsets = []

        for on, off in zip(combined_onsets, combined_offsets):
            count = len(np.where((behavior_onsets >= on) & (behavior_onsets <= off))[0])
            if count >= min_occurrences:
                valid_onsets.append(on)
                valid_offsets.append(off)

        # Update the behavior dictionary with valid bouts
        self.behaviors[behavior_event].onset = valid_onsets 
        self.behaviors[behavior_event].offset = valid_offsets

    '''********************************** PLOTTING **********************************'''
    def plot_behavior_event(self, behavior_name, plot_type='dFF'):
        '''
        Plot Delta F/F (dFF) with behavior events.

        Parameters:
        behavior_name (str): The name of the behavior.
        plot_type (str): The type of plot. Options are 'dFF', 'zscore', 'raw'.
        '''
        if plot_type == 'dFF':
            if self.dFF is None:
                self.compute_dff()
            y_data = self.dFF
            y_label = r'$\Delta$F/F'
            y_title = 'dFF Signal'
        elif plot_type == 'zscore':
            if self.zscore is None:
                self.compute_zscore()
            y_data = self.zscore
            y_label = 'z-score'
            y_title = 'z-score Signal'
        elif plot_type == 'raw':
            y_data = self.streams[self.DA]
            y_label = 'Raw Signal (mV)'
            y_title = 'Raw Signal'
        else:
            raise ValueError("Invalid plot_type. Choose from 'dFF', 'zscore', or 'raw'.")

        behavior_event = behavior_name + '_event'

        # Ensure the behavior events are combined before plotting
        self.combine_consecutive_behaviors(behavior_name)

        behavior_onsets = self.behaviors[behavior_event].onset
        behavior_offsets = self.behaviors[behavior_event].offset

        fig = plt.figure(figsize=(18, 6))
        ax = fig.add_subplot(111)
        for on, off in zip(behavior_onsets, behavior_offsets):
            ax.axvspan(on, off, alpha=0.25, color='dodgerblue')

        ax.plot(self.timestamps, y_data, linewidth=2, color='green', label=plot_type)
        ax.set_ylabel(y_label)
        ax.set_xlabel('Seconds')
        ax.set_title(f'{y_title} with {behavior_name} Bouts')
        ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_raw_trace(self):
        if self.DA in self.streams and self.ISOS in self.streams:
            fig1 = plt.figure(figsize=(18, 6))
            ax1 = fig1.add_subplot(111)
            p1, = ax1.plot(self.timestamps, self.streams[self.DA], linewidth=2, color='blue', label='DA')
            p2, = ax1.plot(self.timestamps, self.streams[self.ISOS], linewidth=2, color='blueviolet', label='ISOS')
            ax1.set_ylabel('mV')
            ax1.set_xlabel('Seconds', fontsize=14)
            ax1.set_title('Raw Demodulated Responses', fontsize=14)
            ax1.legend(handles=[p1, p2], loc='upper right')
            plt.show()

    def plot_dff(self):
        '''
        Plots the Delta F/F (dFF) signal.
        '''
        if self.dFF is not None:
            plt.figure(figsize=(18, 6))
            plt.plot(self.timestamps, self.dFF, label='dFF', color='green')
            plt.xlabel('Seconds')
            plt.ylabel('ΔF/F')
            plt.title('Delta F/F (dFF) Signal')
            plt.legend()
            plt.show()
        else:
            print("dFF data not available. Please compute dFF first.")

    def plot_zscore(self):
        """
        Plots the z-score of the delta F/F (dFF) signal.
        """
        if self.zscore is None or len(self.zscore) == 0:
            raise ValueError("z-score has not been computed or is empty. Run compute_zscore() first.")
        
        plt.figure(figsize=(18, 6))
        plt.plot(self.timestamps, self.zscore, linewidth=2, color='red', label='z-score')
        plt.ylabel('z-score')
        plt.xlabel('Seconds', fontsize=14)
        plt.title('Z-score of Delta F/F (dFF) Signal', fontsize=14)
        plt.legend(loc='upper right')
        plt.show()
