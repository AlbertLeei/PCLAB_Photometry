import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.signal as ss
import tdt
import os
from collections import OrderedDict


class TDTData:
    def __init__(self, tdt_data, folder_path):
        self.streams = {}
        self.behaviors = tdt_data.epocs  # renamed to behaviors

        # Extract the subject name from the folder or file name
        self.subject_name = os.path.basename(folder_path).split('-')[0]

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

        self.psth_df = pd.DataFrame()

    '''********************************** PRINTING INFO **********************************'''
    def print_behaviors(self):
        """
        Prints all behavior names in self.behaviors.
        """
        if not self.behaviors:
            print("No behaviors found.")
        else:
            print("Behavior names in self.behaviors:")
            for behavior_name in self.behaviors.keys():
                print(behavior_name)

    '''********************************** FILTERING **********************************'''
    def smooth_signal(self, filter_window=101, filter_type='moving_average'):
        '''
        Smooths the signal using a specified filter type.

        Parameters:
        filter_window (int): The window size for the filter.
        filter_type (str): The type of filter to use. Options are 'moving_average' or 'lowpass'.
        '''
        for stream_name in ['DA', 'ISOS']:
            if stream_name in self.streams:
                data = self.streams[stream_name]

                if filter_type == 'moving_average':
                    # Moving average filter
                    b = np.ones(filter_window) / filter_window
                    a = 1
                elif filter_type == 'lowpass':
                    # Lowpass filter (Butterworth)
                    nyquist = 0.5 * self.fs
                    cutoff_freq = 1.0  # Set cutoff frequency in Hz (adjust as needed)
                    normal_cutoff = cutoff_freq / nyquist
                    b, a = ss.butter(N=filter_window, Wn=normal_cutoff, btype='low', analog=False)
                else:
                    raise ValueError("Invalid filter_type. Choose 'moving_average' or 'lowpass'.")

                smoothed_data = ss.filtfilt(b, a, data)
                self.streams[stream_name] = smoothed_data

        # Clear dFF and zscore since the raw data has changed
        self.dFF = None
        self.zscore = None

                
    def downsample_data(self, N=10):
        downsampled_timestamps = self.timestamps[::N]
        for stream_name in ['DA', 'ISOS']:
            if stream_name in self.streams:
                data = self.streams[stream_name]
                downsampled_data = [np.mean(data[i:i + N]) for i in range(0, len(data), N)]
                self.streams[stream_name] = downsampled_data
        self.timestamps = downsampled_timestamps

        # Clear dFF and zscore since the raw data has changed
        self.dFF = None
        self.zscore = None

    def remove_time(self, start_time, end_time):
        """
        Removes a segment of time from the data streams and timestamps and then verifies the signal length.
        
        Parameters:
        start_time (float): The start time of the segment to be removed (in seconds).
        end_time (float): The end time of the segment to be removed (in seconds).
        """
        # Find the indices corresponding to the start and end times
        start_index = np.where(self.timestamps >= start_time)[0][0]
        end_index = np.where(self.timestamps <= end_time)[0][-1]
        
        # Create an array of boolean values, keeping all indices outside the specified range
        keep_indices = np.ones_like(self.timestamps, dtype=bool)
        keep_indices[start_index:end_index+1] = False
        
        # Update the streams by concatenating the parts of the signal that are kept
        for stream_name in ['DA', 'ISOS']:
            if stream_name in self.streams:
                self.streams[stream_name] = self.streams[stream_name][keep_indices]
        
        # Update the timestamps by concatenating the kept timestamps
        self.timestamps = self.timestamps[keep_indices]
        
        # Clear dFF and zscore since the raw data has changed
        self.dFF = None
        self.zscore = None

        # Verify the signal lengths
        self.verify_signal()
    
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

        # Clear dFF and zscore since the raw data has changed
        self.dFF = None
        self.zscore = None

    def verify_signal(self):
        """
        Verifies that all streams (DA and ISOS) have the same length by trimming them to the shortest length.
        This function also adjusts the timestamps accordingly. No smoothing is applied.
        """
        da_length = len(self.streams[self.DA])
        isos_length = len(self.streams[self.ISOS])
        min_length = min(da_length, isos_length)
        
        if da_length != min_length or isos_length != min_length:
            # Trim the streams to the shortest length
            self.streams[self.DA] = self.streams[self.DA][:min_length]
            self.streams[self.ISOS] = self.streams[self.ISOS][:min_length]
            
            # Trim the timestamps to match the new signal length
            self.timestamps = self.timestamps[:min_length]
            
            print(f"Signals trimmed to {min_length} samples to match the shortest signal.")


    '''********************************** DFF AND ZSCORE **********************************'''
    def execute_controlFit_dff(self, control, signal, filter_window=101):
        """
        Fits the control channel to the signal channel and calculates delta F/F (dFF).

        Parameters:
        control (numpy.array): The control signal (e.g., isosbestic control signal).
        signal (numpy.array): The signal of interest (e.g., dopamine signal).
        filter_window (int): The window size for the moving average filter.

        Returns:
        norm_data (numpy.array): The normalized delta F/F signal.
        control_fit (numpy.array): The fitted control signal.
        """
        if filter_window > 1:
            # Smoothing both signals
            control_smooth = ss.filtfilt(np.ones(filter_window) / filter_window, 1, control)
            signal_smooth = ss.filtfilt(np.ones(filter_window) / filter_window, 1, signal)
        else:
            control_smooth = control
            signal_smooth = signal

        # Fitting the control signal to the signal of interest
        p = np.polyfit(control_smooth, signal_smooth, 1)
        control_fit = p[0] * control_smooth + p[1]

        # Calculating delta F/F (dFF)
        norm_data = 100 * (signal_smooth - control_fit) / control_fit

        return norm_data, control_fit

    def compute_dff(self, filter_window=101):
        """
        Computes the delta F/F (dFF) signal by fitting the isosbestic control signal to the signal of interest.
        
        Parameters:
        filter_window (int): The window size for the moving average filter.
        """
        if 'DA' in self.streams and 'ISOS' in self.streams:
            signal = np.array(self.streams['DA'])
            control = np.array(self.streams['ISOS'])
            
            # Call the execute_controlFit_dff method
            self.dFF, self.control_fit = self.execute_controlFit_dff(control, signal, filter_window)
            
            # Calculate the standard deviation of dFF
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
        event_name = behavior_name
        
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

    def combine_consecutive_behaviors(self, behavior_name, bout_time_threshold=2, min_occurrences=1):
        behavior_event = behavior_name
        behavior_onsets = self.behaviors[behavior_event].onset
        behavior_offsets = self.behaviors[behavior_event].offset

        combined_onsets = []
        combined_offsets = []

        if len(behavior_onsets) == 0:
            return

        start_idx = 0

        while start_idx < len(behavior_onsets):
            # Identify indices where the difference between consecutive onsets exceeds the threshold
            bout_indices = np.where(np.diff(behavior_onsets[start_idx:]) >= bout_time_threshold)[0]

            if len(bout_indices) == 0:
                # If no more indices found, combine the rest of the events
                combined_onsets.append(behavior_onsets[start_idx])
                combined_offsets.append(behavior_offsets[-1])
                break

            for idx in bout_indices:
                if start_idx + idx < len(behavior_onsets):
                    combined_onsets.append(behavior_onsets[start_idx])
                    combined_offsets.append(behavior_offsets[start_idx + idx])
                    start_idx += idx + 1

            # If we didn't process all onsets, add the last segment
            if start_idx < len(behavior_onsets):
                combined_onsets.append(behavior_onsets[start_idx])
                combined_offsets.append(behavior_offsets[start_idx])

            start_idx += 1

        # Filter out bouts with fewer than the minimum occurrences
        valid_indices = []
        for i in range(len(combined_onsets)):
            num_occurrences = len([1 for onset in behavior_onsets if combined_onsets[i] <= onset <= combined_offsets[i]])
            if num_occurrences >= min_occurrences:
                valid_indices.append(i)

        # Update the behavior with the combined onsets and offsets
        self.behaviors[behavior_event].onset = [combined_onsets[i] for i in valid_indices]
        self.behaviors[behavior_event].offset = [combined_offsets[i] for i in valid_indices]


    '''********************************** PSTH **********************************'''
    def compute_psth(self, behavior_name, pre_time=5, post_time=10, signal_type='dFF'):
        """
        Compute the Peri-Stimulus Time Histogram (PSTH) for a given behavior.

        Parameters:
        behavior_name (str): The name of the behavior event to use for PSTH computation.
        pre_time (float): Time in seconds before the behavior event onset to include in the PSTH.
        post_time (float): Time in seconds after the behavior event onset to include in the PSTH.
        signal_type (str): Type of signal to use for PSTH computation. Options are 'dFF' or 'zscore'.

        Returns:
        psth_df (pd.DataFrame): DataFrame containing the PSTH with columns for each time point.
                                Includes both mean and standard deviation.
        """
        if behavior_name not in self.behaviors.keys():
            raise ValueError(f"Behavior '{behavior_name}' not found in behaviors.")

        behavior_onsets = self.behaviors[behavior_name].onset
        sampling_rate = self.fs

        # Select the appropriate signal type
        if signal_type == 'dFF':
            if self.dFF is None:
                self.compute_dff()
            signal = np.array(self.dFF)
        elif signal_type == 'zscore':
            if self.zscore is None:
                self.compute_zscore()
            signal = np.array(self.zscore)
        else:
            raise ValueError("Invalid signal_type. Choose 'dFF' or 'zscore'.")

        # Initialize PSTH data structure
        n_samples_pre = int(pre_time * sampling_rate)
        n_samples_post = int(post_time * sampling_rate)
        psth_matrix = []

        # Compute PSTH for each behavior onset
        for onset in behavior_onsets:
            onset_idx = np.searchsorted(self.timestamps, onset)
            start_idx = max(onset_idx - n_samples_pre, 0)
            end_idx = min(onset_idx + n_samples_post, len(signal))

            # Extract signal around the event
            psth_segment = signal[start_idx:end_idx]

            # Pad if necessary to ensure equal length
            if len(psth_segment) < n_samples_pre + n_samples_post:
                padding = np.full((n_samples_pre + n_samples_post) - len(psth_segment), np.nan)
                psth_segment = np.concatenate([psth_segment, padding])

            psth_matrix.append(psth_segment)

        # Convert to DataFrame for ease of analysis
        time_axis = np.linspace(-pre_time, post_time, n_samples_pre + n_samples_post)
        psth_df = pd.DataFrame(psth_matrix, columns=time_axis)

        # Calculate the mean and standard deviation for each time point
        psth_mean = psth_df.mean(axis=0)
        psth_std = psth_df.std(axis=0)

        # Return a DataFrame with both mean and std
        result_df = pd.DataFrame({
            'mean': psth_mean,
            'std': psth_std
        })

        self.psth_df = result_df
        return result_df


    def plot_psth(self, behavior_name, signal_type='dFF'):
        """
        Plot the Peri-Stimulus Time Histogram (PSTH).

        Parameters:
        psth_df (pd.DataFrame): DataFrame containing the PSTH data.
        behavior_name (str): Name of the behavior event for labeling the plot.
        signal_type (str): Type of signal used for PSTH computation. Options are 'dFF' or 'zscore'.
        """
        if self.psth_df is None or self.psth_df.empty:
            self.compute_psth(behavior_name, pre_time=5, post_time=10, signal_type=signal_type)

        psth_df = self.psth_df

        mean_psth = psth_df.mean(axis=0)
        std_psth = psth_df.std(axis=0)

        plt.figure(figsize=(10, 5))
        plt.plot(psth_df.columns, mean_psth, label=f'{signal_type} Mean')
        plt.fill_between(psth_df.columns, mean_psth - std_psth, mean_psth + std_psth, alpha=0.3)

        plt.xlabel('Time (s)')
        plt.ylabel(f'{signal_type}')
        plt.title(f'PSTH for {behavior_name}')
        plt.axvline(0, color='r', linestyle='--', label=f'{behavior_name} Onset')
        plt.legend()
        plt.show()

    '''********************************** PLOTTING **********************************'''
    def plot_behavior_event(self, behavior_name, plot_type='dFF', ax=None):
        """
        Plot Delta F/F (dFF) with behavior events. Can be used to plot in a given Axes object or individually.

        Parameters:
        - behavior_name: The name of the behavior to plot. Use 'all' to plot all behaviors.
        - plot_type: The type of plot. Options are 'dFF', 'zscore', or 'raw'.
        - ax: An optional matplotlib Axes object. If provided, the plot will be drawn on this Axes.
        """
        y_data = []
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

        if ax is None:
            fig, ax = plt.subplots(figsize=(18, 6))

        ax.plot(self.timestamps, np.array(y_data), linewidth=2, color='green', label=plot_type)

        if behavior_name == 'all':
            for behavior_event in self.behaviors.keys():
                if behavior_event.endswith('_event'):
                    behavior_onsets = self.behaviors[behavior_event].onset
                    behavior_offsets = self.behaviors[behavior_event].offset
                    for on, off in zip(behavior_onsets, behavior_offsets):
                        ax.axvspan(on, off, alpha=0.25, label=behavior_event, color=np.random.rand(3,))
        else:
            behavior_event = behavior_name
            if behavior_event not in self.behaviors.keys():
                raise ValueError(f"Behavior event '{behavior_event}' not found in behaviors.")
            behavior_onsets = self.behaviors[behavior_event].onset
            behavior_offsets = self.behaviors[behavior_event].offset
            for on, off in zip(behavior_onsets, behavior_offsets):
                ax.axvspan(on, off, alpha=0.25, color='dodgerblue')

        ax.set_ylabel(y_label)
        ax.set_xlabel('Seconds')
        ax.set_title(f'{self.subject_name}: {y_title} with {behavior_name} Bouts' if behavior_name != 'all' else f'{self.subject_name}: {y_title} with All Behavior Events')
        ax.legend()

        if ax is None:
            plt.tight_layout()
            plt.show()

    def plot(self, plot_type='zscore'):
        '''
        Plots the selected signal type.

        Parameters:
        plot_type (str): The type of plot to generate. Options are 'raw', 'dFF', and 'zscore'.
        '''
        total_duration = self.timestamps[-1] - self.timestamps[0]  # Total duration of the data
        num_major_ticks = 10  # Number of major ticks (adjust this as needed)

        if plot_type == 'raw':
            if self.DA in self.streams and self.ISOS in self.streams:
                plt.figure(figsize=(18, 6))
                plt.plot(self.timestamps, self.streams[self.DA], linewidth=2, color='blue', label='DA')
                plt.plot(self.timestamps, self.streams[self.ISOS], linewidth=2, color='blueviolet', label='ISOS')
                plt.ylabel('mV')
                plt.title(f'{self.subject_name}: Raw Demodulated Responses')
                plt.legend(loc='upper right')

        elif plot_type == 'dFF':
            if self.dFF is not None:
                plt.figure(figsize=(18, 6))
                plt.plot(self.timestamps, self.dFF, label='dFF', color='green')
                plt.ylabel('ΔF/F')
                plt.title(f'{self.subject_name}: Delta F/F (dFF) Signal')
                plt.legend(loc='upper right')
            else:
                print("dFF data not available. Please compute dFF first.")
                return

        elif plot_type == 'zscore':
            if self.zscore is not None and len(self.zscore) > 0:
                plt.figure(figsize=(18, 6))
                plt.plot(self.timestamps, self.zscore, linewidth=2, color='red', label='z-score')
                plt.ylabel('z-score')
                plt.title(f'{self.subject_name}: Z-score of Delta F/F (dFF) Signal')
                plt.legend(loc='upper right')
            else:
                print("z-score data not available. Please compute z-score first.")
                return
        else:
            raise ValueError("Invalid plot_type. Choose from 'raw', 'dFF', or 'zscore'.")

        plt.xlabel('Seconds')
        ax = plt.gca()

        # Dynamically set the number of major ticks
        ax.xaxis.set_major_locator(ticker.MultipleLocator(total_duration / num_major_ticks))

        # Remove the grid
        plt.grid(False)
        plt.show()
