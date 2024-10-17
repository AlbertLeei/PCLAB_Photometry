import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.signal as ss
import tdt
import os
from collections import OrderedDict
import sys
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d


root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))  # Go up one directory to P2_Code
# Add the root directory to sys.path
sys.path.append(root_dir)

class TDTData:

    def __init__(self, tdt_data, folder_path):
        self.streams = {}
        self.behaviors = {key: value for key, value in tdt_data.epocs.items() if key not in ['Cam1', 'Tick']}

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
        
        # Preprocessing
        self.smoothed_DA = np.empty(1)
        self.smoothed_ISOS = np.empty(1)
        self.isosbestic_fc = np.empty(1)
        self.DA_fc = np.empty(1)
        self.cropped_DA = np.empty(1)
        self.cropped_ISOS = np.empty(1)
        self.isosbestic_corrected = np.empty(1)
        self.DA_corrected = np.empty(1)
        self.isosbestic_standardized = np.empty(1)
        self.calcium_standardized = np.empty(1)
        
        
        self.dFF = None
        self.std_dFF = None
        self.zscore = None

        self.psth_df = pd.DataFrame()

        # Extra Experiments
        self.s1_events = None
        self.s2_events = None
        self.bout_dict = {}
        self.first_behavior_dict = {}



    from hab_dishab.hab_dishab_extension import hab_dishab_plot_behavior_event, hab_dishab_extract_intruder_bouts, hab_dishab_find_behavior_events_in_bout 
    from home_cage.home_cage_extension import hc_extract_intruder_bouts, hc_plot_behavior_event, hc_find_behavior_events_in_bout
    from social_pref.social_pref_extension import sp_extract_intruder_events, sp_plot_behavior_event, sp_remove_time_around_subject_introduced
    from social_pref.social_pref_extension import sp_extract_intruder_events_combined, sp_plot_behavior_event_combined, sp_remove_time_around_subject_introduced
    from defeat.defeat_extension import d_proc_extract_bout, d_proc_find_behavior_events_in_bout, d_proc_plot_behavior_event
    from reward_training.reward_training_extension import rt_plot_behavior_event, find_overlapping_port_entries, align_port_entries_to_sound_cues
    from aggression.aggression_extension import ag_extract_aggression_events

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

    '''********************************** PREPROCESSING **********************************'''


    def remove_initial_LED_artifact(self, t=30):
        ind = np.where(self.timestamps > t)[0][0]
        
        # Debugging: Print before removing the artifact
        # print(f"Before removing LED artifact: Timestamps range from {self.timestamps[0]} to {self.timestamps[-1]}")

        for stream_name in ['DA', 'ISOS']:
            if stream_name in self.streams:
                self.streams[stream_name] = self.streams[stream_name][ind:]
        self.timestamps = self.timestamps[ind:]

        # Debugging: Print after removing the artifact
        # print(f"After removing LED artifact: Timestamps range from {self.timestamps[0]} to {self.timestamps[-1]}")

        # Clear dFF and zscore since the raw data has changed
        self.dFF = None
        self.zscore = None

    def remove_final_data_segment(self, t=30):
        '''
        This function removes the final segment of the data, assumed to be the last 't' seconds.
        It truncates the streams and timestamps accordingly.
        '''
        end_time = self.timestamps[-1] - t
        ind = np.where(self.timestamps <= end_time)[0][-1]
        
        for stream_name in ['DA', 'ISOS']:
            if stream_name in self.streams:
                self.streams[stream_name] = self.streams[stream_name][:ind+1]
        
        self.timestamps = self.timestamps[:ind+1]

        # Clear dFF and zscore since the raw data has changed
        self.dFF = None
        self.zscore = None

    def verify_signal(self):
        da_length = len(self.streams[self.DA])
        isos_length = len(self.streams[self.ISOS])
        min_length = min(da_length, isos_length)
        
        # Debugging: Print signal lengths before trimming
        # print(f"Before verifying signal: DA length = {da_length}, ISOS length = {isos_length}")

        if da_length != min_length or isos_length != min_length:
            # Trim the streams to the shortest length
            self.streams[self.DA] = self.streams[self.DA][:min_length]
            self.streams[self.ISOS] = self.streams[self.ISOS][:min_length]
            
            # Trim the timestamps to match the new signal length
            self.timestamps = self.timestamps[:min_length]

        # Debugging: Print signal lengths after trimming
        # print(f"After verifying signal: DA length = {len(self.streams[self.DA])}, ISOS length = {len(self.streams[self.ISOS])}, Timestamps length = {len(self.timestamps)}")

    
    def smooth_and_apply(self, window_len=1):
        """Smooth both DA and ISOS signals using a window with requested size, and store them.

        This method smooths the data streams (DA and ISOS) using the convolution of a scaled window 
        with the signal. The signals are extended by reflecting the ends to minimize boundary effects.

        Args:
            window_len (int): The dimension of the smoothing window; should be an odd integer.

        Sets:
            self.smoothed_DA: The smoothed and trimmed DA signal.
            self.smoothed_ISOS: The smoothed and trimmed ISOS signal.
        """

        def smooth_signal(source, window_len):
            """Helper function to smooth a signal."""
            if source.ndim != 1:
                raise ValueError("smooth only accepts 1 dimension arrays.")
            if source.size < window_len:
                raise ValueError("Input vector needs to be bigger than window size.")
            if window_len < 3:
                return source

            # Extend the signal by reflecting at the edges
            s = np.r_[source[window_len-1:0:-1], source, source[-2:-window_len-1:-1]]

            # Create a window for smoothing (using a flat window here)
            w = np.ones(window_len, 'd')

            # Convolve and return the smoothed signal
            return np.convolve(w / w.sum(), s, mode='valid')

        # Debugging: Print before smoothing
        # print(f"Before smoothing: DA length = {len(self.streams['DA'])}, ISOS length = {len(self.streams['ISOS'])}")

        # Apply smoothing to DA and ISOS streams, then trim the excess padding
        if 'DA' in self.streams:
            smoothed_DA = smooth_signal(self.streams['DA'], window_len)
            # Trim the excess by slicing the array to match the original length
            self.smoothed_DA = smoothed_DA[window_len//2:-window_len//2+1]

            # Debugging: Print after smoothing DA
            # print(f"After smoothing DA: Smoothed length = {len(smoothed_DA)}, Trimmed length = {len(self.smoothed_DA)}")

        if 'ISOS' in self.streams:
            smoothed_ISOS = smooth_signal(self.streams['ISOS'], window_len)
            # Trim the excess by slicing the array to match the original length
            self.smoothed_ISOS = smoothed_ISOS[window_len//2:-window_len//2+1]

            # Debugging: Print after smoothing ISOS
            # print(f"After smoothing ISOS: Smoothed length = {len(smoothed_ISOS)}, Trimmed length = {len(self.smoothed_ISOS)}")

        # Debugging: Print final lengths of smoothed signals
        # print(f"Final smoothed lengths: DA = {len(self.smoothed_DA)}, ISOS = {len(self.smoothed_ISOS)}")

 
    def perform_standardization(self):
            """Standardizes the corrected signals (isosbestic and calcium).

            Args: 
                isosbestic (arr): The baseline-corrected isosbestic signal.
                calcium (arr): The baseline-corrected calcium signal.

            Returns:   
                isosbestic_standardized (arr): The standardized isosbestic signal.
                calcium_standardized (arr): The standardized calcium signal.
            """
            isosbestic = self.isosbestic_corrected 
            da = self.DA_corrected

            # print("\nStarting standardization")

            # Standardization: (value - median) / std deviation
            self.isosbestic_standardized = (isosbestic - np.mean(isosbestic)) / np.std(isosbestic)
            self.calcium_standardized = (da - np.mean(da)) / np.std(da)

    def perform_standardization_raw(self):
            """Standardizes the corrected signals (isosbestic and calcium).

            Args: 
                isosbestic (arr): The baseline-corrected isosbestic signal.
                calcium (arr): The baseline-corrected calcium signal.

            Returns:   
                isosbestic_standardized (arr): The standardized isosbestic signal.
                calcium_standardized (arr): The standardized calcium signal.
            """
            isosbestic = self.streams['ISOS']
            da = self.streams['DA']

            # print("\nStarting standardization")

            # Standardization: (value - median) / std deviation
            self.isosbestic_standardized = (isosbestic - np.mean(isosbestic)) / np.std(isosbestic)
            self.calcium_standardized = (da - np.mean(da)) / np.std(da)



    def align_channels(self):
        """
        Function that performs linear regression between isosbestic_corrected and DA_corrected signals, and aligns
        the fitted isosbestic with the DA signal. The results are stored in the class as a dictionary.
        
        This function grabs the timestamps, DA_corrected, and isosbestic_corrected directly from the class attributes.
        """
        # print("\nStarting linear regression and signal alignment for Isosbestic and DA signals!")

        # Ensure necessary data is available
        if len(self.DA_corrected) == 0 or len(self.isosbestic_corrected) == 0:
            raise ValueError("Corrected DA and Isosbestic signals are not available. Please ensure baseline correction has been performed.")

        # Perform linear regression
        reg = LinearRegression()
        
        n = len(self.DA_corrected)
        reg.fit(self.isosbestic_corrected.reshape(n, 1), self.DA_corrected.reshape(n, 1))
        isosbestic_fitted = reg.predict(self.isosbestic_corrected.reshape(n, 1)).reshape(n,)
        
        # Store aligned signals as a class dictionary
        self.aligned_signals = {
            "time": self.timestamps,
            "isosbestic_fitted": isosbestic_fitted,
            "DA": self.DA_corrected
        }

    def align_channels_raw(self):
        """
        Function that performs linear regression between the raw isosbestic (405 nm) and DA (465 nm) signals
        to align the fitted isosbestic signal with the DA signal. The results are stored in the class as a dictionary.
        
        This function grabs the timestamps, raw DA, and ISOS signals directly from the class attributes.
        """
        # Ensure that the raw signals are available
        if len(self.streams['DA']) == 0 or len(self.streams['ISOS']) == 0:
            raise ValueError("Raw DA and Isosbestic signals are not available. Please ensure raw data is available.")

        # Perform linear regression
        reg = LinearRegression()
        
        n = len(self.streams['DA'])
        # Fit the isosbestic signal to predict the DA signal
        reg.fit(self.streams['ISOS'].reshape(n, 1), self.streams['DA'].reshape(n, 1))
        isosbestic_fitted = reg.predict(self.streams['ISOS'].reshape(n, 1)).reshape(n,)

        # Store aligned signals as a class dictionary
        self.aligned_signals = {
            "time": self.timestamps,
            "isosbestic_fitted": isosbestic_fitted,
            "DA": self.streams['DA']
        }


    def compute_dFF(self):
        """
        Function that computes the dF/F of the fitted isosbestic and DA signals and saves it in self.dFF.

        Returns:
            df_f (arr): Relative changes of fluorescence over time.
        """
        # Access time, isosbestic, and DA from aligned_signals
        isosbestic = self.aligned_signals["isosbestic_fitted"]
        da = self.aligned_signals["DA"]
        
        # Compute dF/F by subtracting the fitted isosbestic from the DA signal
        df_f = da - isosbestic
        
        # Save the computed dF/F into the class attribute
        self.dFF = df_f
        
        return df_f


    def remove_time_segment(self, start_time, end_time):
        """
        Remove the specified time segment between start_time and end_time
        from the timestamps and associated signal streams.

        Parameters:
        start_time (float): The start time of the segment to be removed.
        end_time (float): The end time of the segment to be removed.
        """

        # Find the indices for start_time and end_time
        start_idx = np.searchsorted(self.timestamps, start_time)
        end_idx = np.searchsorted(self.timestamps, end_time)

        # Ensure valid indices
        if start_idx >= end_idx:
            raise ValueError("Invalid time segment. start_time must be less than end_time.")

        # Stitch timestamps together (exclude segment)
        self.timestamps = np.concatenate([self.timestamps[:start_idx], self.timestamps[end_idx:]])

        # Stitch DA and ISOS signals together (exclude the segment)
        self.streams[self.DA] = np.concatenate([self.streams[self.DA][:start_idx], self.streams[self.DA][end_idx:]])
        self.streams[self.ISOS] = np.concatenate([self.streams[self.ISOS][:start_idx], self.streams[self.ISOS][end_idx:]])

        # Optionally: update any other derived signals or properties that depend on the timestamps
        if hasattr(self, 'smoothed_DA'):
            self.smoothed_DA = np.concatenate([self.smoothed_DA[:start_idx], self.smoothed_DA[end_idx:]])
        if hasattr(self, 'smoothed_ISOS'):
            self.smoothed_ISOS = np.concatenate([self.smoothed_ISOS[:start_idx], self.smoothed_ISOS[end_idx:]])

        # If you have other signals like baseline-corrected or standardized ones, apply stitching similarly
        if hasattr(self, 'DA_corrected'):
            self.DA_corrected = np.concatenate([self.DA_corrected[:start_idx], self.DA_corrected[end_idx:]])
        if hasattr(self, 'isosbestic_corrected'):
            self.isosbestic_corrected = np.concatenate([self.isosbestic_corrected[:start_idx], self.isosbestic_corrected[end_idx:]])

        # Print a message to indicate successful removal
        print(f"Removed time segment from {start_time}s to {end_time}s.")
 

    def apply_ma_baseline_correction(self, window_len_seconds=30):
        """
        Applies centered moving average (MA) to both DA and ISOS signals and performs baseline correction,
        with padding to avoid shortening the signals.

        Args:
            window_len_seconds (int): The window size in seconds for the moving average filter (default: 30 seconds).
        """
        # Adjust the window length in data points
        window_len = int(self.fs) * window_len_seconds  # 30 seconds by default
        if self.smoothed_ISOS is None or self.smoothed_DA is None:
            self.smooth_and_apply()  # Smooth the signals if not already done

        # Debugging: Print initial lengths
        # print(f"Initial lengths - DA: {len(self.smoothed_DA)}, ISOS: {len(self.smoothed_ISOS)}, Timestamps: {len(self.timestamps)}")

        # Apply centered moving average with padding to both DA and ISOS streams
        self.isosbestic_fc = self.centered_moving_average_with_padding(self.smoothed_ISOS, window_len)
        # print(self.cropped_ISOS)
        self.DA_fc = self.centered_moving_average_with_padding(self.smoothed_DA, window_len)


        self.isosbestic_corrected = (self.smoothed_ISOS - self.isosbestic_fc) / self.isosbestic_fc

        # print(self.isosbestic_corrected)

        self.DA_corrected = (self.smoothed_DA- self.DA_fc) / self.DA_fc

        # Debugging: Print final lengths
        # print(f"Final lengths after padding - DA: {len(self.DA_corrected)}, ISOS: {len(self.isosbestic_corrected)}, Timestamps: {len(self.timestamps)}"


    def centered_moving_average_with_padding(self, source, window=1):
        """
        Applies a centered moving average to the input signal with edge padding to preserve the signal length.
        
        Args:
            source (np.array): The signal for which the moving average is computed.
            window (int): The window size used to compute the moving average.

        Returns:
            np.array: The centered moving average of the input signal with the original length preserved.
        """
        source = np.array(source)

        if len(source.shape) == 1:
            # Pad the signal by reflecting the edges to avoid cutting
            padded_source = np.pad(source, (window // 2, window // 2), mode='reflect')

            # Calculate the cumulative sum and moving average
            cumsum = np.cumsum(padded_source)
            moving_avg = (cumsum[window:] - cumsum[:-window]) / float(window)
            
            # Return the centered moving average with the original length preserved
            return moving_avg[:len(source)]
        else:
            raise RuntimeError(f"Input array has too many dimensions. Input: {len(source.shape)}D, Required: 1D")


    # def centered_moving_average_with_padding(self, source, window=1):
    #     """
    #     Applies a centered moving average to the input signal with edge padding to preserve the signal length.
        
    #     Args:
    #         source (np.array): The signal for which the moving average is computed.
    #         window (int): The window size used to compute the moving average.

    #     Returns:
    #         np.array: The centered moving average of the input signal with the original length preserved.
    #     """
    #     source = np.array(source)

    #     if len(source.shape) == 1:
    #         if len(source) < window:
    #             raise ValueError("Window length cannot be greater than the signal length.")

    #         # Pad the signal by reflecting the edges to avoid cutting
    #         padded_source = np.pad(source, (window // 2, window // 2), mode='reflect')

    #         # Use convolution to calculate the moving average
    #         kernel = np.ones(window) / window
    #         moving_avg = np.convolve(padded_source, kernel, mode='valid')

    #         # Ensure the length of the output matches the length of the original signal
    #         return moving_avg[:len(source)]
    #     else:
    #         raise RuntimeError(f"Input array has too many dimensions. Input: {len(source.shape)}D, Required: 1D")


    '''********************************** ZSCORE **********************************'''
    def find_baseline_period(self):
        """
        Finds the baseline period from the beginning of the timestamps array to 2 minutes after.

        Returns:
        baseline_start (float): The start time of the baseline period (always 0).
        baseline_end (float): The end time of the baseline period (2 minutes after the start).
        """
        if self.timestamps is None or len(self.timestamps) == 0:
            raise ValueError("Timestamps data is missing or empty.")
        
        # Duration of the baseline period in seconds
        baseline_duration_in_seconds = 2 * 60  + 20 # 2 minutes 20 seconds

        # Calculate the end time for the baseline period
        baseline_end_time = self.timestamps[0] + baseline_duration_in_seconds

        # Ensure the baseline period does not exceed the data length
        if baseline_end_time > self.timestamps[-1]:
            baseline_end_time = self.timestamps[-1]

        baseline_start = self.timestamps[0]
        baseline_end = baseline_end_time

        return baseline_start, baseline_end


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
            self.compute_dFF()
        
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
            "type_str": 'epocs',  # Copy type_str from an existing epoc
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


    def combine_consecutive_behaviors(self, behavior_name='all', bout_time_threshold=1, min_occurrences=1):
        """
        Combines consecutive behavior events if they occur within a specified time threshold,
        and updates the Total Duration.

        Parameters:
        - behavior_name (str): The name of the behavior to process. If 'all', process all behaviors.
        - bout_time_threshold (float): Maximum time gap (in seconds) between consecutive behaviors to be combined.
        - min_occurrences (int): Minimum number of occurrences required for a combined bout to be kept.
        """

        # Determine which behaviors to process
        if behavior_name == 'all':
            behaviors_to_process = self.behaviors.keys()  # Process all behaviors
        else:
            behaviors_to_process = [behavior_name]  # Process a single behavior

        for behavior_event in behaviors_to_process:
            behavior_onsets = np.array(self.behaviors[behavior_event].onset)
            behavior_offsets = np.array(self.behaviors[behavior_event].offset)

            combined_onsets = []
            combined_offsets = []
            combined_durations = []

            if len(behavior_onsets) == 0:
                continue  # Skip this behavior if there are no onsets

            start_idx = 0

            while start_idx < len(behavior_onsets):
                # Initialize the combination window with the first behavior onset and offset
                current_onset = behavior_onsets[start_idx]
                current_offset = behavior_offsets[start_idx]

                next_idx = start_idx + 1

                # Check consecutive events and combine them if they fall within the threshold
                while next_idx < len(behavior_onsets) and (behavior_onsets[next_idx] - current_offset) <= bout_time_threshold:
                    # Update the end of the combined bout
                    current_offset = behavior_offsets[next_idx]
                    next_idx += 1

                # Add the combined onset, offset, and total duration to the list
                combined_onsets.append(current_onset)
                combined_offsets.append(current_offset)
                combined_durations.append(current_offset - current_onset)

                # Move to the next set of events
                start_idx = next_idx

            # Filter out bouts with fewer than the minimum occurrences
            valid_indices = []
            for i in range(len(combined_onsets)):
                num_occurrences = len([onset for onset in behavior_onsets if combined_onsets[i] <= onset <= combined_offsets[i]])
                if num_occurrences >= min_occurrences:
                    valid_indices.append(i)

            # Update the behavior with the combined onsets, offsets, and durations
            self.behaviors[behavior_event].onset = [combined_onsets[i] for i in valid_indices]
            self.behaviors[behavior_event].offset = [combined_offsets[i] for i in valid_indices]
            self.behaviors[behavior_event].Total_Duration = [combined_durations[i] for i in valid_indices]  # Update Total Duration

            self.bout_dict = {}


    '''********************************** PLOTTING **********************************'''
    def plot_behavior_event(self, behavior_name, plot_type='zscore', ax=None):
        """
        Plot Delta F/F (dFF) or z-score with behavior events. Can be used to plot in a given Axes object or individually.

        Parameters:
        - behavior_name: The name of the behavior to plot. Use 'all' to plot all behaviors.
        - plot_type: The type of plot. Options are 'dFF' or 'zscore'.
        - ax: An optional matplotlib Axes object. If provided, the plot will be drawn on this Axes.
        """
        # Prepare data based on plot type
        y_data = []
        if plot_type == 'dFF':
            if self.dFF is None:
                self.compute_dFF()
            y_data = self.dFF
            y_label = r'$\Delta$F/F'
            y_title = 'dFF Signal'
        elif plot_type == 'zscore':
            if self.zscore is None:
                self.compute_zscore()
            y_data = self.zscore
            y_label = 'z-score'
            y_title = 'z-score Signal'
        else:
            raise ValueError("Invalid plot_type. Choose from 'dFF' or 'zscore'.")

        # Create plot if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(18, 6))

        # Plot the signal in black
        ax.plot(self.timestamps, np.array(y_data), linewidth=2, color='black', label=plot_type)

        # Define specific colors for behaviors
        behavior_colors = {'Investigation': 'dodgerblue', 'Approach': 'green', 'Defeat': 'red'}

        # Track the behaviors we've already labeled for the legend
        behavior_labels_plotted = set()

        # Plot behavior spans
        if behavior_name == 'all':
            for behavior_event in self.behaviors.keys():
                if behavior_event in behavior_colors:  # Make sure these are the behaviors you're interested in
                    behavior_onsets = self.behaviors[behavior_event].onset
                    behavior_offsets = self.behaviors[behavior_event].offset
                    color = behavior_colors[behavior_event]
                    
                    for on, off in zip(behavior_onsets, behavior_offsets):
                        # Only add a label the first time we encounter a behavior type
                        if behavior_event not in behavior_labels_plotted:
                            ax.axvspan(on, off, alpha=0.25, label=behavior_event, color=color)
                            behavior_labels_plotted.add(behavior_event)
                        else:
                            ax.axvspan(on, off, alpha=0.25, color=color)
        else:
            # Plot a single behavior
            behavior_event = behavior_name
            if behavior_event not in self.behaviors.keys():
                raise ValueError(f"Behavior event '{behavior_event}' not found in behaviors.")
            behavior_onsets = self.behaviors[behavior_event].onset
            behavior_offsets = self.behaviors[behavior_event].offset
            color = behavior_colors.get(behavior_event, 'dodgerblue')  # Default to blue if behavior not in the color map
            for on, off in zip(behavior_onsets, behavior_offsets):
                ax.axvspan(on, off, alpha=0.25, color=color)

        # Add labels and title
        ax.set_ylabel(y_label)
        ax.set_xlabel('Seconds')
        ax.set_title(f'{self.subject_name}: {y_title} with {behavior_name.capitalize()} Bouts' if behavior_name != 'all' else f'{self.subject_name}: {y_title} with All Behavior Events')
        
        # Only display behaviors that were actually plotted
        if len(behavior_labels_plotted) > 0:
            ax.legend()

        # Display the plot
        if ax is None:
            plt.tight_layout()
            plt.show()



    def plot_signal(self, plot_type='zscore'):
        '''
        Plots the selected signal type.

        Parameters:
        plot_type (str): The type of plot to generate. Options are 'raw', 'smoothed', 'dFF', and 'zscore'.
        '''
        total_duration = self.timestamps[-1] - self.timestamps[0]  # Total duration of the data
        tick_interval = 120  # Set the tick interval
        num_major_ticks = int(total_duration // tick_interval)

        fig, axs = plt.subplots(2, 1, figsize=(18, 8), sharex=True, dpi=200)

        if plot_type == 'raw':
            if self.DA in self.streams and self.ISOS in self.streams:
                # Plot DA signal
                axs[0].plot(self.timestamps, self.streams[self.DA], linewidth=2, color='blue', label='DA')
                mean_DA = np.mean(self.streams[self.DA])
                axs[0].axhline(mean_DA, color='red', linestyle='--', label=f'Mean DA: {mean_DA:.2f} mV')
                axs[0].set_ylabel('mV')
                axs[0].set_title(f'{self.subject_name}: DA Raw Demodulated Responses')
                axs[0].legend(loc='upper right')

                # Plot ISOS signal
                axs[1].plot(self.timestamps, self.streams[self.ISOS], linewidth=2, color='purple', label='ISOS')
                mean_ISOS = np.mean(self.streams[self.ISOS])
                axs[1].axhline(mean_ISOS, color='red', linestyle='--', label=f'Mean ISOS: {mean_ISOS:.2f} mV')
                axs[1].set_ylabel('mV')
                axs[1].legend(loc='upper right')

        elif plot_type == 'smoothed':
            if len(self.smoothed_DA) > 0 and len(self.smoothed_ISOS) > 0:
                # Plot smoothed DA signal
                axs[0].plot(self.timestamps, self.smoothed_DA, linewidth=2, color='blue', label='Smoothed DA')
                mean_DA = np.mean(self.smoothed_DA)
                axs[0].axhline(mean_DA, color='red', linestyle='--', label=f'Mean DA: {mean_DA:.2f} mV')
                axs[0].set_ylabel('mV')
                axs[0].set_title(f'{self.subject_name}: DA Smoothed Responses')
                axs[0].legend(loc='upper right')

                # Plot smoothed ISOS signal
                axs[1].plot(self.timestamps, self.smoothed_ISOS, linewidth=2, color='purple', label='Smoothed ISOS')
                mean_ISOS = np.mean(self.smoothed_ISOS)
                axs[1].axhline(mean_ISOS, color='red', linestyle='--', label=f'Mean ISOS: {mean_ISOS:.2f} mV')
                axs[1].set_ylabel('mV')
                axs[1].legend(loc='upper right')

        # Set common x-axis ticks and labels
        xticks = np.arange(self.timestamps[0], self.timestamps[-1], tick_interval)
        axs[1].set_xticks(xticks)
        xticklabels = [f"{i:.0f}s" for i in xticks]
        axs[1].set_xticklabels(xticklabels, fontsize=16, rotation=45)
        axs[0].set_xlim(self.timestamps[0], self.timestamps[-1])
        axs[1].set_xlim(self.timestamps[0], self.timestamps[-1])

        plt.tight_layout()
        plt.show()

    def plot_baseline_corrected_signal(self):
        """
        Plots the baseline-corrected DA and ISOS signals.
        """
        if self.DA_corrected is None or self.isosbestic_corrected is None:
            self.apply_ma_baseline_correction()
        
        # Define the signals to plot
        signal_DA = self.DA_corrected
        signal_ISOS = self.isosbestic_corrected
        total_duration = self.timestamps[-1] - self.timestamps[0]
        tick_interval = 30

        fig, axs = plt.subplots(2, 1, figsize=(18, 8), sharex=True, dpi=200)

        # Plot baseline-corrected DA signal
        axs[0].plot(self.timestamps, signal_DA, linewidth=2, color='blue', label='DA (Baseline-Corrected)')
        mean_DA = np.mean(signal_DA)
        axs[0].axhline(mean_DA, color='red', linestyle='--', label=f'Mean DA: {mean_DA:.2f} mV')
        axs[0].set_ylabel('mV')
        axs[0].set_title(f'{self.subject_name}: Baseline-Corrected DA Signal')
        axs[0].legend(loc='upper right')

        # Plot baseline-corrected ISOS signal
        axs[1].plot(self.timestamps, signal_ISOS, linewidth=2, color='purple', label='ISOS (Baseline-Corrected)')
        mean_ISOS = np.mean(signal_ISOS)
        axs[1].axhline(mean_ISOS, color='red', linestyle='--', label=f'Mean ISOS: {mean_ISOS:.2f} mV')
        axs[1].set_ylabel('mV')
        axs[1].set_title(f'{self.subject_name}: Baseline-Corrected ISOS Signal')
        axs[1].legend(loc='upper right')

        xticks = np.arange(self.timestamps[0], self.timestamps[-1], tick_interval)
        axs[1].set_xticks(xticks)
        xticklabels = [f"{i:.0f}s" for i in xticks]
        axs[1].set_xticklabels(xticklabels, fontsize=16, rotation=45)
        axs[0].set_xlim(self.timestamps[0], self.timestamps[-1])
        axs[1].set_xlim(self.timestamps[0], self.timestamps[-1])

        plt.tight_layout()
        plt.show()

    def plot_standardization(self):
        """
        Plots the standardized isosbestic and calcium (DA) signals.
        """
        if self.isosbestic_standardized is None or self.calcium_standardized is None:
            print("Standardized signals not available. Please perform standardization first.")
            return

        fig, axs = plt.subplots(2, 1, figsize=(18, 8), dpi=200)

        x = self.timestamps
        xticks = np.arange(0, len(x), len(x) // 10)
        xticklabels = [f'{int(tick/self.fs):.0f}s' for tick in xticks]

        axs[0].plot(x, self.isosbestic_standardized, alpha=0.8, c='purple', lw=1.5)
        axs[0].axhline(0, color='black', linestyle='--', lw=1.0)
        axs[0].set_xticks(xticks)
        axs[0].set_xticklabels(xticklabels, fontsize=16, rotation=45)
        axs[0].set_ylabel("z-score")
        axs[0].set_title("Standardized Isosbestic Signal")

        axs[1].plot(x, self.calcium_standardized, alpha=0.8, c='blue', lw=1.5)
        axs[1].axhline(0, color='black', linestyle='--', lw=1.0)
        axs[1].set_xticks(xticks)
        axs[1].set_xticklabels(xticklabels, fontsize=16, rotation=45)
        axs[1].set_ylabel("z-score")
        axs[1].set_title("Standardized DA Signal (Calcium)")
        axs[1].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show()

    def plot_aligned_signals(self):
        """
        Function that plots the aligned isosbestic_fitted and DA_corrected signals.
        """
        if not hasattr(self, 'aligned_signals') or len(self.aligned_signals) == 0:
            raise ValueError("Aligned signals not found. Please run 'align_channels' before plotting.")

        x = self.aligned_signals["time"]
        isosbestic_fitted = self.aligned_signals["isosbestic_fitted"]
        DA = self.aligned_signals["DA"]

        x_max = x[-1]

        fig, ax0 = plt.subplots(figsize=(18, 8), dpi=200)
        
        # Plot DA and isosbestic signals
        ax0.plot(x, DA, alpha=0.8, c='blue', lw=2, zorder=0, label="DA")
        ax0.plot(x, isosbestic_fitted, alpha=0.8, c='purple', lw=2, zorder=1, label="Isosbestic (Fitted)")
        
        ax0.axhline(0, color="black", lw=1.5)

        ax0.set_xlim(0, x_max)
        ax0.set_xlabel("Time (s)", fontsize=12)

        ax0.set_ylabel("Change in signal (%)", fontsize=12)
        ax0.set_ylim(min(np.min(isosbestic_fitted), np.min(DA)) - 0.05, max(np.max(isosbestic_fitted), np.max(DA)) + 0.05)

        ax0.legend(loc=2, fontsize=12)
        ax0.set_title("Alignment of Isosbestic and DA signals", fontsize=14)
        ax0.tick_params(axis='both', which='major', labelsize=10)

        plt.tight_layout()
        plt.show()


    def plot_dFF_train_times(self, train_times):
        """
        Function to plot the computed dF/F signal stored in self.dFF and plot train_times as dashed vertical lines.
        
        Parameters:
        - train_times: Array of time points to mark as dashed vertical lines.
        """
        if self.dFF is None:
            raise ValueError("dF/F not computed. Please run compute_dFF() before plotting.")

        x = self.aligned_signals["time"]
        df_f = self.dFF
        max_x = x[-1]

        fig, ax0 = plt.subplots(figsize=(18, 8), dpi=200)
        
        # Plot the ΔF/F signal
        ax0.plot(x, df_f, alpha=0.8, c="green", lw=2, label="ΔF/F")
        ax0.axhline(0, color="black", lw=1.5)
        
        # Plot train_times as dashed vertical lines
        for train_time in train_times:
            ax0.axvline(train_time, color='blue', linestyle='--', linewidth=1.5, label='Train Time' if train_time == train_times[0] else "")

        ax0.set_xlim(0, max_x)
        ax0.set_xlabel("Time (s)", fontsize=12)

        ax0.set_ylim(min(df_f) - 0.05, max(df_f) + 0.05)
        ax0.set_ylabel(r"$\Delta$F/F", fontsize=12)

        ax0.set_title(r"$\Delta$F/F Signal", fontsize=14)
        ax0.legend(loc=2, fontsize=12)

        plt.tight_layout()
        plt.show()

    def plot_dFF(self):
        """
        Function to plot the computed dF/F signal stored in self.dFF.
        """
        if self.dFF is None:
            raise ValueError("dF/F not computed. Please run compute_dFF() before plotting.")

        x = self.aligned_signals["time"]
        df_f = self.dFF
        max_x = x[-1]

        fig, ax0 = plt.subplots(figsize=(18, 8), dpi=200)
        
        ax0.plot(x, df_f, alpha=0.8, c="green", lw=2, label="ΔF/F")
        ax0.axhline(0, color="black", lw=1.5)
        
        ax0.set_xlim(0, max_x)
        ax0.set_xlabel("Time (s)", fontsize=12)

        ax0.set_ylim(min(df_f) - 0.05, max(df_f) + 0.05)
        ax0.set_ylabel(r"$\Delta$F/F", fontsize=12)

        ax0.set_title(r"$\Delta$F/F Signal", fontsize=14)
        ax0.legend(loc=2, fontsize=12)

        plt.tight_layout()
        plt.show()

    def plot_zscore(self):
        """
        Plots the z-score of the Delta F/F (dFF) signal.

        This function generates a standalone plot of the z-score signal, without any additional subplots.
        """
        if self.zscore is not None and len(self.zscore) > 0:
            # Create the figure for the z-score plot
            fig, ax = plt.subplots(figsize=(18, 8), dpi=200)

            # Plot the z-score signal
            ax.plot(self.timestamps, self.zscore, linewidth=2, color='black', label='z-score')

            # Set labels and title
            ax.set_ylabel('z-score', fontsize=16)
            ax.set_xlabel('Seconds', fontsize=16)
            ax.set_title(f'{self.subject_name}: Z-score of Delta F/F (dFF) Signal', fontsize=18)

            # Set x-ticks at an interval of 120 seconds
            tick_interval = 120
            xticks = np.arange(self.timestamps[0], self.timestamps[-1], tick_interval)
            ax.set_xticks(xticks)
            xticklabels = [f"{i:.0f}s" for i in xticks]
            ax.set_xticklabels(xticklabels, fontsize=14, rotation=45)

            # Set limits for the x-axis
            ax.set_xlim(self.timestamps[0], self.timestamps[-1])

            # Add a legend
            ax.legend(loc='upper right', fontsize=14)

            # Display the plot with tight layout
            plt.tight_layout()
            plt.show()
        else:
            print("Z-score data not available. Please compute z-score first.")

    '''********************************** First Behavior **********************************'''
    def get_first_behavior(self, behaviors=['Investigation']):
        """
        Extracts the first 'Investigation' behavior event from each bout and stores it in the class.
        
        Parameters:
        - behaviors (list): List of behavior names to track (defaults to ['Investigation']).
        
        Populates:
        - first_behavior_dict: Dictionary where each key is the bout name and the value contains the first behavior event details.
        """
        first_behavior_dict = {}

        # Loop through each bout in the bout_dict
        for bout_name, bout_data in self.bout_dict.items():
            first_behavior_dict[bout_name] = {}  # Initialize the dictionary for this bout
            
            # Loop through the behaviors
            for behavior in behaviors:
                # Check if the behavior exists in bout_data and contains valid event data
                if behavior in bout_data and isinstance(bout_data[behavior], list) and len(bout_data[behavior]) > 0:
                    # Get the first event for the behavior
                    first_event = bout_data[behavior][0]  # Assuming the list contains behavior events
                    
                    # Extract the relevant details
                    first_behavior_dict[bout_name][behavior] = {
                        'Start Time': first_event['Start Time'],
                        'End Time': first_event['End Time'],
                        'Total Duration': first_event['End Time'] - first_event['Start Time'],
                        'Mean zscore': first_event.get('Mean zscore', None)
                    }
                else:
                    # If the behavior doesn't exist in this bout, fill in with None
                    first_behavior_dict[bout_name][behavior] = {
                        'Start Time': None,
                        'End Time': None,
                        'Total Duration': None,
                        'Mean zscore': None
                    }

        self.first_behavior_dict = first_behavior_dict


    def remove_short_behaviors(self, behavior_name='all', min_duration=0):
        """
        Removes behaviors that have a total duration less than the specified minimum duration,
        and updates the Total Duration for the remaining behaviors.

        Parameters:
        - behavior_name (str): The name of the behavior to process. If 'all', process all behaviors.
        - min_duration (float): Minimum duration in seconds for a behavior to be retained.
        """

        # Determine which behaviors to process
        if behavior_name == 'all':
            behaviors_to_process = self.behaviors.keys()  # Process all behaviors
        else:
            behaviors_to_process = [behavior_name]  # Process a single behavior

        for behavior_event in behaviors_to_process:
            behavior_onsets = np.array(self.behaviors[behavior_event].onset)
            behavior_offsets = np.array(self.behaviors[behavior_event].offset)

            if len(behavior_onsets) == 0:
                continue  # Skip if there are no events for this behavior

            # Calculate the durations of each behavior
            behavior_durations = behavior_offsets - behavior_onsets

            # Filter events based on the minimum duration
            valid_indices = np.where(behavior_durations >= min_duration)[0]

            # Update the behavior's onsets, offsets, and durations with only the valid events
            self.behaviors[behavior_event].onset = behavior_onsets[valid_indices].tolist()
            self.behaviors[behavior_event].offset = behavior_offsets[valid_indices].tolist()
            self.behaviors[behavior_event].Total_Duration = (behavior_offsets[valid_indices] - behavior_onsets[valid_indices]).tolist()  # Update Total Duration


#******************************PETHS**************************************
    def compute_1st_event_peth(self, behavior_name, pre_time=5, post_time=5, bin_size=0.1):
        """
        Computes the peri-event time histogram (PETH) data for the first occurrence of a given event in each block.
        Stores the peri-event data (custom zscore, dFF, and time axis) as a class variable.

        Z-score is calculated using the pre-time window as the baseline.

        Parameters:
        behavior_name (str): The name of the event to generate the PETH for (e.g., 'Investigation').
        pre_time (float): The time in seconds to include before the event.
        post_time (float): The time in seconds to include after the event.
        bin_size (float): The size of each bin in the histogram (in seconds).

        Returns:
        None. Stores peri-event data as a class variable.
        """
        if behavior_name not in self.behaviors:
            print(f"Event {behavior_name} not found in behaviors.")
            return

        # Ensure ΔF/F is computed (we will calculate the z-score manually using the pre-time as baseline)
        if self.dFF is None:
            self.compute_dFF()
        
        self.dFF = np.array(self.dFF)
        
        # Extract the first onset of the event from the behaviors
        event_onsets = self.behaviors[behavior_name].onset
        if len(event_onsets) == 0:
            print(f"No occurrences of {behavior_name} found.")
            return

        first_event_onset = event_onsets[0]  # Get the first occurrence

        # Find the peri-event window around the first event
        start_time = first_event_onset - pre_time
        end_time = first_event_onset + post_time
        
        start_idx = np.searchsorted(self.timestamps, start_time)
        end_idx = np.searchsorted(self.timestamps, end_time)

        # If the event is too close to the start or end of the recording, skip it
        if start_idx < 0 or end_idx >= len(self.timestamps):
            print("First event is too close to the edge of the recording, skipping.")
            return

        # Define the baseline window for z-score calculation (from pre-time to the event onset)
        baseline_end_idx = np.searchsorted(self.timestamps, first_event_onset)
        baseline_dff = self.dFF[start_idx:baseline_end_idx]  # ΔF/F values during the baseline period

        # Calculate the mean and standard deviation for the baseline period
        baseline_mean = np.mean(baseline_dff)
        baseline_std = np.std(baseline_dff)

        if baseline_std == 0:
            print("Baseline standard deviation is 0. Cannot compute z-score.")
            return

        # Extract the ΔF/F values for the peri-event window
        peri_event_dff = self.dFF[start_idx:end_idx]

        # Calculate z-score using the baseline mean and std
        peri_event_zscore = (peri_event_dff - baseline_mean) / baseline_std

        # Generate time axis for the peri-event window
        time_axis = np.linspace(-pre_time, post_time, len(peri_event_zscore))

        # Store both peri-event zscore, dFF, and time axis in a class variable dictionary
        self.peri_event_data = {
            'zscore': peri_event_zscore,
            'dFF': peri_event_dff,
            'time_axis': time_axis
        }


    def compute_nth_bout_baseline_peth(self, bout_name, behavior_name, pre_time=5, post_time=5, bin_size=0.1, nth_event=1):
        """
        Computes the peri-event time histogram (PETH) data for the nth occurrence of a given behavior in a specific bout,
        using baseline z-scoring based on the pre-event period.

        Parameters:
        bout_name (str): The name of the bout to analyze.
        behavior_name (str): The name of the behavior to analyze.
        pre_time (float): The time in seconds to include before the event.
        post_time (float): The time in seconds to include after the event.
        bin_size (float): The size of each bin in the histogram (in seconds).
        nth_event (int): The nth occurrence of the behavior to analyze.

        Returns:
        dict: A dictionary containing 'zscore', 'dFF', and 'time_axis' arrays.
        """
        if bout_name not in self.bout_dict or behavior_name not in self.bout_dict[bout_name]:
            print(f"No {behavior_name} found in {bout_name}.")
            return None

        behavior_events = self.bout_dict[bout_name][behavior_name]
        
        if len(behavior_events) < nth_event:
            print(f"Less than {nth_event} events found for {behavior_name} in {bout_name}.")
            return None

        # Get the nth event's time
        event_time = behavior_events[nth_event - 1]['Start Time']

        # Find the peri-event window around the event
        start_time = event_time - pre_time
        end_time = event_time + post_time

        start_idx = np.searchsorted(self.timestamps, start_time)
        end_idx = np.searchsorted(self.timestamps, end_time)

        # Handle cases where the event is too close to the start or end of the recording by padding
        pad_start, pad_end = 0, 0

        if start_idx < 0:
            pad_start = abs(start_idx)  # Padding at the start
            start_idx = 0
        if end_idx >= len(self.timestamps):
            pad_end = end_idx - len(self.timestamps) + 1  # Padding at the end
            end_idx = len(self.timestamps)

        # Define the baseline window for z-score calculation (from pre-time to the event start)
        baseline_end_idx = np.searchsorted(self.timestamps, event_time)
        self.dFF = np.array(self.dFF)
        baseline_dff = self.dFF[start_idx:baseline_end_idx]  # ΔF/F values during the baseline period

        # Calculate the mean and standard deviation for the baseline period
        baseline_mean = np.nanmean(baseline_dff)
        baseline_std = np.nanstd(baseline_dff)

        if baseline_std == 0 or np.isnan(baseline_std):
            print("Baseline standard deviation is 0 or NaN. Cannot compute z-score.")
            return None

        # Extract the ΔF/F values for the peri-event window
        peri_event_dff = self.dFF[start_idx:end_idx]

        # Calculate z-score using the baseline mean and std
        peri_event_zscore = (peri_event_dff - baseline_mean) / baseline_std

        # Apply padding if necessary
        if pad_start > 0:
            peri_event_zscore = np.pad(peri_event_zscore, (pad_start, 0), mode='constant', constant_values=np.nan)
        if pad_end > 0:
            peri_event_zscore = np.pad(peri_event_zscore, (0, pad_end), mode='constant', constant_values=np.nan)

        # Generate the time axis for the peri-event window with padding applied
        time_axis = np.linspace(-pre_time, post_time, len(peri_event_zscore))

        # Return the peri-event data as a dictionary
        peri_event_data = {
            'zscore': peri_event_zscore,
            'dFF': peri_event_dff,
            'time_axis': time_axis
        }

        return peri_event_data



    def plot_1st_event_peth(self, signal_type='zscore'):
        """
        Plots the peri-event time histogram (PETH) based on the previously computed data.

        Parameters:
        signal_type (str): The type of signal to plot. Options are 'zscore' or 'dFF'.

        Returns:
        None. Displays the PETH plot.
        """
        # Ensure that peri-event data is already computed
        if not hasattr(self, 'peri_event_data'):
            print("No peri-event data found. Please compute PETH first using compute_first_event_peth.")
            return

        # Extract time axis and the desired signal type from the stored peri-event data
        time_axis = self.peri_event_data['time_axis']
        
        plt.figure(figsize=(10, 6))
        
        if signal_type == 'zscore':
            plt.plot(time_axis, self.peri_event_data['zscore'], color='blue', label='Z-score')
            ylabel = 'Z-scored ΔF/F'
        elif signal_type == 'dFF':
            plt.plot(time_axis, self.peri_event_data['dFF'], color='green', label='ΔF/F')
            ylabel = r'$\Delta$F/F'
        else:
            print("Invalid signal_type. Use 'zscore' or 'dFF'.")
            return

        plt.axvline(0, color='black', linestyle='--', label='Event onset')
        plt.xlabel('Time (s)')
        plt.ylabel(ylabel)
        plt.title('Peri-Event Time Histogram (PETH)')
        plt.legend()
        plt.tight_layout()
        plt.show()
