import os
import tdt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from single_tdt_class import *
import sys
import seaborn as sns
import scipy.stats as stats

root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))  # Go up one directory to P2_Code
# Add the root directory to sys.path
sys.path.append(root_dir)

class GroupTDTData:
    def __init__(self, experiment_folder_path, csv_base_path):
        """
        Initializes the GroupTDTData object with paths.
        """
        self.experiment_folder_path = experiment_folder_path
        self.csv_base_path = csv_base_path
        self.blocks = {}
        self.group_psth = None

        self.parameters = {}  # Initialize the parameters log
        self.load_blocks()

        # Hab Dishab
        self.hab_dishab_df = pd.DataFrame()
        # Hc - Home Cage
        self.hc_df = pd.DataFrame()
    
    from hab_dishab.hab_dishab_extension import hab_dishab_processing, hab_dishab_plot_individual_behavior
    # from P2_Code.social_pref. import 
    from home_cage.home_cage_extension import hc_processing, hc_plot_individual_behavior
    from social_pref.social_pref_extension import sp_processing, sp_compute_first_bout_peth_all_blocks,sp_plot_first_investigation_vs_zscore_4s
    from defeat.defeat_extension import d_proc_processing, d_proc_plot_individual_behavior
    from reward_training.reward_training_extension import rt_processing, rt_plot_individual_behavior, rt_extract_and_plot, rt_compute_peth_per_event, rt_plot_peth_per_event
    from experiment_functions import extract_nth_to_mth_behavior_mean_da_baseline
    from defeat.defeat_extension import plot_peth_individual_traces
    from aggression.aggression_extension import ag_proc_processing_all_blocks, compute_nth_bout_peth_all_blocks_standard_zscore


    def load_blocks(self):
        """
        Loads each block folder as a TDTData object.
        """
        block_folders = [folder for folder in os.listdir(self.experiment_folder_path)
                         if os.path.isdir(os.path.join(self.experiment_folder_path, folder))]

        for block_folder in block_folders:
            block_path = os.path.join(self.experiment_folder_path, block_folder)
            block_data = tdt.read_block(block_path)
            self.blocks[block_folder] = TDTData(block_data, block_path)

    def get_block(self, block_name):
        """
        Retrieves a specific block by its folder name.
        """
        return self.blocks.get(block_name, None)

    def list_blocks(self):
        """
        Lists all the block names available in the group.
        """
        return list(self.blocks.keys())

    def remove_time_segments_from_block(self, block_name, time_segments):
        """
        Remove specified time segments from a given block's data.

        Parameters:
        block_name (str): The name of the block (file) to remove time segments from.
        time_segments (list): A list of tuples representing the time segments to remove [(start_time, end_time), ...].
        """
        tdt_data_obj = self.blocks.get(block_name, None)
        if tdt_data_obj is None:
            print(f"Block {block_name} not found.")
            return

        for (start_time, end_time) in time_segments:
            tdt_data_obj.remove_time_segment(start_time, end_time)

        print(f"Removed specified time segments from block {block_name}.")

    def batch_process(self, remove_led_artifact=True, t=30, time_segments_to_remove=None):
        """
        Batch processes the TDT data by extracting behaviors, removing LED artifacts, and computing z-score.
        """
        for block_folder, tdt_data_obj in self.blocks.items():
            csv_file_name = f"{block_folder}.csv"
            csv_file_path = os.path.join(self.csv_base_path, csv_file_name)
            # Check if the subject name is in the time_segments_to_remove dictionary
            if time_segments_to_remove and tdt_data_obj.subject_name in time_segments_to_remove:
                # Remove specific time segments for this block
                self.remove_time_segments_from_block(block_folder, time_segments_to_remove[tdt_data_obj.subject_name])

            print(f"Processing {block_folder}...")
            if remove_led_artifact:
                tdt_data_obj.remove_initial_LED_artifact(t=t)
                tdt_data_obj.remove_final_data_segment(t = 10)
            
            tdt_data_obj.smooth_and_apply(window_len=int(tdt_data_obj.fs)*2)
            tdt_data_obj.apply_ma_baseline_correction()
            tdt_data_obj.align_channels()
            tdt_data_obj.compute_dFF()
            baseline_start, baseline_end = tdt_data_obj.find_baseline_period()  
            # print(baseline_start)
            # print(baseline_end) 
            # tdt_data_obj.compute_zscore(method = 'baseline', baseline_start = baseline_start, baseline_end = baseline_end)
            tdt_data_obj.compute_zscore(method = 'standard')
            tdt_data_obj.extract_manual_annotation_behaviors(csv_file_path)
            tdt_data_obj.remove_short_behaviors(behavior_name='all', min_duration=0.2)
            # tdt_data_obj.ag_extract_aggression_events(csv_file_path)
            tdt_data_obj.combine_consecutive_behaviors(behavior_name='all', bout_time_threshold=2, min_occurrences=1)


            tdt_data_obj.verify_signal()
            # tdt_data_obj.zscore = None

    '''********************************** BEHAVIORS **********************************'''
    def plot_all_behavior_vs_dff_all(self, behavior_name='Investigation', min_duration=0.0, max_duration=np.inf):
        """
        Plot the specified behavior duration vs. mean Z-scored ΔF/F during all occurrences of that behavior for all blocks,
        color-coded by individual subject identity. Only includes behavior events longer than min_duration and shorter than 
        max_duration seconds.

        Parameters:
        behavior_name (str): The name of the behavior to analyze (e.g., 'Investigation', 'Approach', etc.).
        min_duration (float): The minimum duration of behavior to include in the plot.
        max_duration (float): The maximum duration of behavior to include in the plot.
        """
        behavior_durations = []
        mean_zscored_dffs = []
        subject_names = []

        # Loop through each block in self.blocks
        for block_name, block_data in self.blocks.items():
            if block_data.bout_dict:  # Make sure bout_dict is populated
                for bout, behavior_data in block_data.bout_dict.items():
                    if behavior_name in behavior_data:
                        # Loop through all events of the specified behavior in this bout
                        for event in behavior_data[behavior_name]:
                            duration = event['Total Duration']
                            # Only include behavior events longer than min_duration and shorter than max_duration
                            if min_duration < duration < max_duration:  
                                # Extract behavior duration and mean DA for this event
                                behavior_durations.append(duration)
                                mean_zscored_dffs.append(event['Mean zscore'])
                                subject_names.append(block_data.subject_name)  # Block name as the subject identifier

        # Convert lists to numpy arrays
        behavior_durations = np.array(behavior_durations, dtype=np.float64)
        mean_zscored_dffs = np.array(mean_zscored_dffs, dtype=np.float64)
        subject_names = np.array(subject_names)

        # Filter out any entries where either behavior_durations or mean_zscored_dffs is NaN
        valid_indices = ~np.isnan(behavior_durations) & ~np.isnan(mean_zscored_dffs)
        behavior_durations = behavior_durations[valid_indices]
        mean_zscored_dffs = mean_zscored_dffs[valid_indices]
        subject_names = subject_names[valid_indices]

        if len(mean_zscored_dffs) == 0 or len(behavior_durations) == 0:
            print("No valid data points for correlation.")
            return

        # Calculate Pearson correlation
        r, p = stats.pearsonr(mean_zscored_dffs, behavior_durations)

        # Get unique subjects and assign colors
        unique_subjects = np.unique(subject_names)
        color_palette = sns.color_palette("hsv", len(unique_subjects))
        subject_color_map = {subject: color_palette[i] for i, subject in enumerate(unique_subjects)}

        # Plotting the scatter plot
        plt.figure(figsize=(12, 6))
        
        for subject in unique_subjects:
            # Create a mask for each subject
            mask = subject_names == subject
            plt.scatter(mean_zscored_dffs[mask], behavior_durations[mask], 
                        color=subject_color_map[subject], label=subject, alpha=0.6)

        # Adding the regression line
        slope, intercept = np.polyfit(mean_zscored_dffs, behavior_durations, 1)
        plt.plot(mean_zscored_dffs, slope * mean_zscored_dffs + intercept, color='black', linestyle='--')

        # Add labels and title
        plt.xlabel(f'Mean Z-scored ΔF/F during {behavior_name.lower()} events')
        plt.ylabel(f'{behavior_name} duration (s)')
        plt.title(f'Correlation between {behavior_name} Duration and DA Response (All {behavior_name}s > {min_duration}s and < {max_duration}s)')

        # Display Pearson correlation and p-value
        plt.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.2e}\nn = {len(mean_zscored_dffs)} sessions',
                transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

        # Add a legend with subject names
        plt.legend(title='Subject', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()


    def plot_all_behavior_vs_dff_all_max_time(self, behavior_name='Investigation', min_duration=0.0, max_analysis_time=np.inf):
        """
        Plot the specified behavior duration vs. mean Z-scored ΔF/F during all occurrences of that behavior for all blocks,
        color-coded by individual subject identity. Only includes behavior events longer than min_duration seconds.
        Mean DA is calculated from the behavior onset to a limited time defined by max_analysis_time.

        Parameters:
        behavior_name (str): The name of the behavior to analyze (e.g., 'Investigation', 'Approach', etc.).
        min_duration (float): The minimum duration of behavior to include in the plot.
        max_analysis_time (float): The maximum amount of time, starting from the behavior onset, to calculate mean DA.
        """
        behavior_durations = []
        mean_zscored_dffs = []
        subject_names = []

        # Loop through each block in self.blocks
        for block_name, block_data in self.blocks.items():
            if block_data.bout_dict:  # Make sure bout_dict is populated
                for bout, behavior_data in block_data.bout_dict.items():
                    if behavior_name in behavior_data:
                        # Loop through all events of the specified behavior in this bout
                        for event in behavior_data[behavior_name]:
                            duration = event['Total Duration']
                            # Only include behavior events longer than min_duration
                            if duration >= min_duration:
                                event_start = event['Start Time']
                                # Determine the end of the analysis window based on max_analysis_time
                                analysis_end_time = min(event_start + max_analysis_time, event['End Time'])

                                # Get the z-score signal during the allowed analysis window
                                zscore_indices = (block_data.timestamps >= event_start) & (block_data.timestamps <= analysis_end_time)
                                mean_da = np.mean(block_data.zscore[zscore_indices])  # Compute mean DA within the allowed time window

                                # Append the duration and mean DA to the lists
                                behavior_durations.append(duration)
                                mean_zscored_dffs.append(mean_da)
                                subject_names.append(block_data.subject_name)  # Block name as the subject identifier

        # Convert lists to numpy arrays
        behavior_durations = np.array(behavior_durations, dtype=np.float64)
        mean_zscored_dffs = np.array(mean_zscored_dffs, dtype=np.float64)
        subject_names = np.array(subject_names)

        # Filter out any entries where either behavior_durations or mean_zscored_dffs is NaN
        valid_indices = ~np.isnan(behavior_durations) & ~np.isnan(mean_zscored_dffs)
        behavior_durations = behavior_durations[valid_indices]
        mean_zscored_dffs = mean_zscored_dffs[valid_indices]
        subject_names = subject_names[valid_indices]

        if len(mean_zscored_dffs) == 0 or len(behavior_durations) == 0:
            print("No valid data points for correlation.")
            return

        # Calculate Pearson correlation
        r, p = stats.pearsonr(mean_zscored_dffs, behavior_durations)

        # Get unique subjects and assign colors
        unique_subjects = np.unique(subject_names)
        color_palette = sns.color_palette("hsv", len(unique_subjects))
        subject_color_map = {subject: color_palette[i] for i, subject in enumerate(unique_subjects)}

        # Plotting the scatter plot
        plt.figure(figsize=(12, 6))
        
        for subject in unique_subjects:
            # Create a mask for each subject
            mask = subject_names == subject
            plt.scatter(mean_zscored_dffs[mask], behavior_durations[mask], 
                        color=subject_color_map[subject], label=subject, alpha=0.6)

        # Adding the regression line
        slope, intercept = np.polyfit(mean_zscored_dffs, behavior_durations, 1)
        plt.plot(mean_zscored_dffs, slope * mean_zscored_dffs + intercept, color='black', linestyle='--')

        # Add labels and title
        plt.xlabel(f'Mean Z-scored ΔF/F during {behavior_name.lower()} events (limited to {max_analysis_time}s from onset)')
        plt.ylabel(f'{behavior_name} duration (s)')
        plt.title(f'Correlation between {behavior_name} Duration and DA Response (Max Analysis Time: {max_analysis_time}s)')

        # Display Pearson correlation and p-value
        plt.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.2e}\nn = {len(mean_zscored_dffs)} sessions',
                transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

        # Add a legend with subject names
        plt.legend(title='Subject', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()


    def plot_all_behavior_vs_dff_all_with_flexible_time(self, behavior_name='Investigation', min_duration=0.0, max_analysis_time=2.0):
        """
        Plot the specified behavior duration vs. mean Z-scored ΔF/F during all occurrences of that behavior for all blocks,
        color-coded by individual subject identity. Only includes behavior events longer than min_duration seconds.
        Mean DA is calculated from the behavior onset up to max_analysis_time or the behavior duration, whichever is shorter.

        Parameters:
        behavior_name (str): The name of the behavior to analyze (e.g., 'Investigation', 'Approach', etc.).
        min_duration (float): The minimum duration of behavior to include in the plot.
        max_analysis_time (float): The maximum amount of time, starting from the behavior onset, to calculate mean DA.
                                If the behavior is shorter than max_analysis_time, the actual behavior duration is used.
        """
        behavior_durations = []
        mean_zscored_dffs = []
        subject_names = []

        # Loop through each block in self.blocks
        for block_name, block_data in self.blocks.items():
            if block_data.bout_dict:  # Make sure bout_dict is populated
                for bout, behavior_data in block_data.bout_dict.items():
                    if behavior_name in behavior_data:
                        # Loop through all events of the specified behavior in this bout
                        for event in behavior_data[behavior_name]:
                            duration = event['Total Duration']
                            # Only include behavior events longer than min_duration
                            if duration >= min_duration:
                                event_start = event['Start Time']
                                # Use the actual duration of the behavior if it's shorter than max_analysis_time
                                analysis_duration = min(duration, max_analysis_time)
                                analysis_end_time = event_start + analysis_duration

                                # Get the z-score signal during the allowed analysis window
                                zscore_indices = (block_data.timestamps >= event_start) & (block_data.timestamps <= analysis_end_time)
                                mean_da = np.mean(block_data.zscore[zscore_indices])  # Compute mean DA within the allowed time window

                                # Append the duration and mean DA to the lists
                                behavior_durations.append(duration)
                                mean_zscored_dffs.append(mean_da)
                                subject_names.append(block_data.subject_name)  # Block name as the subject identifier

        # Convert lists to numpy arrays
        behavior_durations = np.array(behavior_durations, dtype=np.float64)
        mean_zscored_dffs = np.array(mean_zscored_dffs, dtype=np.float64)
        subject_names = np.array(subject_names)

        # Filter out any entries where either behavior_durations or mean_zscored_dffs is NaN
        valid_indices = ~np.isnan(behavior_durations) & ~np.isnan(mean_zscored_dffs)
        behavior_durations = behavior_durations[valid_indices]
        mean_zscored_dffs = mean_zscored_dffs[valid_indices]
        subject_names = subject_names[valid_indices]

        if len(mean_zscored_dffs) == 0 or len(behavior_durations) == 0:
            print("No valid data points for correlation.")
            return

        # Calculate Pearson correlation
        r, p = stats.pearsonr(mean_zscored_dffs, behavior_durations)

        # Get unique subjects and assign colors
        unique_subjects = np.unique(subject_names)
        color_palette = sns.color_palette("hsv", len(unique_subjects))
        subject_color_map = {subject: color_palette[i] for i, subject in enumerate(unique_subjects)}

        # Plotting the scatter plot
        plt.figure(figsize=(12, 6))
        
        for subject in unique_subjects:
            # Create a mask for each subject
            mask = subject_names == subject
            plt.scatter(mean_zscored_dffs[mask], behavior_durations[mask], 
                        color=subject_color_map[subject], label=subject, alpha=0.6)

        # Adding the regression line
        slope, intercept = np.polyfit(mean_zscored_dffs, behavior_durations, 1)
        plt.plot(mean_zscored_dffs, slope * mean_zscored_dffs + intercept, color='black', linestyle='--')

        # Add labels and title
        plt.xlabel(f'Mean Z-scored ΔF/F during {behavior_name.lower()} events (up to {max_analysis_time}s)')
        plt.ylabel(f'{behavior_name} duration (s)')
        plt.title(f'Correlation between {behavior_name} Duration and DA Response (Max Analysis Time: {max_analysis_time}s or Actual Duration)')

        # Display Pearson correlation and p-value
        plt.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.2e}\nn = {len(mean_zscored_dffs)} sessions',
                transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

        # Add a legend with subject names
        plt.legend(title='Subject', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()


    def plot_first_investigation_vs_dff(self, bouts=None, behavior_name='Investigation'):
        """
        Plot the first occurrence of the specified behavior duration vs. mean Z-scored ΔF/F relative to the baseline during 
        the duration of the event for all blocks, color-coded by bout type.

        Parameters:
        behavior_name (str): The name of the behavior to analyze (default is 'Investigation').
        bouts (list): A list of bout names to include in the analysis.
        """
        if bouts is None:
            bouts = ['Short_Term_1', 'Novel_1',]

        mean_zscored_dffs = []
        behavior_durations = []
        bout_names = []

        # Step 1: Compute the PETH for the first occurrence of 'Investigation'
        self.compute_nth_bout_peth_all_blocks(
            behavior_name=behavior_name,
            nth_occurrence=1,
            bouts=bouts,
            pre_time=4,
            post_time=4
        )

        # Step 2: Extract mean Z-scored ΔF/F relative to baseline for the duration of the event
        for block_name, bout_data in self.peri_event_data_all_blocks.items():
            for bout, peri_event_data in bout_data.items():
                time_axis = peri_event_data['time_axis']
                zscore = peri_event_data['zscore']

                # Get the duration of the first investigation event for this bout
                investigation_duration = None
                if bout in self.blocks[block_name].bout_dict and 'Investigation' in self.blocks[block_name].bout_dict[bout]:
                    first_investigation = self.blocks[block_name].bout_dict[bout]['Investigation'][0]
                    investigation_duration = first_investigation['Total Duration']

                    # Adjust window indices based on the duration of the event, limited to 4 seconds maximum
                    window_end = min(investigation_duration, 4)
                    window_indices = (time_axis >= 0) & (time_axis <= window_end)

                    # Calculate the mean z-scored ΔF/F for the duration of the event
                    mean_zscore = np.mean(zscore[window_indices])

                    # Store the results
                    if investigation_duration is not None and mean_zscore is not None:
                        mean_zscored_dffs.append(mean_zscore)
                        behavior_durations.append(investigation_duration)
                        bout_names.append(bout)  # Use bout name as the identifier for color-coding

        # Convert lists to numpy arrays
        mean_zscored_dffs = np.array(mean_zscored_dffs, dtype=np.float64)
        behavior_durations = np.array(behavior_durations, dtype=np.float64)
        bout_names = np.array(bout_names)

        if len(mean_zscored_dffs) == 0 or len(behavior_durations) == 0:
            print("No valid data points for correlation.")
            return

        # Calculate Pearson correlation
        r, p = stats.pearsonr(mean_zscored_dffs, behavior_durations)

        # Get unique bout types and assign colors
        unique_bouts = np.unique(bout_names)
        color_palette = sns.color_palette("hsv", len(unique_bouts))
        bout_color_map = {bout: color_palette[i] for i, bout in enumerate(unique_bouts)}

        # Step 3: Plotting the scatter plot
        plt.figure(figsize=(12, 6))

        for bout in unique_bouts:
            # Create a mask for each bout
            mask = bout_names == bout
            plt.scatter(mean_zscored_dffs[mask], behavior_durations[mask],
                        color=bout_color_map[bout], label=bout, alpha=0.6)

        # Adding the regression line
        slope, intercept = np.polyfit(mean_zscored_dffs, behavior_durations, 1)
        plt.plot(mean_zscored_dffs, slope * mean_zscored_dffs + intercept, color='black', linestyle='--')

        # Add labels and title
        plt.xlabel(f'Mean Z-scored ΔF/F (0-4s) during {behavior_name.lower()} events')
        plt.ylabel(f'{behavior_name} Duration (s)')
        plt.title(f'Correlation between {behavior_name} Duration and DA Response\n(First {behavior_name} per Mouse)')

        # Display Pearson correlation and p-value
        plt.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.2e}\nn = {len(mean_zscored_dffs)} events',
                transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

        # Add a legend with bout names
        plt.legend(title='Bout', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()


    def plot_first_investigation_vs_dff_4s(self, bouts=None, behavior_name='Investigation', legend_names=None, ylim=None, legend_loc='upper left'):
        """
        Plot the first occurrence of the specified behavior duration vs. mean Z-scored ΔF/F relative to the baseline during 
        a fixed 4-second window for all blocks, color-coded by bout type, with custom legend names and enhanced plot formatting.

        Parameters:
        behavior_name (str): The name of the behavior to analyze (default is 'Investigation').
        bouts (list): A list of bout names to include in the analysis. If None, defaults to ['Short_Term_1', 'Novel_1'].
        legend_names (dict): A dictionary to map bout names to custom legend labels. If None, defaults to standard labels.
        ylim (tuple): A tuple specifying the y-axis limits (min, max). If None, default limits are used.
        legend_loc (str): The location of the legend. Defaults to 'upper left'.
        """
        # Default bouts if none are provided
        if bouts is None:
            bouts = ['Short_Term_1', 'Novel_1']

        # Default legend names if none are provided
        if legend_names is None:
            legend_names = {'Short_Term_1': 'Acq - ST', 'Novel_1': 'Novel', 'Short_Term_2': 'Short-term', 'Long_Term_1': 'Long-Term'}

        # Define the custom colors
        bout_colors = {'Short_Term_1': '#00B7D7', 'Novel_1': '#E06928',
                    'Short_Term_2': '#0045A6', 'Long_Term_1': '#A839A4'}

        mean_zscored_dffs = []
        behavior_durations = []
        bout_names = []

        # Step 1: Compute the PETH for the first occurrence of 'Investigation'
        self.compute_nth_bout_peth_all_blocks(
            behavior_name=behavior_name,
            nth_occurrence=1,
            bouts=bouts,
            pre_time=4,
            post_time=4
        )

        # Step 2: Extract mean Z-scored ΔF/F relative to baseline for a fixed 4-second window
        for block_name, bout_data in self.peri_event_data_all_blocks.items():
            for bout, peri_event_data in bout_data.items():
                if bout not in bouts:
                    continue  # Skip bouts not selected by the user

                time_axis = peri_event_data['time_axis']
                zscore = peri_event_data['zscore']

                # Get the duration of the first investigation event for this bout
                investigation_duration = None
                if bout in self.blocks[block_name].bout_dict and 'Investigation' in self.blocks[block_name].bout_dict[bout]:
                    first_investigation = self.blocks[block_name].bout_dict[bout]['Investigation'][0]
                    investigation_duration = first_investigation['Total Duration']

                    # Set the window to a fixed 4 seconds
                    window_indices = (time_axis >= 0) & (time_axis <= 4)

                    # Calculate the mean z-scored ΔF/F over the fixed 4-second window
                    mean_zscore = np.mean(zscore[window_indices])

                    # Store the results
                    if investigation_duration is not None and mean_zscore is not None:
                        mean_zscored_dffs.append(mean_zscore)
                        behavior_durations.append(investigation_duration)
                        bout_names.append(bout)  # Use bout name as the identifier for color-coding

        # Convert lists to numpy arrays
        mean_zscored_dffs = np.array(mean_zscored_dffs, dtype=np.float64)
        behavior_durations = np.array(behavior_durations, dtype=np.float64)
        bout_names = np.array(bout_names)

        if len(mean_zscored_dffs) == 0 or len(behavior_durations) == 0:
            print("No valid data points for correlation.")
            return

        # Calculate Pearson correlation
        r, p = stats.pearsonr(mean_zscored_dffs, behavior_durations)

        # Step 3: Plotting the scatter plot
        plt.figure(figsize=(16, 9))

        for bout in bouts:
            # Create a mask for each bout
            mask = bout_names == bout
            plt.scatter(mean_zscored_dffs[mask], behavior_durations[mask],
                        color=bout_colors.get(bout, '#000000'),  # Default to black if bout color not found
                        label=legend_names.get(bout, bout), alpha=1, s=800, edgecolor='black', linewidth=6)

        # Adding the regression line with a consistent dashed style
        slope, intercept = np.polyfit(mean_zscored_dffs, behavior_durations, 1)
        plt.plot(mean_zscored_dffs, slope * mean_zscored_dffs + intercept, color='black', linestyle='--', linewidth=4)

        # Add labels and title with larger font sizes
        plt.xlabel(f'Event Induced Z-scored ΔF/F', fontsize=44, labelpad=20)
        plt.ylabel(f'Bout Duration (s)', fontsize=44, labelpad=20)

        # Apply y-axis limits if provided
        if ylim is not None:
            plt.ylim(ylim)

        # Modify x-ticks and y-ticks to be larger
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)

        # Display Pearson correlation and p-value in the legend with the p-value in 3 decimal places
        correlation_text = f'r = {r:.3f}\np = {p:.3f}\nn = {len(mean_zscored_dffs)} events'

        # Create custom legend markers
        custom_lines = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=bout_colors.get(bout, '#000000'), markersize=20, markeredgecolor='black') 
            for bout in bouts
        ]
        # Add an empty Line2D object for the correlation text
        custom_lines.append(plt.Line2D([0], [0], color='none'))

        # Combine the bout labels and the correlation text
        legend_labels = [legend_names.get(bout, bout) for bout in bouts] + [correlation_text]

        # Add a legend with bout names and correlation, placing it to the right of the plot
        plt.legend(custom_lines, legend_labels, title='Agent', loc=legend_loc, fontsize=26, title_fontsize=28)

        # Remove top and right spines and increase the linewidth of the remaining spines
        sns.despine()
        plt.gca().spines['left'].set_linewidth(5)
        plt.gca().spines['bottom'].set_linewidth(5)

        plt.savefig('Scatter.png', transparent=True, bbox_inches='tight', pad_inches=0.1)

        plt.tight_layout()
        plt.show()




    def plot_behavior_durations_boutwise(self, behavior_name='Investigation', min_duration=0):
        """
        Plot the total duration for all events of a specified behavior during each bout.
        The bar graph will show the mean ± SEM across all subjects, with individual subject data points.
        
        Parameters:
        behavior_name (str): The name of the behavior to analyze (e.g., 'Investigation', 'Approach', etc.).
        min_duration (float): The minimum duration of behavior to include in the analysis.
        """
        # Initialize a dictionary to collect the behavior durations for each bout
        bout_behavior_duration_dict = {}

        # Loop through each block in self.blocks to dynamically build bout_behavior_duration_dict
        for block_name, block_data in self.blocks.items():
            if block_data.bout_dict:  # Ensure bout_dict is populated
                for bout, behavior_data in block_data.bout_dict.items():
                    if behavior_name in behavior_data:
                        # Initialize a list for each bout if it doesn't exist
                        if bout not in bout_behavior_duration_dict:
                            bout_behavior_duration_dict[bout] = []
                        # For each bout, collect the duration for all events of the specified behavior
                        for event in behavior_data[behavior_name]:
                            duration = event['Total Duration']
                            if duration > min_duration:  # Only include events longer than min_duration
                                bout_behavior_duration_dict[bout].append(duration)

        # Prepare lists to store the mean and SEM for each bout
        bouts = list(bout_behavior_duration_dict.keys())  # Dynamically get the bout names
        mean_behavior_duration_per_bout = []
        sem_behavior_duration_per_bout = []

        # Calculate the mean and SEM for each bout
        for bout in bouts:
            behavior_duration_values = bout_behavior_duration_dict[bout]
            if behavior_duration_values:  # If there are any values for the bout
                mean_behavior_duration_per_bout.append(np.nanmean(behavior_duration_values))
                sem_behavior_duration_per_bout.append(np.nanstd(behavior_duration_values) / np.sqrt(len(behavior_duration_values)))
            else:
                mean_behavior_duration_per_bout.append(np.nan)  # If no data for this bout, append NaN
                sem_behavior_duration_per_bout.append(np.nan)

        # Create the plot
        fig, ax = plt.subplots(figsize=(16, 6))

        # Plot the bar plot with error bars
        bars = ax.bar(bouts, mean_behavior_duration_per_bout, yerr=sem_behavior_duration_per_bout, capsize=5, color='skyblue', edgecolor='black', label='Mean')

        # Plot each individual's behavior durations for each bout as scatter points
        for i, bout in enumerate(bouts):
            behavior_duration_values = bout_behavior_duration_dict[bout]
            for subject_data in behavior_duration_values:
                ax.scatter(bout, subject_data, color='black', alpha=0.7)  # Plot individual points for each bout

        # Add labels, title, and format
        ax.set_ylabel(f'{behavior_name} Duration (s)', fontsize=12)
        ax.set_xlabel('Bouts', fontsize=12)
        ax.set_title(f'{behavior_name} Duration Across Bouts', fontsize=14)

        # Set x-ticks to match the dynamically captured bout labels
        ax.set_xticks(np.arange(len(bouts)))
        ax.set_xticklabels(bouts, fontsize=12)

        # Add the legend outside the plot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Subjects")

        # Display the plot
        plt.show()


    def plot_behavior_mean_DA_boutwise(self, behavior_name='Investigation', min_duration=0):
        """
        Plot the mean Z-scored ΔF/F (mean DA) for all events of a specified behavior during each bout.
        The bar graph will show the mean ± SEM across all subjects, with individual subject data points.
        
        Parameters:
        behavior_name (str): The name of the behavior to analyze (e.g., 'Investigation', 'Approach', etc.).
        min_duration (float): The minimum duration of behavior events to include in the analysis.
        """
        # Initialize a dictionary to collect the mean DA values for each bout (use block_data.bout_dict keys dynamically)
        bout_mean_DA_dict = {}

        # Loop through each block in self.blocks to dynamically build bout_mean_DA_dict
        for block_name, block_data in self.blocks.items():
            if block_data.bout_dict:  # Ensure bout_dict is populated
                for bout, behavior_data in block_data.bout_dict.items():
                    if behavior_name in behavior_data:
                        # Initialize a list for each bout if it doesn't exist
                        if bout not in bout_mean_DA_dict:
                            bout_mean_DA_dict[bout] = []
                        # For each bout, collect the mean z-score (mean DA) for all events of the specified behavior
                        for event in behavior_data[behavior_name]:
                            duration = event['Total Duration']
                            if duration > min_duration:  # Only include events longer than min_duration
                                bout_mean_DA_dict[bout].append(event['Mean zscore'])

        # Prepare lists to store the mean and SEM for each bout
        bouts = list(bout_mean_DA_dict.keys())  # Dynamically get the bout names
        mean_DA_per_bout = []
        sem_DA_per_bout = []

        # Calculate the mean and SEM for each bout
        for bout in bouts:
            mean_DA_values = bout_mean_DA_dict[bout]
            if mean_DA_values:  # If there are any values for the bout
                mean_DA_per_bout.append(np.nanmean(mean_DA_values))
                sem_DA_per_bout.append(np.nanstd(mean_DA_values) / np.sqrt(len(mean_DA_values)))
            else:
                mean_DA_per_bout.append(np.nan)  # If no data for this bout, append NaN
                sem_DA_per_bout.append(np.nan)

        # Create the plot
        fig, ax = plt.subplots(figsize=(16, 6))

        # Plot the bar plot with error bars
        bars = ax.bar(bouts, mean_DA_per_bout, yerr=sem_DA_per_bout, capsize=5, color='skyblue', edgecolor='black', label='Mean')

        # Plot each individual's mean DA values for each bout as scatter points
        for i, bout in enumerate(bouts):
            mean_DA_values = bout_mean_DA_dict[bout]
            for subject_data in mean_DA_values:
                ax.scatter(bout, subject_data, color='black', alpha=0.7)  # Plot individual points for each bout

        # Add labels, title, and format
        ax.set_ylabel(f'Mean DA (z-score) during {behavior_name}', fontsize=12)
        ax.set_xlabel('Bouts', fontsize=12)
        ax.set_title(f'Mean DA (Z-scored ΔF/F) During {behavior_name} Across Bouts', fontsize=14)

        # Set x-ticks to match the dynamically captured bout labels
        ax.set_xticks(np.arange(len(bouts)))
        ax.set_xticklabels(bouts, fontsize=12)

        # Add the legend outside the plot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Subjects")

        # Display the plot
        plt.show()


    '''********************************** PETHS **********************************'''
    def compute_nth_bout_peth_all_blocks(
            self, 
            behavior_name='Investigation', 
            nth_occurrence=1, 
            bouts=None, 
            pre_time=5, 
            post_time=5, 
            bin_size=0.1
        ):
            """
            Computes the peri-event time histogram (PETH) data for the nth occurrence of a given behavior in each bout using standard z-score.
            Stores the peri-event data (zscore and time axis) for each bout as a class variable.

            Parameters:
            - behavior_name (str): The name of the behavior to generate the PETH for (e.g., 'Aggression').
            - nth_occurrence (int): The occurrence number of the behavior to analyze (1 for first occurrence, 2 for second, etc.).
            - bouts (list): A list of bout names to process. If None, defaults to ['Short_Term_1', 'Short_Term_2', 'Novel_1', 'Long_Term_1'].
            - pre_time (float): The time in seconds to include before the event.
            - post_time (float): The time in seconds to include after the event.
            - bin_size (float): The size of each bin in the histogram (in seconds).

            Returns:
            - None. Stores peri-event data for all blocks and bouts in self.peri_event_data_all_blocks.
            """
            if bouts is None:
                bouts = ['Short_Term_1', 'Short_Term_2', 'Novel_1', 'Long_Term_1']  # Default bouts

            peri_event_data_all_blocks = {}  # Temporary dictionary to store PETH data

            # Initialize to track the minimum number of bins per bout
            min_num_bins_per_bout = {bout: float('inf') for bout in bouts}

            # Iterate through each block in group_data
            for block_name, block_data in self.blocks.items():
                # Initialize dictionary for the current block
                peri_event_data_all_blocks[block_name] = {}

                # Iterate through each specified bout
                for bout in bouts:
                    # Check if the bout and behavior exist in the current block
                    if bout in block_data.bout_dict and behavior_name in block_data.bout_dict[bout]:
                        # Compute the PETH for the nth occurrence
                        peri_event_data = block_data.compute_nth_bout_baseline_peth(
                            bout_name=bout, 
                            behavior_name=behavior_name, 
                            nth_event=nth_occurrence, 
                            pre_time=pre_time, 
                            post_time=post_time, 
                            bin_size=bin_size
                        )

                        if peri_event_data:
                            # Store the PETH data
                            peri_event_data_all_blocks[block_name][bout] = peri_event_data

                            # Update the minimum number of bins for this bout
                            num_bins = len(peri_event_data['time_axis'])
                            if num_bins < min_num_bins_per_bout[bout]:
                                min_num_bins_per_bout[bout] = num_bins
                        else:
                            print(f"No valid peri-event data found for block '{block_name}', bout '{bout}'.")
                    else:
                        print(f"No '{behavior_name}' behavior found in bout '{bout}' for block '{block_name}'.")

            # After processing all blocks and bouts, truncate all PETHs per bout to the minimum number of bins for that bout
            for bout in bouts:
                min_bins = min_num_bins_per_bout[bout]
                if min_bins == float('inf'):
                    # No PETHs were computed for this bout
                    print(f"No PETH data computed for bout '{bout}'. Skipping truncation.")
                    continue
                for block_name, bouts_data in peri_event_data_all_blocks.items():
                    if bout in bouts_data:
                        peth_data = bouts_data[bout]
                        current_bins = len(peth_data['time_axis'])
                        if current_bins > min_bins:
                            # Truncate the time_axis and zscore lists to min_bins
                            peth_data['time_axis'] = peth_data['time_axis'][:min_bins]
                            peth_data['zscore'] = peth_data['zscore'][:min_bins]
                            print(f"Truncated PETH for block '{block_name}', bout '{bout}' to {min_bins} bins.")
                        elif current_bins < min_bins:
                            # This should not happen if min_bins is correctly tracked
                            print(f"Warning: Block '{block_name}', bout '{bout}' has fewer bins ({current_bins}) than min_bins ({min_bins}).")
                            # Optionally, handle this case (e.g., pad with NaNs)
            # Store the computed PETH data in the class attribute
            self.peri_event_data_all_blocks = peri_event_data_all_blocks

    def plot_1st_event_peth_all_traces(self, signal_type='zscore'):
        """
        Plots the peri-event time histogram (PETH) based on the previously computed data for all blocks.

        Parameters:
        signal_type (str): The type of signal to plot. Options are 'zscore' or 'dFF'.

        Returns:
        None. Displays the PETH plot for all blocks on the same graph.
        """
        # Ensure that peri-event data for all blocks is already computed
        if not hasattr(self, 'peri_event_data_all_blocks'):
            print("No peri-event data found. Please compute PETH first using compute_first_event_peth_all_blocks.")
            return

        plt.figure(figsize=(12, 6))
        ylabel = 'Z-scored ΔF/F'
        # Loop through each block and plot its peri-event data
        for block_name, peri_event_data in self.peri_event_data_all_blocks.items():
            time_axis = peri_event_data['time_axis']
            
            if signal_type == 'zscore':
                plt.plot(time_axis, peri_event_data['zscore'], label=f'{block_name} Z-score')
                ylabel = 'Z-scored ΔF/F'  # Set ylabel for zscore
            elif signal_type == 'dFF':
                plt.plot(time_axis, peri_event_data['dFF'], label=f'{block_name} ΔF/F')
                ylabel = r'$\Delta$F/F'  # Set ylabel for dFF
            else:
                print("Invalid signal_type. Use 'zscore' or 'dFF'.")
                return

        plt.axvline(0, color='black', linestyle='--', label='Event onset')
        plt.xlabel('Time (s)')
        plt.ylabel(ylabel)  
        plt.title(f'Peri-Event Time Histogram (PETH) for First Event Across All Blocks ({signal_type})')
        plt.legend()
        plt.tight_layout()
        plt.show()


    def compute_first_bout_peth_all_blocks_standard(self, behavior_name='Investigation', bouts=None, pre_time=5, post_time=5, bin_size=0.1):
        """
        Computes the peri-event time histogram (PETH) data for the first occurrence of a given event in each bout.
        This version uses standard z-scoring based on the whole trace, using precomputed `self.zscore` data.

        Parameters:
        behavior_name (str): The name of the event to generate the PETH for (e.g., 'Investigation').
        bouts (list): A list of bout names to process.
        pre_time (float): The time in seconds to include before the event.
        post_time (float): The time in seconds to include after the event.
        bin_size (float): The size of each bin in the histogram (in seconds).

        Returns:
        None. Stores peri-event data for all blocks and bouts as a class variable.
        """
        if bouts is None:
            bouts = ['Short_Term_1', 'Short_Term_2', 'Novel_1', 'Long_Term_1']  # Default to these bouts if none provided

        self.peri_event_data_all_blocks = {}  # Initialize a dictionary to store PETH data for each bout

        # Track the shortest time axis across all blocks and bouts
        min_time_length = float('inf')

        # Loop through each block in self.blocks
        for block_name, block_data in self.blocks.items():
            self.peri_event_data_all_blocks[block_name] = {}  # Initialize PETH storage for each block

            # Loop through each bout in the specified bouts
            for bout in bouts:
                if bout in block_data.bout_dict and behavior_name in block_data.bout_dict[bout]:
                    # Extract the first occurrence of the behavior
                    behavior_events = block_data.bout_dict[bout][behavior_name]
                    if len(behavior_events) == 0:
                        print(f"No occurrences of {behavior_name} found in {bout} for {block_name}.")
                        continue

                    first_event = behavior_events[0]
                    event_time = first_event['Start Time']

                    # Define the peri-event window
                    start_time = event_time - pre_time
                    end_time = event_time + post_time

                    start_idx = np.searchsorted(block_data.timestamps, start_time)
                    end_idx = np.searchsorted(block_data.timestamps, end_time)

                    # Extract z-score data from the entire trace for this peri-event window
                    peri_event_zscore = block_data.zscore[start_idx:end_idx]

                    # Generate the time axis for the peri-event window
                    time_axis = np.linspace(-pre_time, post_time, len(peri_event_zscore))

                    # Store both peri-event zscore and time axis in the class variable dictionary
                    self.peri_event_data_all_blocks[block_name][bout] = {
                        'zscore': peri_event_zscore,
                        'time_axis': time_axis
                    }

                    # Check the time axis length to find the shortest one
                    if len(time_axis) < min_time_length:
                        min_time_length = len(time_axis)
                else:
                    print(f"No {behavior_name} found in {bout} for {block_name}.")

        # Truncate all traces to the shortest time axis length to ensure consistency
        for block_name, bout_data in self.peri_event_data_all_blocks.items():
            for bout, peri_event_data in bout_data.items():
                peri_event_data['zscore'] = peri_event_data['zscore'][:min_time_length]
                peri_event_data['time_axis'] = peri_event_data['time_axis'][:min_time_length]


    def plot_peth_for_single_bout(self, 
                                signal_type='zscore', 
                                error_type='sem', 
                                bout=None, 
                                title='PETH for First Investigation', 
                                color='#00B7D7', 
                                display_pre_time=3, 
                                display_post_time=3, 
                                yticks_interval=2, 
                                figsize=(14, 8),
                                ax=None):
        """
        Plots the mean and SEM/Std of the peri-event time histogram (PETH) for a single bout with larger font sizes.
        
        Parameters:
        - signal_type (str): The type of signal to plot. Options are 'zscore' or 'dFF'.
        - error_type (str): The type of error to plot. Options are 'sem' for Standard Error of the Mean or 'std' for Standard Deviation.
        - bout (str): The bout name to plot. If None, defaults to the first bout in the list ['Short_Term_1', 'Short_Term_2', 'Novel_1', 'Long_Term_1'].
        - title (str): Title for the entire figure.
        - color (str): Color for both the trace line and the error area (default is cyan '#00B7D7').
        - display_pre_time (float): Time before the event onset to display on the x-axis (in seconds).
        - display_post_time (float): Time after the event onset to display on the x-axis (in seconds).
        - yticks_interval (float): Interval between y-ticks on the plot.
        - figsize (tuple): Size of the figure in inches (width, height).
        - ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, a new figure and axis are created.
        
        Returns:
        - None. Displays the mean PETH plot for the specified bout with SEM/Std shaded area.
        """
        
        # Define default bouts if none provided
        default_bouts = ['Short_Term_1', 'Short_Term_2', 'Novel_1', 'Long_Term_1']
        if bout is None:
            if default_bouts:
                bout = default_bouts[0]
                print(f"No bout specified. Defaulting to '{bout}'.")
            else:
                raise ValueError("No bouts available to plot. Please provide a bout name.")
        
        # Validate that 'bout' is a single string
        if not isinstance(bout, str):
            raise ValueError("Parameter 'bout' must be a single bout name as a string.")
        
        # Initialize lists to store traces and time axes
        all_traces = []       # To store all signal traces for the specified bout
        all_time_axes = []    # To store all time axes
        
        # Collect peri-event data for the specified bout across all blocks
        for block_name, peth_data_block in self.peri_event_data_all_blocks.items():
            if bout in peth_data_block:
                peri_event_data = peth_data_block[bout]
                signal_data = peri_event_data.get(signal_type)
                current_time_axis = peri_event_data.get('time_axis')
                
                if signal_data is None or current_time_axis is None:
                    print(f"Missing '{signal_type}' or 'time_axis' in bout '{bout}' for block '{block_name}'. Skipping.")
                    continue
                
                all_traces.append(signal_data)
                all_time_axes.append(current_time_axis)
                print(f"Collected trace from block '{block_name}' with {len(signal_data)} data points.")
        
        if not all_traces:
            print(f"No valid traces found for bout '{bout}'.")
            return
        
        # Determine the overlapping time range across all blocks
        # Find the maximum start time and minimum end time
        start_times = [ta[0] for ta in all_time_axes]
        end_times = [ta[-1] for ta in all_time_axes]
        common_start = max(start_times)
        common_end = min(end_times)
        
        print(f"Common overlapping time range: {common_start} to {common_end} seconds.")
        
        if common_end <= common_start:
            print("No overlapping time range found across blocks.")
            return
        
        # Define a common time axis based on the smallest overlapping range
        # Assuming all time axes are uniformly sampled
        # To handle slight differences, we'll define a new common time axis
        # based on the first block's time axis
        reference_time_axis = all_time_axes[0]
        # Find indices within the common range for the reference
        ref_start_idx = np.searchsorted(reference_time_axis, common_start, side='left')
        ref_end_idx = np.searchsorted(reference_time_axis, common_end, side='right')
        common_time_axis = reference_time_axis[ref_start_idx:ref_end_idx]
        print(f"Reference time axis truncated to indices {ref_start_idx} to {ref_end_idx} ({len(common_time_axis)} points).")
        
        # Initialize list for interpolated traces
        interpolated_traces = []
        
        for idx, (trace, ta) in enumerate(zip(all_traces, all_time_axes)):
            # Define the interpolation function
            interp_func = interp1d(ta, trace, kind='linear', bounds_error=False, fill_value='extrapolate')
            # Interpolate the trace onto the common_time_axis
            interpolated_trace = interp_func(common_time_axis)
            interpolated_traces.append(interpolated_trace)
            print(f"Interpolated trace {idx+1} to common time axis with {len(interpolated_trace)} points.")
        
        # Convert interpolated_traces to a NumPy array
        all_traces = np.array(interpolated_traces)
        print(f"All traces stacked into array with shape {all_traces.shape}.")
        
        # Calculate mean and error metrics
        mean_trace = np.mean(all_traces, axis=0)
        
        if error_type.lower() == 'sem':
            error_trace = np.std(all_traces, axis=0) / np.sqrt(len(all_traces))
            error_label = 'SEM'
        elif error_type.lower() == 'std':
            error_trace = np.std(all_traces, axis=0)
            error_label = 'Std'
        else:
            raise ValueError("Invalid 'error_type'. Choose either 'sem' or 'std'.")
        
        # Define the display window
        display_start = -display_pre_time
        display_end = display_post_time
        display_start_idx = np.searchsorted(common_time_axis, display_start, side='left')
        display_end_idx = np.searchsorted(common_time_axis, display_end, side='right')
        
        # Handle cases where display window exceeds common_time_axis
        display_start_idx = max(display_start_idx, 0)
        display_end_idx = min(display_end_idx, len(common_time_axis))
        
        # Truncate data to the display window
        mean_trace = mean_trace[display_start_idx:display_end_idx]
        error_trace = error_trace[display_start_idx:display_end_idx]
        display_time = common_time_axis[display_start_idx:display_end_idx]
        
        print(f"Display window: {display_start} to {display_end} seconds.")
        print(f"Displaying {len(display_time)} data points.")
        
        # Create the plot or use the provided ax
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = None  # Only needed if you need to save or further manipulate the figure
        
        ax.plot(display_time, mean_trace, color=color, label=f'Mean {signal_type.capitalize()}', linewidth=3.5)
        ax.fill_between(display_time, mean_trace - error_trace, mean_trace + error_trace, color=color, alpha=0.4, label=error_label)
        ax.axvline(0, color='black', linestyle='--', label='Event Onset', linewidth=5)
        
        # Customize x-axis
        ax.set_xticks([display_time[0], 0, display_time[-1]])
        ax.set_xticklabels([f'{display_time[0]:.1f}', '0', f'{display_time[-1]:.1f}'], fontsize=24)
        ax.set_xlabel('Time from Onset (s)', fontsize=40)
        
        # Customize y-axis
        y_min, y_max = ax.get_ylim()
        y_ticks = np.arange(np.floor(y_min / yticks_interval) * yticks_interval, 
                            np.ceil(y_max / yticks_interval) * yticks_interval + yticks_interval, 
                            yticks_interval)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{y:.0f}' for y in y_ticks], fontsize=40)
        ax.set_ylabel(f'Standard Z-scored ΔF/F', fontsize=40)
        
        # Remove top and right spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Customize spines' linewidth
        ax.spines['left'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        
        # Customize tick parameters
        ax.tick_params(axis='both', which='major', labelsize=40, width=2)

        plt.savefig('standard.png', transparent=True, bbox_inches='tight', pad_inches=0.1)

        # Add legend with increased fontsize
        ax.legend(fontsize=30)
        
        
        if fig is not None:
            fig.tight_layout()
            plt.show()
        else:
            return ax




    def plot_peth_for_bouts(self, signal_type='zscore', error_type='sem', bouts=None, title='PETH for First Investigation Across Agents', 
                        color='#00B7D7', display_pre_time=3, display_post_time=3, yticks_interval=2):
        """
        Plots the mean and SEM/Std of the peri-event time histogram (PETH) for the first event across all bouts.
        
        Parameters:
        signal_type (str): The type of signal to plot. Options are 'zscore' or 'dFF'.
        error_type (str): The type of error to plot. Options are 'sem' for Standard Error of the Mean or 'std' for Standard Deviation.
        bouts (list or str): A list of bout names to plot or a single bout name. If None, uses default ['Short_Term_1', 'Short_Term_2', 'Novel_1', 'Long_Term_1'].
        title (str): Title for the entire figure.
        color (str): Color for both the trace line and the error area (default is cyan '#00B7D7').
        display_pre_time (float): How much time to show before the event on the x-axis (default is 3 seconds).
        display_post_time (float): How much time to show after the event on the x-axis (default is 3 seconds).
        yticks_interval (float): Interval for the y-ticks on the plots (default is 2).

        Returns:
        None. Displays the mean PETH plot for each bout with SEM/Std shaded area.
        """

        # Handle single bout by converting it to a list if it's not already one
        if isinstance(bouts, str):
            bouts = [bouts]
        
        # Default to these bouts if none provided
        if bouts is None:
            bouts = ['Short_Term_1', 'Short_Term_2', 'Novel_1', 'Long_Term_1']
        
        # Adjust the figure size to be less wide and more tall, with more space between traces
        fig, axes = plt.subplots(1, len(bouts), figsize=(5 * len(bouts), 10), sharey=True)  # Adjusted dimensions
        
        # If there's only one bout, make axes a list to keep the logic consistent
        if len(bouts) == 1:
            axes = [axes]
        
        for i, bout in enumerate(bouts):
            ax = axes[i]  # Get the subplot for this bout

            all_traces = []  # To store all traces across blocks for this bout
            time_axis = []  # Will be set later when available

            # Collect the peri-event data for the bout across all blocks
            for block_name, peth_data_block in self.peri_event_data_all_blocks.items():
                if bout in peth_data_block:
                    # Extract the relevant PETH data (zscore or dFF)
                    peri_event_data = peth_data_block[bout]
                    signal_data = peri_event_data[signal_type]
                    time_axis = peri_event_data['time_axis']

                    # Store the signal data for averaging later
                    all_traces.append(signal_data)

            # Convert list of traces to numpy array for easier manipulation
            all_traces = np.array(all_traces)

            # Find the minimum trace length and truncate all traces to the same length
            min_length = min([len(trace) for trace in all_traces])
            all_traces = np.array([trace[:min_length] for trace in all_traces])
            time_axis = time_axis[:min_length]  # Truncate time axis as well

            # Calculate the mean across all traces
            mean_trace = np.mean(all_traces, axis=0)

            # Calculate SEM or Std across all traces depending on the selected error type
            if error_type == 'sem':
                error_trace = np.std(all_traces, axis=0) / np.sqrt(len(all_traces))  # SEM
                error_label = 'SEM'
            elif error_type == 'std':
                error_trace = np.std(all_traces, axis=0)  # Standard Deviation
                error_label = 'Std'

            # Determine the indices for the display range
            display_start_idx = np.searchsorted(time_axis, -display_pre_time)
            display_end_idx = np.searchsorted(time_axis, display_post_time)

            # Truncate the mean trace, error trace, and time axis to the desired display range
            mean_trace = mean_trace[display_start_idx:display_end_idx]
            error_trace = error_trace[display_start_idx:display_end_idx]
            time_axis = time_axis[display_start_idx:display_end_idx]

            # Plot the mean trace with SEM/Std shaded area, with customizable color
            ax.plot(time_axis, mean_trace, color=color, label=f'Mean {signal_type.capitalize()}', linewidth=1.5)  # Trace color
            ax.fill_between(time_axis, mean_trace - error_trace, mean_trace + error_trace, color=color, alpha=0.3, label=error_label)  # Error color

            # Plot event onset line
            ax.axvline(0, color='black', linestyle='--', label='Event onset')

            # Set the x-ticks to show only the last time, 0, and the very end time
            ax.set_xticks([time_axis[0], 0, time_axis[-1]])
            ax.set_xticklabels([f'{time_axis[0]:.1f}', '0', f'{time_axis[-1]:.1f}'], fontsize= 40)

            # Set the y-tick labels with larger font size and specified interval
            y_min, y_max = ax.get_ylim()
            y_ticks = np.arange(np.floor(y_min / yticks_interval) * yticks_interval, 
                                np.ceil(y_max / yticks_interval) * yticks_interval + yticks_interval, 
                                yticks_interval)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f'{y:.0f}' for y in y_ticks], fontsize=40)  # Significantly larger y-tick labels

            # Set the title for each bout
            # ax.set_title(f'{bout.replace("_", " ")}', fontsize=30)

            ax.set_xlabel('Onset (s)', fontsize=40)

            # Remove the right and top spines for each subplot
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        # Set the y-label with larger font size
        axes[0].set_ylabel(f'{signal_type.capitalize()} dFF', fontsize=40)  # Set shared y-label for all subplots

        plt.suptitle(title, fontsize=30)
        plt.tight_layout()
        plt.show()
