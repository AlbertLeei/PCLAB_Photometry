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
    from social_pref.social_pref_extension import sp_processing, sp_compute_first_bout_peth_all_blocks


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

            if os.path.exists(csv_file_path):
                print(f"Processing {block_folder}...")
                if remove_led_artifact:
                    tdt_data_obj.remove_initial_LED_artifact(t=t)
                    # tdt_data_obj.remove_final_data_segment(t = 10)
                
                tdt_data_obj.smooth_and_apply(window_len=int(tdt_data_obj.fs)*2)
                tdt_data_obj.apply_ma_baseline_correction()
                tdt_data_obj.align_channels()
                tdt_data_obj.compute_dFF()
                tdt_data_obj.compute_zscore()

                tdt_data_obj.extract_manual_annotation_behaviors(csv_file_path)
                tdt_data_obj.combine_consecutive_behaviors(behavior_name='all', bout_time_threshold=1, min_occurrences=1)
                tdt_data_obj.remove_short_behaviors(behavior_name='all', min_duration=0.2)

                tdt_data_obj.verify_signal()

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


    def plot_1st_behavior_vs_dff_all(self, behavior_name='Investigation', min_duration=0.0, max_duration=np.inf):
        """
        Plot the specified behavior duration vs. mean Z-scored ΔF/F during the first occurrence of that behavior
        for all blocks, color-coded by individual subject identity. Only includes the first behavior event that is longer 
        than min_duration and shorter than max_duration seconds.
        
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
            if block_data.bout_dict:
                for bout, behavior_data in block_data.bout_dict.items():
                    # Check if the behavior exists in the bout
                    if behavior_name in behavior_data:
                        # Look through all occurrences of the behavior and find the first one that fits the criteria
                        valid_event_found = False
                        for event in behavior_data[behavior_name]:
                            duration = event.get('Total Duration', None)
                            mean_zscore = event.get('Mean zscore', None)

                            # Check if the duration meets the specified criteria
                            if duration is not None and min_duration < duration < max_duration:
                                behavior_durations.append(duration)
                                mean_zscored_dffs.append(mean_zscore)
                                subject_names.append(block_data.subject_name)  # Block name as the subject identifier
                                valid_event_found = True  # Mark that a valid event has been found
                                break  # Exit the loop once the first valid event is found
                        
                        # If no valid event was found, continue to the next bout
                        if not valid_event_found:
                            continue

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
        plt.xlabel(f'Mean Z-scored ΔF/F during 1st valid {behavior_name.lower()} event')
        plt.ylabel(f'{behavior_name} duration (s)')
        plt.title(f'Correlation between 1st {behavior_name} Duration and DA Response (All Blocks > {min_duration}s and < {max_duration}s)')
        
        # Display Pearson correlation and p-value
        plt.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.2e}\nn = {len(mean_zscored_dffs)} sessions',
                transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

        # Add a legend with subject names
        plt.legend(title='Subject', bbox_to_anchor=(1.05, 1), loc='upper left')

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
    def compute_first_bout_peth_all_blocks(self, behavior_name='Investigation', bouts=None, pre_time=5, post_time=5, bin_size=0.1):
        """
        Computes the peri-event time histogram (PETH) data for the first occurrence of a given event in each bout.
        Uses the TDTData class's `compute_1st_bout_peth` function and stores the peri-event data (zscore, dFF, and time axis) 
        for each bout as a class variable.

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
                    # Use the `compute_1st_bout_peth` method to compute the PETH for the first event in the bout
                    block_data.compute_1st_bout_peth(bout_name=bout, behavior_name=behavior_name, pre_time=pre_time, post_time=post_time, bin_size=bin_size)

                    # Extract and store the peri-event data (assumed to be stored in block_data.peri_event_data)
                    if hasattr(block_data, 'peri_event_data'):
                        self.peri_event_data_all_blocks[block_name][bout] = block_data.peri_event_data

                        # Check the time axis length to find the shortest one
                        time_axis = block_data.peri_event_data['time_axis']
                        if len(time_axis) < min_time_length:
                            min_time_length = len(time_axis)
                    else:
                        print(f"No peri-event data found for {block_name}, {bout}.")
                else:
                    print(f"No {behavior_name} found in {bout} for {block_name}.")

        # Truncate all traces to the shortest time axis length to ensure consistency
        for block_name, bout_data in self.peri_event_data_all_blocks.items():
            for bout, peri_event_data in bout_data.items():
                for key in ['zscore', 'dFF', 'time_axis']:  # Truncate zscore, dFF, and time_axis
                    peri_event_data[key] = peri_event_data[key][:min_time_length]


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



    def plot_mean_peth(self, signal_type='zscore', error_type='sem', title= 'NA'):
        """
        Plots the mean and either SEM or Std of the peri-event time histogram (PETH) for the first event across all blocks.

        Parameters:
        signal_type (str): The type of signal to plot. Options are 'zscore' or 'dFF'.
        error_type (str): The type of error to plot. Options are 'sem' for Standard Error of the Mean or 'std' for Standard Deviation.

        Returns:
        None. Displays the mean PETH plot with SEM or Std shaded area.
        """
        # Ensure that peri-event data for all blocks is already computed
        if not hasattr(self, 'peri_event_data_all_blocks'):
            print("No peri-event data found. Please compute PETH first using compute_first_event_peth_all_blocks.")
            return

        # Collect all the peri-event traces
        all_traces = []
        time_axis = []

        # Loop through each block and collect the peri-event data
        for block_name, peri_event_data in self.peri_event_data_all_blocks.items():
            if signal_type == 'zscore':
                all_traces.append(peri_event_data['zscore'])
                ylabel = 'Z-scored ΔF/F'
            elif signal_type == 'dFF':
                all_traces.append(peri_event_data['dFF'])
                ylabel = r'$\Delta$F/F'
            else:
                print("Invalid signal_type. Use 'zscore' or 'dFF'.")
                return

            time_axis = peri_event_data['time_axis']

        # Find the minimum trace length to truncate all traces to the same length
        min_length = min([len(trace) for trace in all_traces])

        # Truncate all traces to the same length (the shortest one)
        all_traces = np.array([trace[:min_length] for trace in all_traces])

        # Also truncate the time axis to match the trace length
        time_axis = time_axis[:min_length]

        # Calculate the mean across all traces
        mean_trace = np.mean(all_traces, axis=0)

        # Calculate SEM or Std across all traces depending on the selected error type
        if error_type == 'sem':
            error_trace = np.std(all_traces, axis=0) / np.sqrt(len(all_traces))  # SEM
            error_label = 'SEM'
        elif error_type == 'std':
            error_trace = np.std(all_traces, axis=0)  # Standard Deviation
            error_label = 'Std'
        else:
            print("Invalid error_type. Use 'sem' or 'std'.")
            return

        # Plot the mean trace with SEM or Std shading
        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, mean_trace, color='blue', label=f'Mean {signal_type}')
        plt.fill_between(time_axis, mean_trace - error_trace, mean_trace + error_trace, color='blue', alpha=0.3, label=error_label)

        # Plot event onset line
        plt.axvline(0, color='black', linestyle='--', label='Event onset')
        
        # Add labels and titles
        plt.xlabel('Time (s)')
        plt.ylabel(ylabel)
        plt.title(title)
        
        # Add legend and display plot
        plt.legend()
        plt.tight_layout()
        plt.show()



    def plot_peth_for_bouts(self, signal_type='zscore', error_type='sem', bouts=None, title='PETH for First Investigation Across Agents', color='#00B7D7', custom_xtick_labels=None):
        """
        Plots the mean and SEM/Std of the peri-event time histogram (PETH) for the first event across all bouts.

        Parameters:
        signal_type (str): The type of signal to plot. Options are 'zscore' or 'dFF'.
        error_type (str): The type of error to plot. Options are 'sem' for Standard Error of the Mean or 'std' for Standard Deviation.
        bouts (list): A list of bout names to plot. If None, uses default ['Short_Term_1', 'Short_Term_2', 'Novel_1', 'Long_Term_1'].
        title (str): Title for the entire figure.
        color (str): Color for both the trace line and the error area (default is cyan '#00B7D7').
        custom_xtick_labels (list): Custom labels for the x-ticks across all subplots. If None, defaults to bout names.

        Returns:
        None. Displays the mean PETH plot for each bout with SEM/Std shaded area.
        """
        if bouts is None:
            bouts = ['Short_Term_1', 'Short_Term_2', 'Novel_1', 'Long_Term_1']

        fig, axes = plt.subplots(1, len(bouts), figsize=(15, 5), sharey=True)  # Adjusted width of the figure

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

            # Plot the mean trace with SEM/Std shaded area, with customizable color
            ax.plot(time_axis, mean_trace, color=color, label=f'Mean {signal_type.capitalize()}', linewidth=1.5)  # Trace color
            ax.fill_between(time_axis, mean_trace - error_trace, mean_trace + error_trace, color=color, alpha=0.3, label=error_label)  # Error color

            # Plot event onset line
            ax.axvline(0, color='black', linestyle='--', label='Event onset')

            # Set the title for each bout
            if custom_xtick_labels is not None and i < len(custom_xtick_labels):
                ax.set_title(custom_xtick_labels[i], fontsize=14)
            else:
                ax.set_title(bout.replace('_', ' '), fontsize=14)

            ax.set_xlabel('Time (s)', fontsize=12)

            # Remove the right and top spines for each subplot
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        axes[0].set_ylabel(f'{signal_type.capitalize()} dFF', fontsize=12)  # Set shared y-label for all subplots
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

