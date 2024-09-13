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
    from home_cage.home_cage_extension import hc_processing


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

    def batch_process(self, remove_led_artifact=True, t=20):
        """
        Batch processes the TDT data by extracting behaviors, removing LED artifacts, and computing z-score.
        """
        for block_folder, tdt_data_obj in self.blocks.items():
            csv_file_name = f"{block_folder}.csv"
            csv_file_path = os.path.join(self.csv_base_path, csv_file_name)
            if os.path.exists(csv_file_path):
                print(f"Processing {block_folder}...")
                if remove_led_artifact:
                    tdt_data_obj.remove_initial_LED_artifact(t=t)
                tdt_data_obj.smooth_signal()

                tdt_data_obj.extract_manual_annotation_behaviors(csv_file_path)
                tdt_data_obj.combine_consecutive_behaviors(behavior_name='all', bout_time_threshold=1, min_occurrences=1)
                tdt_data_obj.remove_short_behaviors(behavior_name='all', min_duration=0.1)

                # tdt_data_obj.downsample_data(N=16)
                tdt_data_obj.verify_signal()
                tdt_data_obj.compute_dff()
                tdt_data_obj.compute_zscore()

#Short behavior
        

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
