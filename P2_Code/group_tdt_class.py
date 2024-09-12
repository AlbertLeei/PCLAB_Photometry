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
                tdt_data_obj.extract_manual_annotation_behaviors(csv_file_path)
                tdt_data_obj.combine_consecutive_behaviors(behavior_name='all', bout_time_threshold=2, min_occurrences=1)
                if remove_led_artifact:
                    tdt_data_obj.remove_initial_LED_artifact(t=t)
                # tdt_data_obj.smooth_signal()
                # tdt_data_obj.downsample_data(N=16)
                tdt_data_obj.verify_signal()
                tdt_data_obj.compute_dff()
                tdt_data_obj.compute_zscore()

    def compute_group_psth(self, behavior_name='Pinch', pre_time=5, post_time=5, signal_type='zscore'):
        """
        Computes the group-level PSTH for the specified behavior and stores it in self.group_psth.
        """
        psths = []
        for block_folder, tdt_data_obj in self.blocks.items():
            # Compute individual PSTH
            psth_df = tdt_data_obj.compute_psth(behavior_name, pre_time=pre_time, post_time=post_time, signal_type=signal_type)
            psths.append(psth_df.mean(axis=0).values)

        # Ensure all PSTHs have the same length by trimming or padding
        min_length = min(len(psth) for psth in psths)
        trimmed_psths = [psth[:min_length] for psth in psths]

        # Convert list of arrays to a DataFrame
        self.group_psth = pd.DataFrame(trimmed_psths).mean(axis=0)
        self.group_psth.index = psth_df.columns[:min_length]

    def plot_group_psth(self, behavior_name='Pinch', pre_time=5, post_time=5, signal_type='zscore'):
        """
        Plots the group-level PSTH for the specified behavior, including the variance (standard deviation) between trials.
        """
        if self.group_psth is None:
            raise ValueError("Group PSTH has not been computed. Call compute_group_psth first.")

        psths_mean = []
        psths_std = []
        for block_folder, tdt_data_obj in self.blocks.items():
            # Compute individual PSTH
            psth_df = tdt_data_obj.compute_psth(behavior_name, pre_time=pre_time, post_time=post_time, signal_type=signal_type)
            psths_mean.append(psth_df['mean'].values)
            psths_std.append(psth_df['std'].values)

        # Convert list of arrays to DataFrames
        psth_mean_df = pd.DataFrame(psths_mean)
        psth_std_df = pd.DataFrame(psths_std)

        # Calculate the mean and standard deviation across blocks
        group_psth_mean = psth_mean_df.mean(axis=0)
        group_psth_std = psth_std_df.mean(axis=0)

        # Ensure the index matches the time points
        time_points = psth_df.index

        plt.figure(figsize=(10, 6))
        plt.plot(time_points, group_psth_mean, label=f'Group {signal_type} Mean')
        plt.fill_between(time_points, group_psth_mean - group_psth_std, group_psth_mean + group_psth_std, 
                        color='gray', alpha=0.3, label='_nolegend_')  # Exclude from legend
        plt.xlabel('Time (s)')
        plt.ylabel(f'{signal_type}')
        plt.title(f'Group PSTH for {behavior_name}')
        plt.axvline(0, color='r', linestyle='--', label=f'{behavior_name} Onset')
        # Set x-ticks at each second
        plt.xticks(np.arange(int(time_points.min()), int(time_points.max())+1, 1))  # Ticks every second
        plt.legend()
        plt.show()

def plot_all_individual_psth(self, behavior_name='Pinch', pre_time=5, post_time=5, signal_type='zscore'):
    """
    Plots the individual PSTHs for each block for the specified behavior.

    Parameters:
    behavior_name (str): The name of the behavior to plot.
    pre_time (float): Time in seconds before the behavior event onset to include in the PSTH.
    post_time (float): Time in seconds after the behavior event onset to include in the PSTH.
    signal_type (str): The type of signal to use for PSTH computation. Options are 'zscore' or 'dFF'.
    """
    plt.figure(figsize=(10, 6))

    # Iterate through each block and plot its PSTH
    for block_folder, tdt_data_obj in self.blocks.items():
        # Compute individual PSTH
        psth_df = tdt_data_obj.compute_psth(behavior_name, pre_time=pre_time, post_time=post_time, signal_type=signal_type)
        
        # Plot the individual trace
        time_points = psth_df.index
        plt.plot(time_points, psth_df['mean'].values, label=f'{block_folder}', alpha=0.6)

    plt.xlabel('Time (s)')
    plt.ylabel(f'{signal_type}')
    plt.title(f'Individual PSTHs for {behavior_name}')
    plt.axvline(0, color='r', linestyle='--', label=f'{behavior_name} Onset')

    # Set x-ticks at each second
    plt.xticks(np.arange(int(time_points.min()), int(time_points.max())+1, 1))  # Ticks every second
    
    # Add a legend outside the plot showing each block trace
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()



    '''********************************** PLOTTING **********************************'''
    def plot_individual_psths(self, behavior_name='Pinch', pre_time=5, post_time=5, signal_type='dFF'):
        """
        Plots the PSTH for each block individually.
        """
        rows = len(self.blocks)
        figsize = (18, 5 * rows)

        fig, axs = plt.subplots(rows, 1, figsize=figsize)
        axs = axs.flatten()
        plt.rcParams.update({'font.size': 16})

        for i, (block_folder, tdt_data_obj) in enumerate(self.blocks.items()):
            psth_df = tdt_data_obj.compute_psth(behavior_name, pre_time=pre_time, post_time=post_time, signal_type=signal_type)

            # Extract the time axis (which should be the columns of the DataFrame)
            time_axis = psth_df.index
            
            # Plot the mean PSTH with standard deviation shading
            psth_mean = psth_df['mean'].values
            psth_std = psth_df['std'].values
            
            axs[i].plot(time_axis, psth_mean, label=f'{tdt_data_obj.subject_name}', color='blue')
            axs[i].fill_between(time_axis, psth_mean - psth_std, psth_mean + psth_std, color='blue', alpha=0.3)
            
            axs[i].set_title(f'{tdt_data_obj.subject_name}: {signal_type.capitalize()} Signal with {behavior_name} Bouts')
            axs[i].set_ylabel(f'{signal_type}')
            axs[i].set_xlabel('Time (s)')
            axs[i].axvline(0, color='r', linestyle='--', label=f'{behavior_name} Onset')

            # Set x-ticks at each second for this subplot
            axs[i].set_xticks(np.arange(int(time_axis.min()), int(time_axis.max()) + 1, 1))
            axs[i].legend()

        plt.tight_layout()
        plt.show()


    '''********************************** BEHAVIORS **********************************'''
    def plot_all_behavior_vs_dff_all(self, behavior_name='Investigation', min_duration=0):
        """
        Plot the specified behavior duration vs. mean Z-scored ΔF/F during all occurrences of that behavior for all blocks,
        color-coded by individual subject identity. Only includes behavior events longer than min_duration seconds.
        
        Parameters:
        behavior_name (str): The name of the behavior to analyze (e.g., 'Investigation', 'Approach', etc.).
        min_duration (float): The minimum duration of behavior to include in the plot.
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
                            if duration > min_duration:  # Only include behavior events longer than min_duration
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
        plt.title(f'Correlation between {behavior_name} Duration and DA Response (All {behavior_name}s > {min_duration}s)')

        # Display Pearson correlation and p-value
        plt.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.2e}\nn = {len(mean_zscored_dffs)} sessions',
                transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

        # Add a legend with subject names
        plt.legend(title='Subject', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()


    def plot_1st_behavior_vs_dff_all(self, behavior_name='Investigation', min_duration=0):
        """
        Plot the specified behavior duration vs. mean Z-scored ΔF/F during the first occurrence of that behavior
        for all blocks, color-coded by individual subject identity.
        
        Parameters:
        behavior_name (str): The name of the behavior to analyze (e.g., 'Investigation', 'Approach', etc.).
        min_duration (float): The minimum duration of behavior to include in the plot.
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
                        # Extract the first occurrence of the behavior
                        if behavior_data[behavior_name]:  # Ensure there is at least one event for the behavior
                            first_event = behavior_data[behavior_name][0]  # Get the first event of the behavior
                            
                            # Extract the behavior duration and mean DA for this first event
                            duration = first_event.get('Total Duration', None)
                            mean_zscore = first_event.get('Mean zscore', None)
                            
                            # Ensure the duration is valid (not None) and greater than min_duration
                            if duration is not None and duration > min_duration:
                                behavior_durations.append(duration)
                                mean_zscored_dffs.append(mean_zscore)
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
        plt.xlabel(f'Mean Z-scored ΔF/F during 1st {behavior_name.lower()}')
        plt.ylabel(f'{behavior_name} duration (s)')
        plt.title(f'Correlation between 1st {behavior_name} Duration and DA Response (All Blocks)')
        
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
