import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import scipy.stats as stats


'''********************************** FOR SINGLE OBJECT  **********************************'''
def hab_dishab_plot_behavior_event(self, behavior_name='all', plot_type='dFF', ax=None):
    """
    Plots Delta F/F (dFF) or z-scored signal with behavior events for the habituation-dishabituation experiment.

    Parameters:
    - behavior_name (str): The name of the behavior to plot. Use 'all' to plot all behaviors.
    - plot_type (str): The type of plot. Options are 'dFF' and 'zscore'.
    - ax: An optional matplotlib Axes object. If provided, the plot will be drawn on this Axes.
    """
    # Prepare data based on plot type
    y_data = []
    if plot_type == 'dFF':
        if self.dFF is None:
            self.compute_dff()
        y_data = self.dFF
        y_label = r'$\Delta$F/F'
        y_title = 'Delta F/F Signal'
    elif plot_type == 'zscore':
        if self.zscore is None:
            self.compute_zscore()
        y_data = self.zscore
        y_label = 'z-score'
        y_title = 'Z-scored Signal'
    else:
        raise ValueError("Invalid plot_type. Only 'dFF' and 'zscore' are supported.")

    # Create plot if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 8))

    ax.plot(self.timestamps, np.array(y_data), linewidth=2, color='black', label=plot_type)

    # Define specific colors for behaviors
    behavior_colors = {'Investigation': 'dodgerblue', 'Approach': 'green', 'Defeat': 'red'}

    # Track which labels have been plotted to avoid duplicates
    plotted_labels = set()

    # Plot behavior spans
    if behavior_name == 'all':
        for behavior_event in self.behaviors.keys():
            if behavior_event in behavior_colors:
                behavior_onsets = self.behaviors[behavior_event].onset
                behavior_offsets = self.behaviors[behavior_event].offset
                color = behavior_colors[behavior_event]

                for on, off in zip(behavior_onsets, behavior_offsets):
                    label = behavior_event if behavior_event not in plotted_labels else None
                    ax.axvspan(on, off, alpha=0.25, label=label, color=color)
                    plotted_labels.add(behavior_event)
    else:
        # Plot a single behavior
        if behavior_name not in self.behaviors.keys():
            raise ValueError(f"Behavior event '{behavior_name}' not found in behaviors.")
        behavior_onsets = self.behaviors[behavior_name].onset
        behavior_offsets = self.behaviors[behavior_name].offset
        color = behavior_colors.get(behavior_name, 'dodgerblue')  # Default to blue if behavior not in color map

        for on, off in zip(behavior_onsets, behavior_offsets):
            label = behavior_name if behavior_name not in plotted_labels else None
            ax.axvspan(on, off, alpha=0.25, color=color, label=label)
            plotted_labels.add(behavior_name)

    # Plot s1 introduced/removed events if provided
    if hasattr(self, 's1_events') and self.s1_events:
        for on in self.s1_events['introduced']:
            label = 's1 Introduced' if 's1 Introduced' not in plotted_labels else None
            ax.axvline(on, color='blue', linestyle='--', label=label, alpha=0.7)
            plotted_labels.add('s1 Introduced')
        for off in self.s1_events['removed']:
            label = 's1 Removed' if 's1 Removed' not in plotted_labels else None
            ax.axvline(off, color='blue', linestyle='-', label=label, alpha=0.7)
            plotted_labels.add('s1 Removed')

    # Plot s2 introduced/removed events if provided
    if hasattr(self, 's2_events') and self.s2_events:
        for on in self.s2_events['introduced']:
            label = 's2 Introduced' if 's2 Introduced' not in plotted_labels else None
            ax.axvline(on, color='red', linestyle='--', label=label, alpha=0.7)
            plotted_labels.add('s2 Introduced')
        for off in self.s2_events['removed']:
            label = 's2 Removed' if 's2 Removed' not in plotted_labels else None
            ax.axvline(off, color='red', linestyle='-', label=label, alpha=0.7)
            plotted_labels.add('s2 Removed')

    # Add labels and title
    ax.set_ylabel(y_label)
    ax.set_xlabel('Seconds')
    ax.set_title(f'{self.subject_name}: {y_title} with {behavior_name.capitalize()} Bouts' if behavior_name != 'all' else f'{self.subject_name}: {y_title} with All Behavior Events')

    ax.legend()
    plt.tight_layout()

    # Show the plot if no external axis is provided
    if ax is None:
        plt.show()

def extract_intruder_bouts(self, csv_base_path):
    """
    Extracts 's1 Introduced', 's1 Removed', 's2 Introduced', and 's2 Removed' events from a CSV file,
    and removes the ITI times (Inter-Trial Intervals) from the data using the remove_time function.

    Parameters:
    - csv_base_path (str): The file path to the CSV file.
    """
    data = pd.read_csv(csv_base_path)

    # Filter rows for specific behaviors
    s1_introduced = data[data['Behavior'] == 's1_Introduced'].head(6)  # Get first 6
    s1_removed = data[data['Behavior'] == 's1_Removed'].head(6)  # Get first 6
    s2_introduced = data[data['Behavior'] == 's2_Introduced']
    s2_removed = data[data['Behavior'] == 's2_Removed']

    # Extract event times
    s1_events = {
        "introduced": s1_introduced['Start (s)'].tolist(),
        "removed": s1_removed['Start (s)'].tolist()
    }

    s2_events = {
        "introduced": s2_introduced['Start (s)'].tolist(),
        "removed": s2_removed['Start (s)'].tolist()
    }

    self.s1_events = s1_events
    self.s2_events = s2_events

    # Now compute z-score with baseline being from initial artifact removal to the first s1 Introduced event
    if s1_events['introduced']:
        baseline_end_time = s1_events['introduced'][0]
        self.compute_zscore()
        # self.compute_zscore(method='baseline', baseline_start=self.timestamps[0], baseline_end=baseline_end_time)

    # # Remove ITI times (Time between when a mouse is removed and then introduced)
    # for i in range(len(s1_events['removed']) - 1):
    #     s1_removed_time = s1_events['removed'][i]
    #     s1_next_introduced_time = s1_events['introduced'][i + 1]
    #     self.remove_time(s1_removed_time, s1_next_introduced_time)

    # # Handle ITI between last s1 removed and first s2 introduced (if applicable)
    # if s1_events['removed'] and s2_events['introduced']:
    #     self.remove_time(s1_events['removed'][-1], s2_events['introduced'][0])

    # # Handle ITI between s2 removed and the next s2 introduced (for subsequent bouts)
    # for i in range(len(s2_events['removed']) - 1):
    #     s2_removed_time = s2_events['removed'][i]
    #     s2_next_introduced_time = s2_events['introduced'][i + 1]
    #     self.remove_time(s2_removed_time, s2_next_introduced_time)

    # print("ITI times removed successfully.")


def find_behavior_events_in_bout(self):
    """
    Finds all behavior events within each bout defined by s1 and s2 introduced and removed. 
    For each event found, returns the start time, end time, total duration, and mean z-score during the event.

    Parameters:
    - s1_events (dict): Dictionary containing "introduced" and "removed" timestamps for s1.
    - s2_events (dict, optional): Dictionary containing "introduced" and "removed" timestamps for s2.

    Returns:
    - bout_dict (dict): Dictionary where each key is the bout number (starting from 1), and the value contains 
                        details about each behavior event found in that bout.
    """
    bout_dict = {}

    # Compute z-score if not already done
    if self.zscore is None:
        self.compute_zscore()

    # Extract behavior events
    behavior_events = self.behaviors

    # Function to process a bout (sub-function within this function to avoid repetition)
    def process_bout(bout_key, start_time, end_time):
        bout_dict[bout_key] = {}

        # Iterate through each behavior event to find those within the bout
        for behavior_name, behavior_data in behavior_events.items():
            bout_dict[bout_key][behavior_name] = []

            behavior_onsets = np.array(behavior_data.onset)
            behavior_offsets = np.array(behavior_data.offset)

            # Find events that start and end within the bout
            within_bout = (behavior_onsets >= start_time) & (behavior_offsets <= end_time)

            # If any events are found in this bout, process them
            if np.any(within_bout):
                for onset, offset in zip(behavior_onsets[within_bout], behavior_offsets[within_bout]):
                    # Calculate total duration of the event
                    duration = offset - onset

                    # Find the z-score during this event
                    zscore_indices = (self.timestamps >= onset) & (self.timestamps <= offset)
                    mean_zscore = np.mean(self.zscore[zscore_indices])

                    # Store the details in the dictionary
                    event_dict = {
                        'Start Time': onset,
                        'End Time': offset,
                        'Total Duration': duration,
                        'Mean zscore': mean_zscore
                    }

                    bout_dict[bout_key][behavior_name].append(event_dict)

    # Iterate through each bout defined by s1 introduced and removed
    for i, (start_time, end_time) in enumerate(zip(self.s1_events['introduced'], self.s1_events['removed']), start=1):
        bout_key = f'Bout_s1_{i}'
        process_bout(bout_key, start_time, end_time)

    for i, (start_time, end_time) in enumerate(zip(self.s2_events['introduced'], self.s2_events['removed']), start=1):
            bout_key = f'Bout_s2_{i}'
            process_bout(bout_key, start_time, end_time)
        

    self.bout_dict = bout_dict

def get_first_behavior(self, behaviors=['Investigation', 'Approach']):
    """
    Extracts the mean z-score and other details for the first 'Investigation' and 'Approach' behavior events
    from each bout in the bout_dict and stores the values in a new dictionary.

    Parameters:
    - bout_dict (dict): Dictionary containing bout data with behavior events for each bout.
    - behaviors (list): List of behavior events to track (defaults to ['Investigation', 'Approach']).

    Returns:
    - first_behavior_dict (dict): Dictionary containing the start time, end time, duration, 
                                  and mean z-score for each behavior in each bout.
    """
    first_behavior_dict = {}

    # Loop through each bout in the bout_dict
    for bout_name, bout_data in self.bout_dict.items():
        first_behavior_dict[bout_name] = {}  # Initialize the dictionary for this bout
        
        # Loop through each behavior we want to track
        for behavior in behaviors:
            # Check if behavior exists in bout_data and if it contains valid event data
            if behavior in bout_data and isinstance(bout_data[behavior], list) and len(bout_data[behavior]) > 0:
                # Access the first event for the behavior
                first_event = bout_data[behavior][0]  # Assuming this is a list of events
                
                # Extract the relevant details for this behavior event
                first_behavior_dict[bout_name][behavior] = {
                    'Start Time': first_event['Start Time'],
                    'End Time': first_event['End Time'],
                    'Total Duration': first_event['End Time'] - first_event['Start Time'],
                    'Mean zscore': first_event['Mean zscore']
                }
            else:
                # If the behavior doesn't exist in this bout, add None placeholders
                first_behavior_dict[bout_name][behavior] = {
                    'Start Time': None,
                    'End Time': None,
                    'Total Duration': None,
                    'Mean zscore': None
                }


    self.first_behavior_dict = first_behavior_dict


def calculate_meta_data(self):
    """
    Calculate the total amount of 'Investigation' and 'Approach' time for each bout.
    This function will store the result in self.hab_dishab_metadata as a dictionary
    with the total duration for each behavior per bout.

    The structure of self.hab_dishab_metadata will be:
    {
        'bout_1': {
            'Total Investigation Time': X seconds,
            'Total Approach Time': Y seconds
        },
        'bout_2': {
            'Total Investigation Time': Z seconds,
            'Total Approach Time': W seconds
        },
        ...
    }
    """

    # Loop through each bout in the bout_dict
    for bout_name, bout_data in self.bout_dict.items():
        # Initialize total times for this bout
        total_investigation_time = 0
        total_approach_time = 0

        # Calculate total 'Investigation' time
        if 'Investigation' in bout_data and isinstance(bout_data['Investigation'], list):
            for event in bout_data['Investigation']:
                start_time = event['Start Time']
                end_time = event['End Time']
                total_investigation_time += end_time - start_time  # Add duration to total

        # Calculate total 'Approach' time
        if 'Approach' in bout_data and isinstance(bout_data['Approach'], list):
            for event in bout_data['Approach']:
                start_time = event['Start Time']
                end_time = event['End Time']
                total_approach_time += end_time - start_time  # Add duration to total

        # Store the total times in the metadata dictionary for this bout
        self.hab_dishab_metadata[bout_name] = {
            'Total Investigation Time': total_investigation_time,
            'Total Approach Time': total_approach_time
        }


'''********************************** FOR GROUP CLASS  **********************************'''
def hab_dishab_processing(self):
    """
    Processes each block for the habituation-dishabituation experiment by finding behavior events,
    getting the first behavior, and calculating metadata for each block.

    For each bout (e.g., s1, s2), it extracts investigation and approach times as well as mean DA 
    (dFF or z-score) for investigation and approach, storing them in a group-based DataFrame.
    
    The DataFrame has the following columns:
    - Subject: Block name
    - Investigation Total Time: Total investigation time for each subject
    - Approach Total Time: Total approach time for each subject
    - First Investigation Mean DA: Mean DA for the first investigation bout
    - First Approach Mean DA: Mean DA for the first approach bout
    - s1 Investigation Time, s2 Investigation Time, ... : Investigation time for each bout
    - s1 Approach Time, s2 Approach Time, ... : Approach time for each bout
    - s1 Investigation Mean DA, s2 Investigation Mean DA, ... : Mean DA for investigation for each bout
    - s1 Approach Mean DA, s2 Approach Mean DA, ... : Mean DA for approach for each bout
    """
    # Initialize a list to hold the data
    data_rows = []

    for block_folder, tdt_data_obj in self.blocks.items():
        csv_file_name = f"{block_folder}.csv"
        csv_file_path = os.path.join(self.csv_base_path, csv_file_name)

        if os.path.exists(csv_file_path):
            print(f"Processing {block_folder}...")

            # Call the three functions in sequence using the CSV file path
            tdt_data_obj.extract_intruder_bouts(csv_file_path)
            tdt_data_obj.find_behavior_events_in_bout()  # Find behavior events within bouts
            tdt_data_obj.get_first_behavior()            # Get the first behavior in each bout
            tdt_data_obj.calculate_meta_data()           # Calculate metadata for each bout
            name = tdt_data_obj.subject_name

            # Initialize variables to store total times and mean DA for this block
            total_investigation_time = 0
            total_approach_time = 0
            first_investigation_mean_DA = None
            first_approach_mean_DA = None

            # Initialize dictionaries to store information by bout (s1, s2, etc.)
            bout_investigation_times = {}
            bout_approach_times = {}
            bout_investigation_mean_DA = {}
            bout_approach_mean_DA = {}

            # Loop through the bouts to gather the information for each bout
            for bout_key, behavior_dict in tdt_data_obj.bout_dict.items():
                investigation_times = 0
                approach_times = 0
                mean_DA_investigation = 0
                mean_DA_approach = 0

                # Process Investigation events within the bout
                if 'Investigation' in behavior_dict:
                    for event in behavior_dict['Investigation']:
                        investigation_times += event['End Time'] - event['Start Time']
                    total_investigation_time += investigation_times

                # Process Approach events within the bout
                if 'Approach' in behavior_dict:
                    for event in behavior_dict['Approach']:
                        approach_times += event['End Time'] - event['Start Time']
                    total_approach_time += approach_times

                # Process mean DA values from the first behavior dict
                if bout_key in tdt_data_obj.first_behavior_dict:
                    first_dict = tdt_data_obj.first_behavior_dict[bout_key]

                    # Mean DA during Investigation
                    if 'Investigation' in first_dict and first_dict['Investigation']['Mean zscore'] is not None:
                        mean_DA_investigation = first_dict['Investigation']['Mean zscore']

                    # Mean DA during Approach
                    if 'Approach' in first_dict and first_dict['Approach']['Mean zscore'] is not None:
                        mean_DA_approach = first_dict['Approach']['Mean zscore']

                # Store bout-specific data
                bout_investigation_times[bout_key] = investigation_times
                bout_approach_times[bout_key] = approach_times
                bout_investigation_mean_DA[bout_key] = mean_DA_investigation
                bout_approach_mean_DA[bout_key] = mean_DA_approach

            # Get the first bout values for the first investigation and approach
            if 'Bout_s1' in bout_investigation_mean_DA:
                first_investigation_mean_DA = bout_investigation_mean_DA['Bout_s1']
            if 'Bout_s1' in bout_approach_mean_DA:
                first_approach_mean_DA = bout_approach_mean_DA['Bout_s1']

            # Create a row for this block (subject)
            row_data = {
                "Subject": name,
                "Investigation Total Time": total_investigation_time,
                "Approach Total Time": total_approach_time,
                "First Investigation Mean DA": first_investigation_mean_DA,
                "First Approach Mean DA": first_approach_mean_DA
            }

            # Add bout-specific columns (e.g., s1, s2, etc.)
            for bout_key in bout_investigation_times.keys():
                row_data[f'{bout_key} Investigation Time'] = bout_investigation_times[bout_key]
                row_data[f'{bout_key} Approach Time'] = bout_approach_times[bout_key]
                row_data[f'{bout_key} First Investigation Mean DA'] = bout_investigation_mean_DA[bout_key]
                row_data[f'{bout_key} First Approach Mean DA'] = bout_approach_mean_DA[bout_key]

            # Append the row to the data rows list
            data_rows.append(row_data)

            print(f"Finished processing {block_folder}")

    # Convert the list of data rows into a DataFrame
    df = pd.DataFrame(data_rows)

    # Store the resulting DataFrame as an attribute
    self.hab_dishab_df = df


def hab_dishab_plot_individual_behavior(self, behavior_name='all', plot_type='zscore', figsize=(18, 5)):
    """
    Plots the specified behavior and y-axis signal type for each processed block.
    
    Parameters:
    - behavior_name: The name of the behavior to plot.
    - plot_type: The type of signal to plot ('dFF', 'zscore', or 'raw').
    - figsize: The size of the figure.
    """
    # Determine the number of rows based on the number of blocks
    rows = len(self.blocks)
    figsize = (figsize[0], figsize[1] * rows)

    # Initialize the figure with the calculated size and adjust font size
    fig, axs = plt.subplots(rows, 1, figsize=figsize)
    axs = axs.flatten()
    plt.rcParams.update({'font.size': 16})

    # Loop over each block and plot
    for i, (block_folder, tdt_data_obj) in enumerate(self.blocks.items()):
        # Plot the behavior event using the plot_behavior_event method in the single block class
        if i == 0:
            # For the first plot, include the legend
            tdt_data_obj.hab_dishab_plot_behavior_event(behavior_name=behavior_name, plot_type=plot_type, ax=axs[i])
        else:
            # For the other plots, skip the legend
            tdt_data_obj.hab_dishab_plot_behavior_event(behavior_name=behavior_name, plot_type=plot_type, ax=axs[i])
            axs[i].get_legend().remove()

        subject_name = tdt_data_obj.subject_name
        axs[i].set_title(f'{subject_name}: {plot_type.capitalize()} Signal with {behavior_name.capitalize()} Bouts' if behavior_name != 'all' else f'{subject_name}: {plot_type.capitalize()} Signal with All Bouts', fontsize=18)

    plt.tight_layout()
    plt.show()

def plot_investigation_vs_dff_all(self):
    """
    Plot investigation duration vs. mean Z-scored ΔF/F during 1st investigation for all blocks,
    color-coded by individual subject identity.
    """
    investigation_durations = []
    mean_zscored_dffs = []
    subject_names = []

    # Loop through each block in self.blocks
    for block_name, block_data in self.blocks.items():
        if block_data.first_behavior_dict:
            for bout, behavior_data in block_data.first_behavior_dict.items():
                if 'Investigation' in behavior_data:
                    # Extract investigation duration and mean DA for this investigation
                    investigation_durations.append(behavior_data['Investigation']['Total Duration'])
                    mean_zscored_dffs.append(behavior_data['Investigation']['Mean zscore'])
                    subject_names.append(block_name)  # Block name as the subject identifier
    
    # Convert lists to numpy arrays
    investigation_durations = np.array(investigation_durations, dtype=np.float64)
    mean_zscored_dffs = np.array(mean_zscored_dffs, dtype=np.float64)
    subject_names = np.array(subject_names)

    # Filter out any entries where either investigation_durations or mean_zscored_dffs is NaN
    valid_indices = ~np.isnan(investigation_durations) & ~np.isnan(mean_zscored_dffs)
    investigation_durations = investigation_durations[valid_indices]
    mean_zscored_dffs = mean_zscored_dffs[valid_indices]
    subject_names = subject_names[valid_indices]

    if len(mean_zscored_dffs) == 0 or len(investigation_durations) == 0:
        print("No valid data points for correlation.")
        return

    # Calculate Pearson correlation
    r, p = stats.pearsonr(mean_zscored_dffs, investigation_durations)

    # Get unique subjects and assign colors
    unique_subjects = np.unique(subject_names)
    color_palette = sns.color_palette("hsv", len(unique_subjects))
    subject_color_map = {subject: color_palette[i] for i, subject in enumerate(unique_subjects)}

    # Plotting the scatter plot
    plt.figure(figsize=(12, 6))
    
    for subject in unique_subjects:
        # Create a mask for each subject
        mask = subject_names == subject
        plt.scatter(mean_zscored_dffs[mask], investigation_durations[mask], 
                    color=subject_color_map[subject], label=subject, alpha=0.6)

    # Adding the regression line
    slope, intercept = np.polyfit(mean_zscored_dffs, investigation_durations, 1)
    plt.plot(mean_zscored_dffs, slope * mean_zscored_dffs + intercept, color='black', linestyle='--')

    # Add labels and title
    plt.xlabel('Mean Z-scored ΔF/F during 1st investigation')
    plt.ylabel('Investigation duration (s)')
    plt.title('Correlation between Investigation Duration and DA Response (All Blocks)')
    
    # Display Pearson correlation and p-value
    plt.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.2e}\nn = {len(mean_zscored_dffs)} sessions',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    # Add a legend with subject names
    plt.legend(title='Subject', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()



def plot_all_investigation_vs_dff_all(self):
    """
    Plot investigation duration vs. mean Z-scored ΔF/F during all investigations for all blocks,
    color-coded by individual subject identity.
    """
    investigation_durations = []
    mean_zscored_dffs = []
    subject_names = []

    # Loop through each block in self.blocks
    for block_name, block_data in self.blocks.items():
        if block_data.bout_dict:  # Make sure bout_dict is populated
            for bout, behavior_data in block_data.bout_dict.items():
                if 'Investigation' in behavior_data:
                    # Loop through all investigation events in this bout
                    for event in behavior_data['Investigation']:
                        # Extract investigation duration and mean DA for this investigation
                        investigation_durations.append(event['Total Duration'])
                        mean_zscored_dffs.append(event['Mean zscore'])
                        subject_names.append(block_name)  # Block name as the subject identifier
    
    # Convert lists to numpy arrays
    investigation_durations = np.array(investigation_durations, dtype=np.float64)
    mean_zscored_dffs = np.array(mean_zscored_dffs, dtype=np.float64)
    subject_names = np.array(subject_names)

    # Filter out any entries where either investigation_durations or mean_zscored_dffs is NaN
    valid_indices = ~np.isnan(investigation_durations) & ~np.isnan(mean_zscored_dffs)
    investigation_durations = investigation_durations[valid_indices]
    mean_zscored_dffs = mean_zscored_dffs[valid_indices]
    subject_names = subject_names[valid_indices]

    if len(mean_zscored_dffs) == 0 or len(investigation_durations) == 0:
        print("No valid data points for correlation.")
        return

    # Calculate Pearson correlation
    r, p = stats.pearsonr(mean_zscored_dffs, investigation_durations)

    # Get unique subjects and assign colors
    unique_subjects = np.unique(subject_names)
    color_palette = sns.color_palette("hsv", len(unique_subjects))
    subject_color_map = {subject: color_palette[i] for i, subject in enumerate(unique_subjects)}

    # Plotting the scatter plot
    plt.figure(figsize=(12, 6))
    
    for subject in unique_subjects:
        # Create a mask for each subject
        mask = subject_names == subject
        plt.scatter(mean_zscored_dffs[mask], investigation_durations[mask], 
                    color=subject_color_map[subject], label=subject, alpha=0.6)

    # Adding the regression line
    slope, intercept = np.polyfit(mean_zscored_dffs, investigation_durations, 1)
    plt.plot(mean_zscored_dffs, slope * mean_zscored_dffs + intercept, color='black', linestyle='--')

    # Add labels and title
    plt.xlabel('Mean Z-scored ΔF/F during investigations')
    plt.ylabel('Investigation duration (s)')
    plt.title('Correlation between Investigation Duration and DA Response (All Investigations)')

    # Display Pearson correlation and p-value
    plt.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.2e}\nn = {len(mean_zscored_dffs)} sessions',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    # Add a legend with subject names
    plt.legend(title='Subject', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

def plot_investigation_mean_DA_boutwise(self):
    """
    Plot the mean Z-scored ΔF/F (mean DA) for all investigation events during each bout.
    The bar graph will show the mean ± SEM across all subjects, with individual subject data points.
    """
    # Initialize a dictionary to collect the mean DA values for each bout (use block_data.bout_dict keys dynamically)
    bout_mean_DA_dict = {}

    # Loop through each block in self.blocks to dynamically build bout_mean_DA_dict
    for block_name, block_data in self.blocks.items():
        if block_data.bout_dict:  # Ensure bout_dict is populated
            for bout, behavior_data in block_data.bout_dict.items():
                if 'Investigation' in behavior_data:
                    # Initialize a list for each bout if it doesn't exist
                    if bout not in bout_mean_DA_dict:
                        bout_mean_DA_dict[bout] = []
                    # For each bout, collect the mean z-score (mean DA) for all investigation events
                    for event in behavior_data['Investigation']:
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
    ax.set_ylabel('Mean DA (z-score)', fontsize=12)
    ax.set_xlabel('Bouts', fontsize=12)
    ax.set_title('Mean DA (Z-scored ΔF/F) During Investigation Across Bouts', fontsize=14)

    # Set x-ticks to match the dynamically captured bout labels
    ax.set_xticks(np.arange(len(bouts)))
    ax.set_xticklabels(bouts, fontsize=12)

    # Add the legend outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Subjects")

    # Display the plot
    plt.show()


def plot_investigation_durations_boutwise(self):
    """
    Plot the total investigation duration for all investigation events during each bout.
    The bar graph will show the mean ± SEM across all subjects, with individual subject data points.
    """
    # Initialize a dictionary to collect the investigation durations for each bout
    bout_investigation_duration_dict = {}

    # Loop through each block in self.blocks to dynamically build bout_investigation_duration_dict
    for block_name, block_data in self.blocks.items():
        if block_data.bout_dict:  # Ensure bout_dict is populated
            # print(block_data.bout_dict.keys())
            for bout, behavior_data in block_data.bout_dict.items():
                if 'Investigation' in behavior_data:
                    # Initialize a list for each bout if it doesn't exist
                    if bout not in bout_investigation_duration_dict:
                        bout_investigation_duration_dict[bout] = []
                    # For each bout, collect the investigation duration for all investigation events
                    for event in behavior_data['Investigation']:
                        bout_investigation_duration_dict[bout].append(event['Total Duration'])

    # Prepare lists to store the mean and SEM for each bout
    bouts = list(bout_investigation_duration_dict.keys())  # Dynamically get the bout names
    mean_investigation_duration_per_bout = []
    sem_investigation_duration_per_bout = []

    # Calculate the mean and SEM for each bout
    for bout in bouts:
        investigation_duration_values = bout_investigation_duration_dict[bout]
        if investigation_duration_values:  # If there are any values for the bout
            mean_investigation_duration_per_bout.append(np.nanmean(investigation_duration_values))
            sem_investigation_duration_per_bout.append(np.nanstd(investigation_duration_values) / np.sqrt(len(investigation_duration_values)))
        else:
            mean_investigation_duration_per_bout.append(np.nan)  # If no data for this bout, append NaN
            sem_investigation_duration_per_bout.append(np.nan)

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot the bar plot with error bars
    bars = ax.bar(bouts, mean_investigation_duration_per_bout, yerr=sem_investigation_duration_per_bout, capsize=5, color='skyblue', edgecolor='black', label='Mean')

    # Plot each individual's investigation durations for each bout as scatter points
    for i, bout in enumerate(bouts):
        investigation_duration_values = bout_investigation_duration_dict[bout]
        for subject_data in investigation_duration_values:
            ax.scatter(bout, subject_data, color='black', alpha=0.7)  # Plot individual points for each bout

    # Add labels, title, and format
    ax.set_ylabel('Investigation Duration (s)', fontsize=12)
    ax.set_xlabel('Bouts', fontsize=12)
    ax.set_title('Investigation Duration During Investigation Across Bouts', fontsize=14)

    # Set x-ticks to match the dynamically captured bout labels
    ax.set_xticks(np.arange(len(bouts)))
    ax.set_xticklabels(bouts, fontsize=12)

    # Add the legend outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Subjects")

    # Display the plot
    plt.show()
