import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import scipy.stats as stats


'''********************************** FOR SINGLE OBJECT  **********************************'''
def hc_extract_intruder_bouts(self, csv_base_path):
    """
    Extracts 'Short_Term_Introduced', 'Short_Term_Removed', 'Novel_Introduced', 'Novel_Removed',
    'Long_Term_Introduced', and 'Long_Term_Removed' events from a CSV file,
    and removes the ITI times (Inter-Trial Intervals) from the data using the remove_time function.

    Parameters:
    - csv_base_path (str): The file path to the CSV file.
    """
    data = pd.read_csv(csv_base_path)

    # Filter rows for specific behaviors
    short_term_introduced = data[data['Behavior'] == 'Short_Term_Introduced']
    short_term_removed = data[data['Behavior'] == 'Short_Term_Removed']
    novel_introduced = data[data['Behavior'] == 'Novel_Introduced']
    novel_removed = data[data['Behavior'] == 'Novel_Removed']
    long_term_introduced = data[data['Behavior'] == 'Long_Term_Introduced']
    long_term_removed = data[data['Behavior'] == 'Long_Term_Removed']

    # Extract event times
    short_term_events = {
        "introduced": short_term_introduced['Start (s)'].tolist(),
        "removed": short_term_removed['Start (s)'].tolist()
    }

    novel_events = {
        "introduced": novel_introduced['Start (s)'].tolist(),
        "removed": novel_removed['Start (s)'].tolist()
    }

    long_term_events = {
        "introduced": long_term_introduced['Start (s)'].tolist(),
        "removed": long_term_removed['Start (s)'].tolist()
    }

    self.short_term_events = short_term_events
    self.novel_events = novel_events
    self.long_term_events = long_term_events

    # Compute z-score with baseline being from initial artifact removal to the first Short_Term_Introduced event
    if short_term_events['introduced']:
        baseline_end_time = short_term_events['introduced'][0]
        self.compute_zscore()
        # self.compute_zscore(method='baseline', baseline_start=self.timestamps[0], baseline_end=baseline_end_time)

def hc_plot_behavior_event(self, behavior_name='all', plot_type='dFF', ax=None):
    """
    Plots Delta F/F (dFF) or z-scored signal with behavior events for the new habituation-context experiment.

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

    # Plot Short_Term_Introduced/Removed events if provided
    if hasattr(self, 'short_term_events') and self.short_term_events:
        for on in self.short_term_events['introduced']:
            label = 'Short Term Introduced' if 'Short Term Introduced' not in plotted_labels else None
            ax.axvline(on, color='blue', linestyle='--', label=label, alpha=0.7)
            plotted_labels.add('Short Term Introduced')
        for off in self.short_term_events['removed']:
            label = 'Short Term Removed' if 'Short Term Removed' not in plotted_labels else None
            ax.axvline(off, color='blue', linestyle='-', label=label, alpha=0.7)
            plotted_labels.add('Short Term Removed')

    # Plot Novel_Introduced/Removed events if provided
    if hasattr(self, 'novel_events') and self.novel_events:
        for on in self.novel_events['introduced']:
            label = 'Novel Introduced' if 'Novel Introduced' not in plotted_labels else None
            ax.axvline(on, color='orange', linestyle='--', label=label, alpha=0.7)
            plotted_labels.add('Novel Introduced')
        for off in self.novel_events['removed']:
            label = 'Novel Removed' if 'Novel Removed' not in plotted_labels else None
            ax.axvline(off, color='orange', linestyle='-', label=label, alpha=0.7)
            plotted_labels.add('Novel Removed')

    # Plot Long_Term_Introduced/Removed events if provided
    if hasattr(self, 'long_term_events') and self.long_term_events:
        for on in self.long_term_events['introduced']:
            label = 'Long Term Introduced' if 'Long Term Introduced' not in plotted_labels else None
            ax.axvline(on, color='red', linestyle='--', label=label, alpha=0.7)
            plotted_labels.add('Long Term Introduced')
        for off in self.long_term_events['removed']:
            label = 'Long Term Removed' if 'Long Term Removed' not in plotted_labels else None
            ax.axvline(off, color='red', linestyle='-', label=label, alpha=0.7)
            plotted_labels.add('Long Term Removed')

    # Add labels and title
    ax.set_ylabel(y_label)
    ax.set_xlabel('Seconds')
    ax.set_title(f'{self.subject_name}: {y_title} with {behavior_name.capitalize()} Bouts' if behavior_name != 'all' else f'{self.subject_name}: {y_title} with All Behavior Events')

    ax.legend()
    plt.tight_layout()

    # Show the plot if no external axis is provided
    if ax is None:
        plt.show()

def hc_find_behavior_events_in_bout(self):
    """
    Finds all behavior events within each bout defined by Short Term, Novel, and Long Term introduced and removed.
    For each event found, returns the start time, end time, total duration, and mean z-score during the event.

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

    # Iterate through each bout defined by Short Term introduced and removed
    for i, (start_time, end_time) in enumerate(zip(self.short_term_events['introduced'], self.short_term_events['removed']), start=1):
        bout_key = f'Bout_Short_Term_{i}'
        process_bout(bout_key, start_time, end_time)

    # Iterate through each bout defined by Novel introduced and removed
    for i, (start_time, end_time) in enumerate(zip(self.novel_events['introduced'], self.novel_events['removed']), start=1):
        bout_key = f'Bout_Novel_{i}'
        process_bout(bout_key, start_time, end_time)

    # Iterate through each bout defined by Long Term introduced and removed
    for i, (start_time, end_time) in enumerate(zip(self.long_term_events['introduced'], self.long_term_events['removed']), start=1):
        bout_key = f'Bout_Long_Term_{i}'
        process_bout(bout_key, start_time, end_time)

    self.bout_dict = bout_dict


'''********************************** GROUP OBJECT **********************************'''

def hc_processing(self):
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

            tdt_data_obj.hc_extract_intruder_bouts(csv_file_path)
            tdt_data_obj.hc_find_behavior_events_in_bout()






