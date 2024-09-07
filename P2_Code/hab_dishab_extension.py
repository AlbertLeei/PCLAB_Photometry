import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def hab_dishab_plot_behavior_event(self, behavior_name, plot_type='dFF', ax=None):
    """
    Plots Delta F/F (dFF) or z-scored signal with behavior events for the habituation-dishabituation experiment.

    Parameters:
    - behavior_name (str): The name of the behavior to plot. Use 'all' to plot all behaviors.
    - plot_type (str): The type of plot. Options are 'dFF' and 'zscore'.
    - s1_events, s2_events (dict): Dictionaries containing 'introduced' and 'removed' timestamps for s1 and s2.
    - ax: An optional matplotlib Axes object. If provided, the plot will be drawn on this Axes.
    """
    s1_events = self.s1_events
    s2_events = self.s2_events

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
            if behavior_event in behavior_colors:  # Use behavior_colors to ensure distinct colors
                behavior_onsets = self.behaviors[behavior_event].onset
                behavior_offsets = self.behaviors[behavior_event].offset
                color = behavior_colors[behavior_event]

                for on, off in zip(behavior_onsets, behavior_offsets):
                    # Only add a label if it hasn't been plotted yet
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
    if s1_events:
        for on in s1_events['introduced']:
            label = 's1 Introduced' if 's1 Introduced' not in plotted_labels else None
            ax.axvline(on, color='blue', linestyle='--', label=label, alpha=0.7)
            plotted_labels.add('s1 Introduced')
        for off in s1_events['removed']:
            label = 's1 Removed' if 's1 Removed' not in plotted_labels else None
            ax.axvline(off, color='blue', linestyle='-', label=label, alpha=0.7)
            plotted_labels.add('s1 Removed')

    # Plot s2 introduced/removed events if provided
    if s2_events:
        for on in s2_events['introduced']:
            label = 's2 Introduced' if 's2 Introduced' not in plotted_labels else None
            ax.axvline(on, color='red', linestyle='--', label=label, alpha=0.7)
            plotted_labels.add('s2 Introduced')
        for off in s2_events['removed']:
            label = 's2 Removed' if 's2 Removed' not in plotted_labels else None
            ax.axvline(off, color='red', linestyle='-', label=label, alpha=0.7)
            plotted_labels.add('s2 Removed')

    ax.set_ylabel(y_label)
    ax.set_xlabel('Seconds')
    ax.set_title(f'{self.subject_name}: {y_title} with {behavior_name.capitalize()} Bouts' if behavior_name != 'all' else f'{self.subject_name}: {y_title} with All Behavior Events')

    ax.legend()
    plt.tight_layout()
    plt.show()

def extract_intruder_bouts(self, csv_base_path):
    """
    Extracts 's1 Introduced', 's1 Removed', 's2 Introduced', and 's2 Removed' events from a CSV file.

    Parameters:
    - csv_base_path (str): The file path to the CSV file.
    """
    data = pd.read_csv(csv_base_path)

    # Filter rows for specific behaviors
    s1_introduced = data[data['Behavior'] == 's1_Introduced']
    s1_removed = data[data['Behavior'] == 's1_Removed']
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


def get_first_behavior_means(bout_dict, behaviors=['Investigation', 'Approach']):
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
    for bout_name, bout_data in bout_dict.items():
        first_behavior_dict[bout_name] = {}  # Initialize the dictionary for this bout
        
        # Loop through each behavior we want to track
        for behavior in behaviors:
            if behavior in bout_data:
                # Find the first occurrence of this behavior event
                first_event = bout_data[behavior]
                
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

    return first_behavior_dict