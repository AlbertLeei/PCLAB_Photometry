import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def d_proc_extract_bout(self, csv_file_path):
    """
    Extracts 'Subject Introduced' and 'Subject Removed' events from the CSV file
    and determines the start and end times of the bout.

    Parameters:
    - csv_file_path (str): The file path to the CSV file.
    """
    # Load the CSV file into a DataFrame
    data = pd.read_csv(csv_file_path)

    # Filter rows for specific behaviors
    introduced_events = data[data['Behavior'].str.lower() == 'subject_introduced']
    removed_events = data[data['Behavior'].str.lower() == 'subject_removed']

    if len(introduced_events) == 0 or len(removed_events) == 0:
        print(f"No 'Subject Introduced' or 'Subject Removed' events found in {csv_file_path}.")
        return

    # Assuming there is only one bout, take the first introduced and removed times
    bout_start_time = introduced_events['Start (s)'].iloc[0]
    bout_end_time = removed_events['Start (s)'].iloc[0]

    self.bout_times = {
        "start": bout_start_time,
        "end": bout_end_time
    }

    # Compute z-score with baseline being from start of recording to the bout start
    self.compute_zscore()



def d_proc_find_behavior_events_in_bout(self):
    """
    Finds all behavior events within the bout defined by Subject Introduced and Subject Removed.
    For each event found, returns the start time, end time, total duration, and mean z-score during the event.

    Updates self.bout_dict with the behavior events in the bout.
    """
    bout_dict = {}

    # Compute z-score if not already done
    if self.zscore is None:
        self.compute_zscore()

    # Extract behavior events
    behavior_events = self.behaviors

    # Process the single bout
    bout_key = 'bout_1'  # Assuming there's only one bout, adjust if there are multiple
    start_time = self.bout_times['start']
    end_time = self.bout_times['end']
    bout_dict[bout_key] = {}

    # Iterate through each behavior event to find those within the bout
    for behavior_name, behavior_data in behavior_events.items():
        bout_dict[bout_key][behavior_name] = []

        behavior_onsets = np.array(behavior_data['onset'])
        behavior_offsets = np.array(behavior_data['offset'])

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

    self.bout_dict = bout_dict


def d_proc_plot_behavior_event(self, behavior_name='all', plot_type='dFF', ax=None):
    """
    Plots Delta F/F (dFF) or z-scored signal with behavior events for the Social Defeat experiment.

    Parameters:
    - behavior_name (str): The name of the behavior to plot. Use 'all' to plot all behaviors.
    - plot_type (str): The type of plot. Options are 'dFF' and 'zscore'.
    - ax: An optional matplotlib Axes object. If provided, the plot will be drawn on this Axes.
    """
    # Prepare data based on plot type
    y_data = []
    if plot_type == 'dFF':
        if self.dFF is None:
            self.compute_dFF()
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
    behavior_colors = {'Investigation': 'dodgerblue', 'Approach': 'green', 'Defeat': 'red', 'Aggression': 'purple'}

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

    # Plot bout start and end
    if hasattr(self, 'bout_times'):
        start_time = self.bout_times['start']
        end_time = self.bout_times['end']
        ax.axvline(start_time, color='blue', linestyle='--', label='Bout Start', alpha=0.7)
        ax.axvline(end_time, color='red', linestyle='--', label='Bout End', alpha=0.7)
        plotted_labels.add('Bout Start')
        plotted_labels.add('Bout End')

    # Add labels and title
    ax.set_ylabel(y_label)
    ax.set_xlabel('Seconds')
    ax.set_title(f'{self.subject_name}: {y_title} with {behavior_name.capitalize()} Events' if behavior_name != 'all' else f'{self.subject_name}: {y_title} with All Behavior Events')

    # Adjust x-tick marks
    num_ticks = len(ax.get_xticks()) * 3  # Triple the number of ticks
    ax.set_xticks(np.linspace(self.timestamps[0], self.timestamps[-1], num_ticks))

    ax.legend()
    plt.tight_layout()

    # Show the plot if no external axis is provided
    if ax is None:
        plt.show()


def d_proc_processing(self):
    """
    Processes all blocks in the group for the Social Defeat experiment.
    """
    for block_folder, tdt_data_obj in self.blocks.items():
        csv_file_name = f"{block_folder}.csv"
        csv_file_path = os.path.join(self.csv_base_path, csv_file_name)

        if os.path.exists(csv_file_path):
            print(f"Social Defeat Processing {block_folder}...")

            # Pass the CSV file path to the extract_bout function
            tdt_data_obj.d_proc_extract_bout(csv_file_path)
            tdt_data_obj.d_proc_find_behavior_events_in_bout()
            tdt_data_obj.get_first_behavior()  # Get the first behavior in the bout


def d_proc_plot_individual_behavior(self, behavior_name='all', plot_type='zscore', figsize=(18, 5)):
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
        # Plot the behavior event using the d_proc_plot_behavior_event method
        tdt_data_obj.d_proc_plot_behavior_event(behavior_name=behavior_name, plot_type=plot_type, ax=axs[i])

        # Set the title for each subplot
        subject_name = tdt_data_obj.subject_name
        axs[i].set_title(f'{subject_name}: {plot_type.capitalize()} Signal with {behavior_name.capitalize()} Events' 
                         if behavior_name != 'all' 
                         else f'{subject_name}: {plot_type.capitalize()} Signal with All Events', fontsize=18)

    # Adjust the layout to prevent overlap
    plt.tight_layout()
    
    # Show the plot
    plt.show()
