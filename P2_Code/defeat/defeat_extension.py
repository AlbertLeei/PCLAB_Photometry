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



import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def plot_peth_individual_traces(self, 
                                signal_type='zscore', 
                                bout=None, 
                                title='PETH for First Investigation', 
                                color='#00B7D7', 
                                display_pre_time=3, 
                                display_post_time=3, 
                                yticks_interval=2, 
                                figsize=(14, 8),
                                ax=None):
    """
    Plots individual traces of the peri-event time histogram (PETH) for a single bout with larger font sizes.

    Parameters:
    - signal_type (str): The type of signal to plot. Options are 'zscore' or 'dFF'.
    - bout (str): The bout name to plot. If None, defaults to the first bout in the list ['short_term_1', 'short_term_2', 'novel_1', 'long_term_1'].
    - title (str): Title for the entire figure.
    - color (str): Color for the individual traces (default is cyan '#00B7D7'). Individual traces can have varying shades if desired.
    - display_pre_time (float): Time before the event onset to display on the x-axis (in seconds).
    - display_post_time (float): Time after the event onset to display on the x-axis (in seconds).
    - yticks_interval (float): Interval between y-ticks on the plot.
    - figsize (tuple): Size of the figure in inches (width, height).
    - ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, a new figure and axis are created.

    Returns:
    - None. Displays the individual PETH traces for the specified bout.
    """
    # Define default bouts if none provided
    default_bouts = ['short_term_1', 'short_term_2', 'novel_1', 'long_term_1']
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
    
    # Define a common time axis based on the overlapping time range
    # Assuming all time axes are uniformly sampled, we use the first block as reference
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
    
    # Define the display window
    display_start = -display_pre_time
    display_end = display_post_time
    display_start_idx = np.searchsorted(common_time_axis, display_start, side='left')
    display_end_idx = np.searchsorted(common_time_axis, display_end, side='right')
    
    # Handle cases where display window exceeds common_time_axis
    display_start_idx = max(display_start_idx, 0)
    display_end_idx = min(display_end_idx, len(common_time_axis))
    
    # Truncate data to the display window
    display_time = common_time_axis[display_start_idx:display_end_idx]
    truncated_traces = all_traces[:, display_start_idx:display_end_idx]
    
    print(f"Display window: {display_start} to {display_end} seconds.")
    print(f"Displaying {len(display_time)} data points.")
    
    # Create the plot or use the provided ax
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None  # Only needed if you need to save or further manipulate the figure
    
    # Plot each individual trace
    for trace in truncated_traces:
        ax.plot(display_time, trace, color=color, alpha=1)  # Reduced alpha for better visibility
    
    # Add a vertical line at event onset
    ax.axvline(0, color='black', linestyle='--', label='Event Onset', linewidth=3)  # Thicker event onset line
    
    # Customize x-axis
    ax.set_xticks([display_time[0], 0, display_time[-1]])
    ax.set_xticklabels([f'{display_time[0]:.1f}', '0', f'{display_time[-1]:.1f}'], fontsize=24)  # Increased fontsize for x-tick labels
    ax.set_xlabel('Time from Onset (s)', fontsize=32)  # Increased fontsize for x-axis label
    
    # Customize y-axis
    y_min, y_max = ax.get_ylim()
    y_ticks = np.arange(np.floor(y_min / yticks_interval) * yticks_interval, 
                        np.ceil(y_max / yticks_interval) * yticks_interval + yticks_interval, 
                        yticks_interval)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y:.0f}' for y in y_ticks], fontsize=30)  # Increased fontsize for y-tick labels
    ax.set_ylabel(f'{signal_type.capitalize()} ΔF/F', fontsize= 32)  # Increased fontsize for y-axis label

    # Set title
    ax.set_title(title, fontsize=32)  # Increased fontsize for title
    
    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Customize spines' linewidth
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    
    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=30, width=2)  # Increased tick label size and tick width
    
    # Add legend for event onset if not already present
    handles, labels = ax.get_legend_handles_labels()
    if 'Event Onset' in labels:
        ax.legend(fontsize=20)

    plt.savefig('all.png', transparent=True, bbox_inches='tight', pad_inches=0.1)

    if fig is not None:
        fig.tight_layout()
        plt.show()
    else:
        return ax
