import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def rt_processing(self):
    for block_folder, tdt_data_obj in self.blocks.items():
        print(f"Reward Training Processing {block_folder}...")

        tdt_data_obj.remove_initial_LED_artifact(t=30)
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
        tdt_data_obj.combine_consecutive_behaviors(behavior_name='all', bout_time_threshold=2, min_occurrences=1)
        tdt_data_obj.remove_short_behaviors(behavior_name='all', min_duration=0.2)

        tdt_data_obj.verify_signal()
        tdt_data_obj.behaviors['sound cues'] = tdt_data_obj.behaviors.pop('PC0_')
        tdt_data_obj.behaviors['port entries'] = tdt_data_obj.behaviors.pop('PC3_')
        tdt_data_obj.behaviors['sound cues'].onset = tdt_data_obj.behaviors['sound cues'].onset[1:]
        tdt_data_obj.behaviors['sound cues'].offset = tdt_data_obj.behaviors['sound cues'].offset[1:]
        tdt_data_obj.behaviors['port entries'].onset = tdt_data_obj.behaviors['port entries'].onset[1:]
        tdt_data_obj.behaviors['port entries'].offset = tdt_data_obj.behaviors['port entries'].offset[1:]

        port_entries_onset = np.array(tdt_data_obj.behaviors['port entries'].onset)
        port_entries_offset = np.array(tdt_data_obj.behaviors['port entries'].offset)

        # Get the first sound cue onset time
        first_sound_cue_onset = tdt_data_obj.behaviors['sound cues'].onset[0]

        # Use np.where to find indices of port entries occurring after the first sound cue onset
        indices = np.where(port_entries_onset >= first_sound_cue_onset)[0]

        # Filter the port entries using the indices
        tdt_data_obj.behaviors['port entries'].onset = port_entries_onset[indices].tolist()
        tdt_data_obj.behaviors['port entries'].offset = port_entries_offset[indices].tolist()

        tdt_data_obj.find_overlapping_port_entries()
        tdt_data_obj.align_port_entries_to_sound_cues()


def rt_plot_behavior_event(self, behavior_name='all', plot_type='zscore', ax=None):
    """
    Plots z-scored Delta F/F (dFF) with behavior events for the experiment.

    Parameters:
    - behavior_name (str): The name of the behavior to plot. Use 'all' to plot all behaviors.
    - plot_type (str): The type of plot. Only 'zscore' is supported in this context.
    - ax: An optional matplotlib Axes object. If provided, the plot will be drawn on this Axes.
    """
    # Prepare data based on plot type
    if plot_type == 'zscore':
        if self.zscore is None:
            self.compute_zscore()
        y_data = self.zscore
        y_label = 'z-score'
        y_title = 'Z-scored ΔF/F Signal'
    else:
        raise ValueError("Invalid plot_type. Only 'zscore' is supported.")

    # Create plot if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 8))

    ax.plot(self.timestamps, np.array(y_data), linewidth=2, color='black', label='zscore')

    # Define specific colors for behaviors
    behavior_colors = {'sound cues': 'blue', 'port entries': 'green'}

    # Track which labels have been plotted to avoid duplicates
    plotted_labels = set()

    # Plot behavior spans
    if behavior_name == 'all':
        for behavior_event in self.behaviors.keys():
            behavior_onsets = self.behaviors[behavior_event].onset
            behavior_offsets = self.behaviors[behavior_event].offset
            color = behavior_colors.get(behavior_event, 'gray')  # Use a default color if not specified

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
        color = behavior_colors.get(behavior_name, 'gray')  # Default color if not in behavior_colors

        for on, off in zip(behavior_onsets, behavior_offsets):
            label = behavior_name if behavior_name not in plotted_labels else None
            ax.axvspan(on, off, alpha=0.25, color=color, label=label)
            plotted_labels.add(behavior_name)

    # Add labels and title
    ax.set_ylabel(y_label)
    ax.set_xlabel('Seconds')
    ax.set_title(f'{self.subject_name}: {y_title} with {behavior_name.capitalize()} Events' if behavior_name != 'all' else f'{self.subject_name}: {y_title} with All Behavior Events')

    ax.legend()
    plt.tight_layout()

    # Show the plot if no external axis is provided
    if ax is None:
        plt.show()


def rt_plot_individual_behavior(self, behavior_name='all', plot_type='zscore', figsize=(18, 5)):
    """
    Plots the specified behavior and z-scored ΔF/F signal for each processed block in the experiment.

    Parameters:
    - behavior_name: The name of the behavior to plot.
    - plot_type: The type of signal to plot ('zscore').
    - figsize: The size of the figure.
    """
    # Determine the number of rows based on the number of blocks
    rows = len(self.blocks)
    figsize = (figsize[0], figsize[1] * rows)

    # Initialize the figure with the calculated size and adjust font size
    fig, axs = plt.subplots(rows, 1, figsize=figsize)
    axs = axs.flatten() if rows > 1 else [axs]
    plt.rcParams.update({'font.size': 16})

    # Loop over each block and plot
    for i, (block_folder, tdt_data_obj) in enumerate(self.blocks.items()):
        # Correct method call
        tdt_data_obj.rt_plot_behavior_event(behavior_name=behavior_name, plot_type=plot_type, ax=axs[i])

        subject_name = tdt_data_obj.subject_name
        axs[i].set_title(f'{subject_name}: {plot_type.capitalize()} Signal with {behavior_name.capitalize()} Events' if behavior_name != 'all' else f'{subject_name}: {plot_type.capitalize()} Signal with All Events', fontsize=18)

        # Remove legend from all but the first subplot to avoid duplication
        if i != 0:
            axs[i].get_legend().remove()

    plt.tight_layout()
    plt.show()


def find_overlapping_port_entries(self):
        """
        Finds port entries that overlap with sound cues and saves the data in the object.
        """
        import numpy as np

        # Extract sound cues and port entries
        sound_cues_onsets = np.array(self.behaviors['sound cues'].onset)
        sound_cues_offsets = np.array(self.behaviors['sound cues'].offset)
        port_entries_onsets = np.array(self.behaviors['port entries'].onset)
        port_entries_offsets = np.array(self.behaviors['port entries'].offset)

        overlapping_port_entries = []

        for i in range(len(sound_cues_onsets)):
            sc_onset = sound_cues_onsets[i]
            sc_offset = sound_cues_offsets[i]

            # Find overlapping port entries
            overlap_indices = np.where((port_entries_onsets < sc_offset) & (port_entries_offsets > sc_onset))[0]

            overlapping_entries = {
                'sound_cue_onset': sc_onset,
                'sound_cue_offset': sc_offset,
                'port_entries_onsets': port_entries_onsets[overlap_indices],
                'port_entries_offsets': port_entries_offsets[overlap_indices]
            }
            overlapping_port_entries.append(overlapping_entries)

        # Save the overlapping port entries in the object
        self.overlapping_port_entries = overlapping_port_entries


def align_port_entries_to_sound_cues(self):
    """
    Aligns port entries to sound cues and saves the data in the object.
    """

    # Extract sound cues and port entries
    sound_cues_onsets = np.array(self.behaviors['sound cues'].onset)
    sound_cues_offsets = np.array(self.behaviors['sound cues'].offset)
    port_entries_onsets = np.array(self.behaviors['port entries'].onset)
    port_entries_offsets = np.array(self.behaviors['port entries'].offset)

    sound_cue_port_entries = []

    for i in range(len(sound_cues_onsets)):
        sc_onset = sound_cues_onsets[i]
        sc_offset = sound_cues_offsets[i]

        # Determine the end time (next sound cue onset or end of session)
        if i < len(sound_cues_onsets) - 1:
            end_time = sound_cues_onsets[i + 1]
        else:
            end_time = max(self.timestamps)  # End of the recording session

        # Find port entries from sound cue onset to the next sound cue onset
        indices = np.where((port_entries_onsets >= sc_onset) & (port_entries_onsets < end_time))[0]

        associated_entries = {
            'sound_cue_onset': sc_onset,
            'sound_cue_offset': sc_offset,
            'port_entries_onsets': port_entries_onsets[indices],
            'port_entries_offsets': port_entries_offsets[indices]
        }
        sound_cue_port_entries.append(associated_entries)

    # Save the aligned port entries in the object
    self.sound_cue_port_entries = sound_cue_port_entries


def rt_extract_and_plot(self, n_entries=10, max_entries_per_sound_cue=3, behavior='port entries'):
    """
    Extracts and plots the mean DA during specified behavior events within sound cues for each subject.

    Parameters:
    - n_entries (int): The total number of events to consider per subject (default 10).
    - max_entries_per_sound_cue (int): The maximum number of events to consider per sound cue (default 3).
    - behavior (str): The behavior to extract mean DA for ('port entries' or 'sound cues').

    Returns:
    - behavior_mean_df (pd.DataFrame): A DataFrame where each row represents a subject,
                                       and columns represent the mean DA during each event.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    data_list = []

    for block_data in self.blocks.values():
        subject_name = block_data.subject_name

        # Ensure that 'sound_cue_port_entries' is populated
        if hasattr(block_data, 'sound_cue_port_entries') and block_data.sound_cue_port_entries:
            # Collect mean DA values for events
            da_values = []
            event_numbers = []

            total_events_collected = 0

            for sc_entry in block_data.sound_cue_port_entries:
                if total_events_collected >= n_entries:
                    break  # Stop if we've collected the desired number of events

                if behavior == 'port entries':
                    event_onsets = sc_entry['port_entries_onsets']
                    event_offsets = sc_entry['port_entries_offsets']
                elif behavior == 'sound cues':
                    event_onsets = [sc_entry['sound_cue_onset']]
                    event_offsets = [sc_entry['sound_cue_offset']]
                else:
                    print(f"Invalid behavior '{behavior}' specified.")
                    continue

                # Determine the number of events to collect in this sound cue
                num_events_in_this_cue = min(len(event_onsets), max_entries_per_sound_cue)
                for i in range(num_events_in_this_cue):
                    if total_events_collected >= n_entries:
                        break  # Stop if we've collected the desired number of events

                    event_onset = event_onsets[i]
                    event_offset = event_offsets[i]

                    # Extract DA signal during this event
                    indices = np.where((block_data.timestamps >= event_onset) & (block_data.timestamps <= event_offset))[0]
                    da_segment = block_data.zscore[indices]

                    # Compute mean DA
                    mean_da = np.mean(da_segment)
                    da_values.append(mean_da)
                    event_numbers.append(total_events_collected + 1)

                    total_events_collected += 1

            if da_values:
                # Create a dictionary with subject name and mean DA values
                data_dict = {'Subject': subject_name}
                for i, mean_da in enumerate(da_values):
                    data_dict[f'Event {i+1}'] = mean_da

                data_list.append(data_dict)
        else:
            print(f"No sound cue port entries found for subject {subject_name}")

    # Convert the data_list into a DataFrame
    behavior_mean_df = pd.DataFrame(data_list)
    behavior_mean_df.set_index('Subject', inplace=True)

    return behavior_mean_df


def rt_compute_peth_per_event(self, behavior_name='sound cues', n_events=None, pre_time=5, post_time=5, bin_size=0.1):
    """
    Computes the peri-event time histogram (PETH) data for each occurrence of a given behavior across all blocks.
    Stores the peri-event data (zscore, time axis) for each event index as a class variable.

    Parameters:
    - behavior_name (str): The name of the behavior to generate the PETH for (e.g., 'sound cues').
    - n_events (int): The maximum number of events to analyze. If None, analyze all events.
    - pre_time (float): The time in seconds to include before the event.
    - post_time (float): The time in seconds to include after the event.
    - bin_size (float): The size of each bin in the histogram (in seconds).

    Returns:
    - None. Stores peri-event data for each event index across blocks as a class variable.
    """
    import numpy as np

    # Initialize a dictionary to store peri-event data for each event index
    self.peri_event_data_per_event = {}

    # First, determine the maximum number of events across all blocks if n_events is None
    if n_events is None:
        n_events = 0
        for block_data in self.blocks.values():
            if behavior_name in block_data.behaviors:
                num_events = len(block_data.behaviors[behavior_name].onset)
                if num_events > n_events:
                    n_events = num_events

    # Define a common time axis
    time_axis = np.arange(-pre_time, post_time + bin_size, bin_size)
    self.time_axis = time_axis  # Store time_axis in the object

    # Initialize data structure
    for event_index in range(n_events):
        self.peri_event_data_per_event[event_index] = []

    # Loop through each block in self.blocks
    for block_name, block_data in self.blocks.items():
        # Get the onset times for the behavior
        if behavior_name in block_data.behaviors:
            event_onsets = block_data.behaviors[behavior_name].onset
            # Limit to the first n_events if necessary
            event_onsets = event_onsets[:n_events]
            # For each event onset, compute the peri-event data
            for i, event_onset in enumerate(event_onsets):
                # Define start and end times
                start_time = event_onset - pre_time
                end_time = event_onset + post_time

                # Get indices for timestamps within this window
                indices = np.where((block_data.timestamps >= start_time) & (block_data.timestamps <= end_time))[0]

                if len(indices) == 0:
                    continue  # Skip if no data in this window

                # Extract the corresponding zscore values
                signal_segment = block_data.zscore[indices]

                # Create a time axis relative to the event onset
                timestamps_segment = block_data.timestamps[indices] - event_onset

                # Interpolate the signal onto the common time axis
                interpolated_signal = np.interp(time_axis, timestamps_segment, signal_segment)

                # Store the interpolated signal in the data structure
                self.peri_event_data_per_event[i].append(interpolated_signal)
        else:
            print(f"Behavior '{behavior_name}' not found in block '{block_name}'.")

    # Now, self.peri_event_data_per_event[event_index] contains a list of traces across blocks for that event index


def rt_plot_peth_per_event(self, signal_type='zscore', error_type='sem', title='PETH for First n Sound Cues',
                           color='#00B7D7', display_pre_time=5, display_post_time=5, yticks_interval=2):
    """
    Plots the PETH for each event index (e.g., each sound cue) across all blocks in one figure with subplots.

    Parameters:
    - signal_type (str): The type of signal to plot. Options are 'zscore' or 'dFF'.
    - error_type (str): The type of error to plot. Options are 'sem' for Standard Error of the Mean or 'std' for Standard Deviation.
    - title (str): Title for the figure.
    - color (str): Color for both the trace line and the error area (default is cyan '#00B7D7').
    - display_pre_time (float): How much time to show before the event on the x-axis (default is 5 seconds).
    - display_post_time (float): How much time to show after the event on the x-axis (default is 5 seconds).
    - yticks_interval (float): Interval for the y-ticks on the plots (default is 2).

    Returns:
    - None. Displays the PETH plot for each event index in one figure.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Get the time axis
    time_axis = self.time_axis

    # Determine the indices for the display range
    display_start_idx = np.searchsorted(time_axis, -display_pre_time)
    display_end_idx = np.searchsorted(time_axis, display_post_time)
    time_axis = time_axis[display_start_idx:display_end_idx]

    num_events = len(self.peri_event_data_per_event)
    if num_events == 0:
        print("No peri-event data available to plot.")
        return

    # Create subplots arranged horizontally
    fig, axes = plt.subplots(1, num_events, figsize=(5 * num_events, 5), sharey=True)

    # If there's only one event, make axes a list to keep the logic consistent
    if num_events == 1:
        axes = [axes]

    for idx, event_index in enumerate(range(num_events)):
        ax = axes[idx]
        event_traces = self.peri_event_data_per_event[event_index]
        if not event_traces:
            print(f"No data for event {event_index + 1}")
            continue
        # Convert list of traces to numpy array
        event_traces = np.array(event_traces)
        # Truncate the traces to the display range
        event_traces = event_traces[:, display_start_idx:display_end_idx]

        # Calculate the mean across blocks
        mean_trace = np.mean(event_traces, axis=0)
        # Calculate SEM or Std across blocks depending on the selected error type
        if error_type == 'sem':
            error_trace = np.std(event_traces, axis=0) / np.sqrt(len(event_traces))  # SEM
            error_label = 'SEM'
        elif error_type == 'std':
            error_trace = np.std(event_traces, axis=0)  # Standard Deviation
            error_label = 'Std'

        # Plot the mean trace with SEM/Std shaded area, with customizable color
        ax.plot(time_axis, mean_trace, color=color, label=f'Mean {signal_type.capitalize()}', linewidth=1.5)  # Trace color
        ax.fill_between(time_axis, mean_trace - error_trace, mean_trace + error_trace, color=color, alpha=0.3, label=error_label)  # Error color

        # Plot event onset line
        ax.axvline(0, color='black', linestyle='--', label='Event onset')

        # Set the x-ticks to show only the last time, 0, and the very end time
        ax.set_xticks([time_axis[0], 0, time_axis[-1]])
        ax.set_xticklabels([f'{time_axis[0]:.1f}', '0', f'{time_axis[-1]:.1f}'], fontsize=12)

        # Set the y-tick labels with specified interval
        if idx == 0:
            y_min, y_max = ax.get_ylim()
            y_ticks = np.arange(np.floor(y_min / yticks_interval) * yticks_interval,
                                np.ceil(y_max / yticks_interval) * yticks_interval + yticks_interval,
                                yticks_interval)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f'{y:.0f}' for y in y_ticks], fontsize=12)
        else:
            ax.set_yticks([])  # Hide y-ticks for other subplots

        ax.set_xlabel('Time (s)', fontsize=14)
        if idx == 0:
            ax.set_ylabel(f'{signal_type.capitalize()} dFF', fontsize=14)

        # Set the title for each event
        ax.set_title(f'Event {event_index + 1}', fontsize=14)

        # Remove the right and top spines for each subplot
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # Adjust layout and add a common title
    plt.suptitle(title, fontsize=16)
    plt.show()
