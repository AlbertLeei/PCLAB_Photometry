import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
# from scipy import stats

import sys
import scipy.stats as stats
from matplotlib.lines import Line2D  # Added import



def sp_extract_intruder_events(self, behavior_csv_path, cup_assignment_csv_path):
    """
    Extracts behaviors from the behavior CSV and maps them to the correct agent in each cup based on the 
    cup assignment CSV. Keeps 'sniff' and 'chew' behaviors separate.
    """
    # Load both CSV files
    behavior_data = pd.read_csv(behavior_csv_path)
    cup_assignments = pd.read_csv(cup_assignment_csv_path)

    # Clean up the cup assignments DataFrame by removing extra columns
    cup_assignments = cup_assignments[['Subject', 'Cup 1', 'Cup 2', 'Cup 3', 'Cup 4']]

    # Make behavior names lowercase for consistency
    behavior_data['Behavior'] = behavior_data['Behavior'].str.lower()

    # Initialize a dictionary to store events based on agents (bouts)
    behavior_event_dict = {}

    # Subject name should be obtained from the class attribute `self.subject_name`
    subject = self.subject_name

    # Loop through each row in the behavior data
    for index, row in behavior_data.iterrows():
        behavior_type = row['Behavior']

        # Check if the behavior involves a cup (e.g., 'sniff cup 3' or 'chew cup 4')
        if 'cup' in behavior_type:
            try:
                # Extract the cup number
                cup_number = int(behavior_type.split('cup')[-1].strip())

                # Ensure the subject exists in the cup assignments
                if subject not in cup_assignments['Subject'].values:
                    print(f"Skipping row: subject {subject} not found in cup assignments")
                    continue

                # Ensure the cup number is within valid bounds (1 to 4)
                if cup_number < 1 or cup_number > 4:
                    print(f"Skipping row: invalid cup number {cup_number}")
                    continue

                # Find the corresponding agent (bout) from the cup assignment CSV based on subject and cup number
                agent = cup_assignments.loc[cup_assignments['Subject'] == subject, f'Cup {cup_number}'].iloc[0]

                # Extract the specific behavior (e.g., 'sniff' or 'chew')
                behavior = behavior_type.split(' ')[0]  # 'sniff' or 'chew'

                # Initialize the dictionary for the agent if not already done
                if agent not in behavior_event_dict:
                    behavior_event_dict[agent] = {}

                # Initialize the list for the specific behavior if not already done
                if behavior not in behavior_event_dict[agent]:
                    behavior_event_dict[agent][behavior] = []

                # Add the event to the agent's specific behavior list
                behavior_event_dict[agent][behavior].append({
                    'Start Time': row['Start (s)'],
                    'End Time': row['Stop (s)'],
                    'Duration': row['Duration (s)'],
                    # 'Behavior': behavior,  # Optional: Since the key represents the behavior
                })
            except (ValueError, IndexError) as e:
                # Handle cases where the behavior does not have a valid cup number or the subject is invalid
                print(f"Skipping row due to error: {e}")

        elif 'introduced' in behavior_type or 'removed' in behavior_type:
            # For 'introduced' or 'removed' events, assign them to a special "Subject Presence" category
            movement_agent = 'Subject Presence'
            
            if movement_agent not in behavior_event_dict:
                behavior_event_dict[movement_agent] = {'introduced': [], 'removed': []}

            # Store the event under 'introduced' or 'removed' as appropriate
            if 'introduced' in behavior_type:
                behavior_event_dict[movement_agent]['introduced'].append({
                    'Start Time': row['Start (s)'],
                })
            elif 'removed' in behavior_type:
                behavior_event_dict[movement_agent]['removed'].append({
                    'Start Time': row['Start (s)'],
                })

        else:
            # Handle behaviors that don't involve cups or subject movements
            print(f"Skipping unrecognized behavior: {behavior_type}")

    # Store the result in the class, with keys as agents (bouts)
    self.behavior_event_dict = behavior_event_dict


def sp_extract_intruder_events_combined(self, behavior_csv_path, cup_assignment_csv_path):
    """
    Extracts behaviors from the behavior CSV and maps them to the correct agent in each cup based on the 
    cup assignment CSV. Combines 'sniff' and 'chew' behaviors into 'investigation'.
    """
    # Load both CSV files
    behavior_data = pd.read_csv(behavior_csv_path)
    cup_assignments = pd.read_csv(cup_assignment_csv_path)

    # Clean up the cup assignments DataFrame by removing extra columns
    cup_assignments = cup_assignments[['Subject', 'Cup 1', 'Cup 2', 'Cup 3', 'Cup 4']]

    # Make behavior names lowercase for consistency
    behavior_data['Behavior'] = behavior_data['Behavior'].str.lower()

    # Initialize a dictionary to store events based on agents (bouts)
    behavior_event_dict = {}

    # Subject name should be obtained from the class attribute `self.subject_name`
    subject = self.subject_name

    # Loop through each row in the behavior data
    for index, row in behavior_data.iterrows():
        behavior_type = row['Behavior']

        # Check if the behavior involves a cup (e.g., 'sniff cup 3' or 'chew cup 4')
        if 'cup' in behavior_type:
            try:
                # Extract the cup number
                cup_number = int(behavior_type.split('cup')[-1].strip())

                # Ensure the subject exists in the cup assignments
                if subject not in cup_assignments['Subject'].values:
                    print(f"Skipping row: subject {subject} not found in cup assignments")
                    continue

                # Ensure the cup number is within valid bounds (1 to 4)
                if cup_number < 1 or cup_number > 4:
                    print(f"Skipping row: invalid cup number {cup_number}")
                    continue

                # Find the corresponding agent (bout) from the cup assignment CSV based on subject and cup number
                agent = cup_assignments.loc[cup_assignments['Subject'] == subject, f'Cup {cup_number}'].iloc[0]

                # Initialize the dictionary for the agent if not already done
                if agent not in behavior_event_dict:
                    behavior_event_dict[agent] = {'investigation': []}

                # Add the event to the agent's 'investigation' list
                behavior_event_dict[agent]['investigation'].append({
                    'Start Time': row['Start (s)'],
                    'End Time': row['Stop (s)'],
                    'Duration': row['Duration (s)'],
                    'Behavior': behavior_type.split(' ')[0],  # Store 'sniff' or 'chew' if needed later
                })
            except (ValueError, IndexError) as e:
                # Handle cases where the behavior does not have a valid cup number or the subject is invalid
                print(f"Skipping row due to error: {e}")

        elif 'introduced' in behavior_type or 'removed' in behavior_type:
            # For 'introduced' or 'removed' events, assign them to a special "Subject Presence" category
            movement_agent = 'Subject Presence'
            
            if movement_agent not in behavior_event_dict:
                behavior_event_dict[movement_agent] = {'introduced': [], 'removed': []}

            # Store the event under 'introduced' or 'removed' as appropriate
            if 'introduced' in behavior_type:
                behavior_event_dict[movement_agent]['introduced'].append({
                    'Start Time': row['Start (s)'],
                })
            elif 'removed' in behavior_type:
                behavior_event_dict[movement_agent]['removed'].append({
                    'Start Time': row['Start (s)'],
                })

        else:
            # Handle behaviors that don't involve cups or subject movements
            print(f"Skipping unrecognized behavior: {behavior_type}")

    # Store the result in the class, with keys as agents (bouts)
    self.behavior_event_dict = behavior_event_dict


def sp_plot_behavior_event(self, plot_type='dFF', ax=None):
    """
    Plots Delta F/F (dFF) or z-scored signal with behavior events for the social preference experiment.

    Parameters:
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
    behavior_colors = {'sniff': 'dodgerblue', 'chew': 'green', 'novel': 'orange'}

    # Track which agents and behaviors have been plotted to avoid duplicates in the legend
    plotted_labels = set()

    # Color map for agents (if you want to assign unique colors per agent)
    color_map = plt.get_cmap('tab10')  # Use 'tab10' or any other colormap

    # Plot agent behavior spans
    for idx, (agent, behaviors) in enumerate(self.behavior_event_dict.items()):
        if agent == "Subject Presence":
            continue  # Skip Subject Presence for now

        for behavior, events in behaviors.items():
            agent_color = behavior_colors.get(behavior, color_map(idx % 10))  # Use predefined color or assign from colormap

            for event in events:
                on, off = event['Start Time'], event['End Time']
                ax.axvspan(on, off, alpha=0.25, color=agent_color, label=f"{agent} - {behavior}")

            # Add to legend if not already added
            label = f"{agent} - {behavior}"
            if label not in plotted_labels:
                ax.axvspan(0, 0, alpha=0.25, color=agent_color, label=label)  # Dummy span for legend
                plotted_labels.add(label)

    # Plot Subject Presence (Introduced/Removed events separately)
    if "Subject Presence" in self.behavior_event_dict:
        for event in self.behavior_event_dict["Subject Presence"].get('introduced', []):
            on = event['Start Time']  # Only Start Time is used here
            label = "Subject Introduced" if "Subject Introduced" not in plotted_labels else None
            ax.axvline(on, color='red', linestyle='--', label=label, alpha=0.7)
            if label:
                plotted_labels.add(label)
        for event in self.behavior_event_dict["Subject Presence"].get('removed', []):
            on = event['Start Time']  # Only Start Time is used here
            label = "Subject Removed" if "Subject Removed" not in plotted_labels else None
            ax.axvline(on, color='blue', linestyle='-', label=label, alpha=0.7)
            if label:
                plotted_labels.add(label)

    # Add labels and title
    ax.set_ylabel(y_label)
    ax.set_xlabel('Seconds')
    ax.set_title(f'{self.subject_name}: {y_title} with All Agent Events')

    # Add the legend for the agents and behaviors
    ax.legend(title="Agents & Behaviors", loc='upper left')

    plt.tight_layout()

    # Show the plot if no external axis is provided
    if ax is None:
        plt.show()


def sp_plot_behavior_event_combined(self, plot_type='dFF', ax=None):
    """
    Plots Delta F/F (dFF) or z-scored signal with behavior events for the social preference experiment.

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

    # Define specific colors for behaviors (optional, for any additional behaviors like 'sniff', 'chew', etc.)
    behavior_colors = {'sniff': 'dodgerblue', 'chew': 'green', 'novel': 'orange'}

    # Track which agents have been plotted to avoid duplicates in the legend
    plotted_agents = set()
    agent_colors = {}  # To keep track of agent and assigned color

    # Color map for agents
    color_map = plt.get_cmap('tab10')  # Use 'tab10' or any other colormap

    # Plot agent behavior spans
    agent_list = [agent for agent in self.behavior_event_dict.keys() if agent != "Subject Presence"]  # Exclude "Subject Presence"
    for idx, agent in enumerate(agent_list):
        agent_events = self.behavior_event_dict[agent]
        agent_color = color_map(idx % 10)  # Pick color from the color map

        # Assign color to the agent
        agent_colors[agent] = agent_color

        for event in agent_events:
            behavior_type = event.get('Behavior', '')  # Get the behavior type

            # Skip 'introduced' and 'removed' behaviors
            if 'introduced' in behavior_type or 'removed' in behavior_type:
                continue

            # Plot the behavior event
            on, off = event['Start Time'], event['End Time']
            ax.axvspan(on, off, alpha=0.25, color=agent_color)

        # Add agent to the legend only if not already added
        if agent not in plotted_agents:
            ax.axvspan(0, 0, alpha=0.25, color=agent_color, label=agent)  # Add to legend with a dummy span
            plotted_agents.add(agent)

    # Plot Subject Presence (Introduced/Removed events separately)
    if "Subject Presence" in self.behavior_event_dict:
        for event in self.behavior_event_dict["Subject Presence"]:
            behavior = event['Behavior']
            on = event['Start Time']  # Only Start Time is used here
            label = "Subject Movements" if "Subject Movements" not in plotted_agents else None
            if 'introduced' in behavior:
                ax.axvline(on, color='red', linestyle='--', label=label, alpha=0.7)
            elif 'removed' in behavior:
                ax.axvline(on, color='blue', linestyle='-', label=label, alpha=0.7)
            plotted_agents.add("Subject Movements")

    # Add labels and title
    ax.set_ylabel(y_label)
    ax.set_xlabel('Seconds')
    ax.set_title(f'{self.subject_name}: {y_title} with All Agent Events')

    # Add the legend for the agents
    ax.legend(title="Agents", loc='upper left')

    plt.tight_layout()

    # Show the plot if no external axis is provided
    if ax is None:
        plt.show()


def sp_remove_time_around_subject_introduced(self, buffer_time=9):
    """
    Removes the specified buffer time before the 'subject introduced' event.

    Parameters:
    buffer_time (float): The time in seconds to remove before and after the 'subject introduced' event.
    """
    # Check if 'Subject Presence' exists in the behavior_event_dict
    if 'Subject Presence' not in self.behavior_event_dict:
        print(f"No 'Subject Presence' events found.")
        return

    # Get the 'introduced' events from 'Subject Presence'
    subject_presence_events = self.behavior_event_dict['Subject Presence']
    
    if 'introduced' not in subject_presence_events:
        print(f"No 'introduced' events found in 'Subject Presence'.")
        return

    # Get the first 'introduced' event
    introduced_event = subject_presence_events['introduced'][0] if subject_presence_events['introduced'] else None

    # If no 'introduced' event is found, exit
    if not introduced_event:
        print(f"No 'subject introduced' event found.")
        return

    # Get the start time of the 'subject introduced' event
    introduced_time = introduced_event['Start Time']

    # Define the start time to remove (buffer_time before the introduction)
    start_time = introduced_time - buffer_time

    # Ensure start time is valid
    if start_time < 0:
        start_time = 0

    # Remove the time segment
    self.remove_time_segment(start_time, introduced_time)
    # print(f"Removed {buffer_time} seconds before 'subject introduced' at {introduced_time}s.")



# Group Data
def sp_processing(self, cup_assignment_csv_path):
    data_rows = []

    for block_folder, tdt_data_obj in self.blocks.items():
        csv_file_name = f"{block_folder}.csv"
        csv_file_path = os.path.join(self.csv_base_path, csv_file_name)

        if os.path.exists(csv_file_path):
            print(f"Social Pref Processing {block_folder}...")

            # Call the three functions in sequence using the CSV file path
            tdt_data_obj.sp_extract_intruder_events(csv_file_path, cup_assignment_csv_path)
            tdt_data_obj.sp_remove_time_around_subject_introduced(buffer_time = 9)
            # tdt_data_obj.get_first_behavior()            # Get the first behavior in each bout
            tdt_data_obj.remove_initial_LED_artifact(t=30)
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


def sp_compute_first_bout_peth_all_blocks(self, bouts=None, pre_time=5, post_time=5, bin_size=0.1):
    """
    Computes the peri-event time histogram (PETH) data for the first occurrence of each behavior in each bout.
    Stores the peri-event data (zscore, dFF, and time axis) for each bout as a class variable.

    Parameters:
    - bouts (list): A list of bout names (agents) to process.
    - pre_time (float): The time in seconds to include before the event.
    - post_time (float): The time in seconds to include after the event.
    - bin_size (float): The size of each bin in the histogram (in seconds).

    Returns:
    None. Stores peri-event data for all blocks and bouts as a class variable.
    """
    if bouts is None:
        bouts = ['Novel', 'Long_Term', 'Short_Term', 'Nothing']  # Update with your actual bout names

    self.peri_event_data_all_blocks = {}  # Initialize a dictionary to store PETH data for each bout

    # Track the shortest time axis across all blocks and bouts
    min_time_length = float('inf')

    # Loop through each block in self.blocks
    for block_name, block_data in self.blocks.items():
        self.peri_event_data_all_blocks[block_name] = {}  # Initialize PETH storage for each block

        # Ensure that zscore and timestamps are computed and synchronized
        if block_data.zscore is None or block_data.timestamps is None:
            print(f"Block {block_name} is missing zscore or timestamps data.")
            continue

        # Loop through each bout in the specified bouts
        for bout in bouts:
            if bout in block_data.behavior_event_dict:
                # Get the behaviors for this bout
                behaviors = block_data.behavior_event_dict[bout]

                for behavior, events in behaviors.items():
                    if len(events) > 0:
                        # Get the first event of this behavior
                        first_event = events[0]
                        event_time = first_event['Start Time']

                        # Compute peri-event data directly within this function
                        # Define the time window
                        start_time = event_time - pre_time
                        end_time = event_time + post_time

                        # Get indices within the time window
                        indices = (block_data.timestamps >= start_time) & (block_data.timestamps <= end_time)

                        # Check if indices are valid
                        if not np.any(indices):
                            print(f"No data found in the time window around {event_time} for block {block_name}, bout {bout}, behavior {behavior}.")
                            continue  # Skip to the next behavior

                        # Extract data within the window
                        time_axis = block_data.timestamps[indices] - event_time  # Align time to event_time
                        zscore = block_data.zscore[indices]
                        dFF = block_data.dFF[indices]

                        # Optional: Resample data to the specified bin_size if needed
                        # For simplicity, we'll assume data is already at the desired resolution

                        # Store the peri-event data
                        peri_event_data = {'time_axis': time_axis, 'zscore': zscore, 'dFF': dFF}
                        self.peri_event_data_all_blocks[block_name][f"{bout}_{behavior}"] = peri_event_data

                        # Check the time axis length to find the shortest one
                        if len(time_axis) < min_time_length:
                            min_time_length = len(time_axis)
                    else:
                        print(f"No '{behavior}' events found in {bout} for block {block_name}.")
            else:
                print(f"No data for bout '{bout}' in block '{block_name}'.")

    # Truncate all traces to the shortest time axis length to ensure consistency
    for block_name, bout_data in self.peri_event_data_all_blocks.items():
        for bout_behavior, peri_event_data in bout_data.items():
            for key in ['zscore', 'dFF', 'time_axis']:  # Truncate zscore, dFF, and time_axis
                peri_event_data[key] = peri_event_data[key][:min_time_length]


def sp_compute_first_bout_peth_all_blocks_combined(self, bouts=None, pre_time=5, post_time=5, bin_size=0.1):
    """
    Computes the peri-event time histogram (PETH) data for the first occurrence of 'investigation' in each bout.
    Stores the peri-event data (zscore, dFF, and time axis) for each bout as a class variable.

    Parameters:
    bouts (list): A list of bout names (agents) to process.
    pre_time (float): The time in seconds to include before the event.
    post_time (float): The time in seconds to include after the event.
    bin_size (float): The size of each bin in the histogram (in seconds).

    Returns:
    None. Stores peri-event data for all blocks and bouts as a class variable.
    """
    if bouts is None:
        bouts = ['Novel', 'Long_Term', 'Short_Term', 'Nothing']  # Update with your actual bout names

    self.peri_event_data_all_blocks = {}  # Initialize a dictionary to store PETH data for each bout

    # Track the shortest time axis across all blocks and bouts
    min_time_length = float('inf')

    # Loop through each block in self.blocks
    for block_name, block_data in self.blocks.items():
        self.peri_event_data_all_blocks[block_name] = {}  # Initialize PETH storage for each block

        # Ensure that zscore and timestamps are computed and synchronized
        if block_data.zscore is None or block_data.timestamps is None:
            print(f"Block {block_name} is missing zscore or timestamps data.")
            continue

        # Loop through each bout in the specified bouts
        for bout in bouts:
            if bout in block_data.behavior_event_dict:
                # Get the list of 'investigation' events for this bout
                investigation_events = block_data.behavior_event_dict[bout]['investigation']

                if len(investigation_events) > 0:
                    # Get the first 'investigation' event
                    first_event = investigation_events[0]
                    event_time = first_event['Start Time']

                    # Compute peri-event data directly within this function
                    # Define the time window
                    start_time = event_time - pre_time
                    end_time = event_time + post_time

                    # Get indices within the time window
                    indices = (block_data.timestamps >= start_time) & (block_data.timestamps <= end_time)

                    # Check if indices are valid
                    if not np.any(indices):
                        print(f"No data found in the time window around {event_time} for block {block_name}, bout {bout}.")
                        continue  # Skip to the next bout

                    # Extract data within the window
                    time_axis = block_data.timestamps[indices] - event_time  # Align time to event_time
                    zscore = block_data.zscore[indices]
                    dFF = block_data.dFF[indices]

                    # Optional: Resample data to the specified bin_size if needed
                    # For simplicity, we'll assume data is already at the desired resolution

                    # Store the peri-event data
                    peri_event_data = {'time_axis': time_axis, 'zscore': zscore, 'dFF': dFF}
                    self.peri_event_data_all_blocks[block_name][bout] = peri_event_data

                    # Check the time axis length to find the shortest one
                    if len(time_axis) < min_time_length:
                        min_time_length = len(time_axis)
                else:
                    print(f"No 'investigation' events found in {bout} for block {block_name}.")
            else:
                print(f"No data for bout '{bout}' in block '{block_name}'.")

    # Truncate all traces to the shortest time axis length to ensure consistency
    for block_name, bout_data in self.peri_event_data_all_blocks.items():
        for bout, peri_event_data in bout_data.items():
            for key in ['zscore', 'dFF', 'time_axis']:  # Truncate zscore, dFF, and time_axis
                peri_event_data[key] = peri_event_data[key][:min_time_length]


def plot_peri_event_data_group(self):
    """
    Plots the peri-event data (average across blocks) for each bout and behavior.
    """

    # Gather all unique bout_behavior keys
    sample_block = next(iter(self.peri_event_data_all_blocks.values()), None)
    if not sample_block:
        print("No peri-event data available to plot.")
        return

    bout_behaviors = list(sample_block.keys())
    num_bout_behaviors = len(bout_behaviors)

    fig, axes = plt.subplots(num_bout_behaviors, 1, figsize=(10, 4 * num_bout_behaviors), sharex=True)
    if num_bout_behaviors == 1:
        axes = [axes]  # Ensure axes is iterable

    for i, bout_behavior in enumerate(bout_behaviors):
        all_zscores = []
        for block_name, bout_data in self.peri_event_data_all_blocks.items():
            if bout_behavior in bout_data:
                peri_event_data = bout_data[bout_behavior]
                all_zscores.append(peri_event_data['zscore'])

        # Compute the average z-score across blocks
        if all_zscores:
            mean_zscore = np.mesan(all_zscores, axis=0)
            time_axis = peri_event_data['time_axis']  # Time axis should be the same after truncation

            axes[i].plot(time_axis, mean_zscore, label=f'{bout_behavior}')
            axes[i].set_title(f'Average Z-score for {bout_behavior.replace("_", " ")}')
            axes[i].set_ylabel('Z-score')
            axes[i].legend()
        else:
            axes[i].set_title(f'No data for {bout_behavior.replace("_", " ")}')
            axes[i].set_ylabel('Z-score')

    axes[-1].set_xlabel('Time (s) relative to event')
    plt.tight_layout()
    plt.show()


def plot_peri_event_data_group_combined(self):
    """
    Plots the peri-event data (average across blocks) for each bout.
    """
    import matplotlib.pyplot as plt

    bouts = list(self.peri_event_data_all_blocks[next(iter(self.peri_event_data_all_blocks))].keys())
    num_bouts = len(bouts)

    fig, axes = plt.subplots(num_bouts, 1, figsize=(8, num_bouts * 3), sharex=True)
    if num_bouts == 1:
        axes = [axes]  # Ensure axes is iterable

    for i, bout in enumerate(bouts):
        all_zscores = []
        for block_name, bout_data in self.peri_event_data_all_blocks.items():
            if bout in bout_data:
                peri_event_data = bout_data[bout]
                all_zscores.append(peri_event_data['zscore'])

        # Compute the average z-score across blocks
        if all_zscores:
            mean_zscore = np.mean(all_zscores, axis=0)
            time_axis = peri_event_data['time_axis']  # Time axis should be the same after truncation

            axes[i].plot(time_axis, mean_zscore, label=f'{bout}')
            axes[i].set_title(f'Average Z-score for {bout}')
            axes[i].set_ylabel('Z-score')
            axes[i].legend()
        else:
            axes[i].set_title(f'No data for {bout}')
            axes[i].set_ylabel('Z-score')

    axes[-1].set_xlabel('Time (s) relative to event')
    plt.tight_layout()
    plt.show()


def plot_peri_event_data_group_mean_sem(self, signal_type='zscore', save_path=None):
    """
    Plots the peri-event data (average across blocks) with mean and SEM for each bout and behavior.

    Parameters:
    - signal_type (str): The type of signal to plot ('zscore' or 'dFF').
    - save_path (str, optional): If provided, saves the plot to the specified path.

    Returns:
    - None. Displays the plot.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import sem  # Ensure SEM is imported

    # Validate signal_type
    if signal_type not in ['zscore', 'dFF']:
        raise ValueError("Invalid signal_type. Supported types are 'zscore' and 'dFF'.")

    # Gather all unique bout_behavior keys from the first block
    sample_block = next(iter(self.peri_event_data_all_blocks.values()), None)
    if not sample_block:
        print("No peri-event data available to plot.")
        return

    bout_behaviors = list(sample_block.keys())
    num_bout_behaviors = len(bout_behaviors)

    # Create subplots
    fig, axes = plt.subplots(num_bout_behaviors, 1, figsize=(10, 4 * num_bout_behaviors), sharex=True)
    if num_bout_behaviors == 1:
        axes = [axes]  # Ensure axes is iterable

    for i, bout_behavior in enumerate(bout_behaviors):
        all_signals = []
        for block_name, bout_data in self.peri_event_data_all_blocks.items():
            if bout_behavior in bout_data:
                peri_event_data = bout_data[bout_behavior]
                all_signals.append(peri_event_data[signal_type])

        # Convert list to numpy array for easier manipulation
        all_signals = np.array(all_signals)

        # Check if there are any signals to plot
        if all_signals.size == 0:
            axes[i].set_title(f'No data for {bout_behavior.replace("_", " ")}')
            axes[i].set_ylabel(f'{signal_type.upper()}')
            continue

        # Compute the mean and SEM across blocks (axis=0)
        mean_signal = np.mean(all_signals, axis=0)
        sem_signal = sem(all_signals, axis=0)

        # Extract time_axis from the first available peri_event_data
        time_axis = peri_event_data['time_axis']

        # Plot mean signal
        axes[i].plot(time_axis, mean_signal, color='blue', label=f'Mean {signal_type.upper()}')

        # Plot SEM as shaded area
        axes[i].fill_between(time_axis, mean_signal - sem_signal, mean_signal + sem_signal, color='blue', alpha=0.3, label='SEM')

        # Set title and labels
        axes[i].set_title(f'Average {signal_type.upper()} for {bout_behavior.replace("_", " ")}')
        axes[i].set_ylabel(f'{signal_type.upper()}')

        # Add legend
        axes[i].legend()

    # Set common x-label
    axes[-1].set_xlabel('Time (s) relative to event')

    # Adjust layout
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path)

    # Display the plot
    plt.show()

   


# Regular functions
def sp_extract_total_behavior_time(group_data, behaviors=['sniff', 'chew'], bouts=['Novel', 'Long_Term', 'Short_Term', 'Nothing']):
    """
    Extracts the total time spent on each behavior for each bout across all blocks in the group.

    Parameters:
    - group_data: The object containing multiple blocks of behavior event data.
    - behaviors (list): List of behaviors to extract time for.
    - bouts (list): List of bout names.

    Returns:
    pd.DataFrame: Total time spent on each behavior per bout per subject.
    """
    # Initialize an empty list to hold the data for each subject
    data_list = []

    # Loop through each block in the group data
    for block_data in group_data.blocks.values():
        # Use the subject name as the row identifier
        subject_data_dict = {'Subject': block_data.subject_name}
        
        # Initialize total times for each bout and behavior to zero
        for bout in bouts:
            for behavior in behaviors:
                subject_data_dict[f"{bout}_{behavior}"] = 0.0

        # Loop over agents (which are the bouts)
        for agent, behaviors_dict in block_data.behavior_event_dict.items():
            if agent in bouts:
                for behavior, events in behaviors_dict.items():
                    if behavior in behaviors:
                        # Sum up the durations of the behavior events
                        total_time = sum(
                            event['Duration'] for event in events
                        )
                        subject_data_dict[f"{agent}_{behavior}"] = total_time

        # Append the subject's data to the data_list
        data_list.append(subject_data_dict)

    # Convert the data_list into a DataFrame
    behavior_time_df = pd.DataFrame(data_list)

    # Set the index to 'Subject'
    behavior_time_df.set_index('Subject', inplace=True)

    return behavior_time_df


def sp_extract_average_sniff_behavior_time(group_data, bouts=['Novel', 'Long_Term', 'Short_Term', 'Nothing']):
    """
    Extracts the average duration of the 'sniff' behavior for each bout across all blocks in the group.
    
    Parameters:
    - group_data: The object containing multiple blocks of behavior event data.
    - bouts (list): List of bout names to extract average sniff durations for.
    
    Returns:
    pd.DataFrame: A DataFrame where each row represents a subject,
                  and each column represents a bout,
                  containing the average duration of 'sniff' behavior.
    """
    # Initialize an empty list to hold the data for each subject
    data_list = []
    
    # Loop through each block in the group data
    for block_data in group_data.blocks.values():
        # Use the subject name as the row identifier
        subject_data_dict = {'Subject': block_data.subject_name}
        
        # Loop through each specified bout
        for bout in bouts:
            # Initialize average duration to NaN
            average_duration = np.nan
            
            # Check if the bout exists in the behavior_event_dict
            if bout in block_data.behavior_event_dict:
                # Access the behaviors within the bout
                behaviors_dict = block_data.behavior_event_dict[bout]
                
                # Check if 'sniff' behavior exists for the bout
                if 'sniff' in behaviors_dict:
                    sniff_events = behaviors_dict['sniff']
                    
                    if sniff_events:
                        # Calculate the average duration of 'sniff' events
                        total_duration = sum(event['Duration'] for event in sniff_events)
                        num_events = len(sniff_events)
                        average_duration = total_duration / num_events
                    else:
                        # No 'sniff' events found
                        average_duration = np.nan
                else:
                    # 'sniff' behavior does not exist for the bout
                    average_duration = np.nan
            else:
                # Bout does not exist in the behavior_event_dict
                average_duration = np.nan
            
            # Assign the average duration to the subject's data dict
            subject_data_dict[f"{bout}_sniff_avg_duration"] = average_duration
        
        # Append the subject's data to the data_list
        data_list.append(subject_data_dict)
    
    # Convert the data_list into a DataFrame
    average_sniff_duration_df = pd.DataFrame(data_list)
    
    # Set the index to 'Subject'
    average_sniff_duration_df.set_index('Subject', inplace=True)
    
    return average_sniff_duration_df


def sp_extract_nth_behavior_mean_da_corrected(group_data, behavior, n=1, max_duration=5.0):
    """
    Extracts the mean DA during the n-th specified behavior event ('sniff' or 'chew') for each agent 
    across all blocks in the group, limiting the analysis to a maximum of max_duration seconds. 
    Returns the data in a DataFrame.

    Parameters:
    - group_data (object): The object containing multiple blocks of behavior event data.
    - behavior (str): The specific behavior to analyze (e.g., 'sniff' or 'chew').
    - n (int): The occurrence number of the behavior event to extract (default is 1).
    - max_duration (float): The maximum duration in seconds to limit DA analysis for each behavior event 
                             (default is 5.0 seconds).

    Returns:
    - pd.DataFrame: A DataFrame where each row represents a subject,
                    and each column represents an agent,
                    with the mean DA during the n-th specified behavior event,
                    limited to max_duration seconds.
    """
    # Validate the behavior parameter
    valid_behaviors = ['sniff', 'chew']
    if behavior not in valid_behaviors:
        raise ValueError(f"Invalid behavior '{behavior}'. Supported behaviors are: {valid_behaviors}")

    # Initialize an empty list to hold the data for each subject
    data_list = []

    # Loop through each block in the group data
    for block_data in group_data.blocks.values():
        # Initialize a dictionary to store the mean DA for this subject
        subject_data_dict = {'Subject': block_data.subject_name}

        for agent in block_data.behavior_event_dict:
            # Skip 'Subject Presence' as it's not an agent
            if agent == 'Subject Presence':
                continue

            # Check if the specified behavior exists for the agent
            if behavior not in block_data.behavior_event_dict[agent]:
                print(f"Agent '{agent}' does not have behavior '{behavior}'. Assigning NaN.")
                mean_da_behavior = np.nan
            else:
                # Get the list of specified behavior events for this agent
                behavior_events = block_data.behavior_event_dict[agent][behavior]

                # Ensure the requested n-th occurrence exists
                if len(behavior_events) >= n:
                    nth_behavior = behavior_events[n - 1]  # Get the n-th occurrence
                    event_start = nth_behavior['Start Time']
                    event_end = nth_behavior['End Time']

                    # Limit the analysis to max_duration seconds
                    analysis_end_time = min(event_start + max_duration, event_end)

                    # Get the z-score signal during the limited event window
                    zscore_indices = (block_data.timestamps >= event_start) & (block_data.timestamps <= analysis_end_time)
                    
                    if not np.any(zscore_indices):
                        print(f"No z-score data found in the time window around {event_start}s for block '{block_data.subject_name}', agent '{agent}', behavior '{behavior}'. Assigning NaN.")
                        mean_da_behavior = np.nan
                    else:
                        mean_da_behavior = np.mean(block_data.zscore[zscore_indices])  # Compute mean DA
                else:
                    print(f"Agent '{agent}' does not have {n} occurrences of behavior '{behavior}'. Assigning NaN.")
                    mean_da_behavior = np.nan  # If fewer than n occurrences, assign NaN

            # Store the mean DA for this agent in the subject's data dict
            subject_data_dict[agent] = mean_da_behavior

        # Append the subject's data to the data_list
        data_list.append(subject_data_dict)

    # Convert the data_list into a DataFrame
    behavior_mean_df = pd.DataFrame(data_list)

    # Set the index to 'Subject'
    behavior_mean_df.set_index('Subject', inplace=True)

    return behavior_mean_df


def sp_plot_first_investigation_vs_zscore_4s(self, bouts=None, behavior_name='investigation', legend_names=None, legend_loc='upper left'):
    """
    Plot the first occurrence of the specified behavior duration vs. mean Z-scored ΔF/F relative to the event onset 
    within a fixed 4-second window for all blocks, color-coded by bout type, with custom legend names and enhanced plot formatting.

    Parameters:
    - behavior_name (str): The name of the behavior to analyze (default is 'investigation').
    - bouts (list): A list of bout names to include in the analysis. If None, defaults to ['short_term', 'novel'].
    - legend_names (dict): A dictionary to map bout names to custom legend labels. If None, defaults to standard labels.
    - legend_loc (str): The location of the legend. Defaults to 'upper left'.
    """
    # Default bouts if none are provided
    if bouts is None:
        bouts = ['short_term', 'novel']

    # Default legend names if none are provided
    if legend_names is None:
        legend_names = {
            'short_term': 'Short-Term',
            'novel': 'Novel',
            'long_term': 'Long-Term',
            'nothing': 'Empty'
        }

    # Define the custom colors
    bout_colors = {
        'short_term': '#0045A6',
        'novel': '#E06928',
        'long_term': '#A839A4',
        'nothing': '#A839A4'
    }

    mean_zscored_dffs = []
    behavior_durations = []
    bout_names_collected = []

    # Iterate through each block
    for block_name, block_data in self.blocks.items():
        # Ensure zscore and timestamps are available
        if block_data.zscore is None or block_data.timestamps is None:
            print(f"Block {block_name} is missing zscore or timestamps data. Skipping.")
            continue

        # Iterate through each specified bout
        for bout in bouts:
            # Check if the bout exists in the behavior_event_dict
            if bout not in block_data.behavior_event_dict:
                print(f"Bout '{bout}' not found in block '{block_name}'. Skipping.")
                continue

            # Check if the specified behavior exists for the bout
            if behavior_name not in block_data.behavior_event_dict[bout]:
                print(f"Behavior '{behavior_name}' not found in bout '{bout}' for block '{block_name}'. Skipping.")
                continue

            investigation_events = block_data.behavior_event_dict[bout][behavior_name]

            if len(investigation_events) == 0:
                print(f"No '{behavior_name}' events found in bout '{bout}' for block '{block_name}'. Skipping.")
                continue

            # Get the first occurrence of the behavior
            first_event = investigation_events[0]
            event_start = first_event['Start Time']
            event_duration = first_event['Duration']

            # Define the fixed 4-second window starting from the event onset
            window_start = event_start
            window_end = event_start + 4.0

            # Ensure the window does not exceed the available timestamps
            if window_end > block_data.timestamps[-1]:
                print(f"Window end time {window_end}s exceeds the available data in block '{block_name}'. Adjusting window.")
                window_end = block_data.timestamps[-1]

            # Find indices within the window
            window_mask = (block_data.timestamps >= window_start) & (block_data.timestamps <= window_end)

            if not np.any(window_mask):
                print(f"No data found in the window [{window_start}, {window_end}]s for block '{block_name}', bout '{bout}'. Skipping.")
                continue

            # Extract zscore data within the window
            zscore_window = block_data.zscore[window_mask]

            # Calculate the mean z-scored ΔF/F over the window
            mean_zscore = np.mean(zscore_window)

            # Store the results
            mean_zscored_dffs.append(mean_zscore)
            behavior_durations.append(event_duration)
            bout_names_collected.append(bout)

    # Convert lists to numpy arrays
    mean_zscored_dffs = np.array(mean_zscored_dffs, dtype=np.float64)
    behavior_durations = np.array(behavior_durations, dtype=np.float64)
    bout_names_collected = np.array(bout_names_collected)

    # Check if there are valid data points
    if len(mean_zscored_dffs) == 0 or len(behavior_durations) == 0:
        print("No valid data points for correlation and plotting.")
        return

    # Calculate Pearson correlation
    r, p = stats.pearsonr(mean_zscored_dffs, behavior_durations)

    # Step 3: Plotting the scatter plot
    plt.figure(figsize=(16, 9))

    # Plot each bout separately
    for bout in bouts:
        # Create a mask for each bout
        mask = bout_names_collected == bout
        plt.scatter(
            mean_zscored_dffs[mask], 
            behavior_durations[mask],
            color=bout_colors.get(bout, '#000000'),  # Default to black if bout color not found
            label=legend_names.get(bout, bout), 
            alpha=1, 
            s=800, 
            edgecolor='black', 
            linewidth=6
        )

    # Adding the regression line with a consistent dashed style
    slope, intercept = np.polyfit(mean_zscored_dffs, behavior_durations, 1)
    regression_x = np.linspace(mean_zscored_dffs.min(), mean_zscored_dffs.max(), 100)
    regression_y = slope * regression_x + intercept
    plt.plot(regression_x, regression_y, color='black', linestyle='--', linewidth=4)

    # Add labels and title with larger font sizes
    plt.xlabel('Event Induced Z-scored ΔF/F', fontsize=44, labelpad=20)
    plt.ylabel(f'Bout Duration (s)', fontsize=44, labelpad=20)

    # Modify x-ticks and y-ticks to be larger
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)

    # Display Pearson correlation and number of events in the legend
    correlation_text = f'r = {r:.3f}\np = {p:.3f}\nn = {len(mean_zscored_dffs)} events'
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=bout_colors.get(bout, '#000000'), markersize=20, markeredgecolor='black') 
        for bout in bouts
    ]
    # Append an empty Line2D object for spacing in the legend
    custom_lines.append(Line2D([0], [0], color='none'))

    legend_labels = [legend_names.get(bout, bout) for bout in bouts] + [correlation_text]

    # Add a legend with bout names and correlation, placing it according to the provided location
    plt.legend(custom_lines, legend_labels, title='Agent', loc=legend_loc, fontsize=26, title_fontsize=28)

    # Remove top and right spines and increase the linewidth of the remaining spines
    sns.despine()
    ax = plt.gca()
    ax.spines['left'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)

    plt.tight_layout()
    plt.savefig('Scatter_First_Investigation_vs_Zscore.png', transparent=True, bbox_inches='tight', pad_inches=0.1)
    plt.show()



def sp_plot_average_investigation_vs_zscore_4s(self, bouts=None, behavior_name='investigation', legend_names=None):
    """
    Plot the average duration of the specified behavior vs. average mean Z-scored ΔF/F relative to the event onset 
    within a fixed 4-second window for all blocks, color-coded by bout type, with custom legend names and enhanced plot formatting.

    Parameters:
    - behavior_name (str): The name of the behavior to analyze (default is 'investigation').
    - bouts (list): A list of bout names to include in the analysis. If None, defaults to ['short_term', 'novel'].
    - legend_names (dict): A dictionary to map bout names to custom legend labels. If None, defaults to standard labels.
    """
    # Default bouts if none are provided
    if bouts is None:
        bouts = ['short_term', 'novel']

    # Default legend names if none are provided
    if legend_names is None:
        legend_names = {
            'short_term': 'Short-Term',
            'novel': 'Novel Object',
            'long_term': 'Long-Term',
            'nothing': 'Empty'
        }

    # Define the custom colors
    bout_colors = {
        'short_term': '#00B7D7',
        'novel': '#E06928',
        'long_term': '#0045A6',
        'nothing': '#A839A4'
    }

    # Initialize dictionaries to hold durations and z-scores per bout
    bout_durations = {bout: [] for bout in bouts}
    bout_zscores = {bout: [] for bout in bouts}

    # Iterate through each block
    for block_name, block_data in self.blocks.items():
        # Ensure zscore and timestamps are available
        if block_data.zscore is None or block_data.timestamps is None:
            print(f"Block {block_name} is missing zscore or timestamps data. Skipping.")
            continue

        # Iterate through each specified bout
        for bout in bouts:
            # Check if the bout exists in the behavior_event_dict
            if bout not in block_data.behavior_event_dict:
                print(f"Bout '{bout}' not found in block '{block_name}'. Skipping.")
                continue

            # Check if the specified behavior exists for the bout
            if behavior_name not in block_data.behavior_event_dict[bout]:
                print(f"Behavior '{behavior_name}' not found in bout '{bout}' for block '{block_name}'. Skipping.")
                continue

            investigation_events = block_data.behavior_event_dict[bout][behavior_name]

            if len(investigation_events) == 0:
                print(f"No '{behavior_name}' events found in bout '{bout}' for block '{block_name}'. Skipping.")
                continue

            # Get the first occurrence of the behavior
            first_event = investigation_events[0]
            event_start = first_event['Start Time']
            event_duration = first_event['Duration']

            # Define the fixed 4-second window starting from the event onset
            window_start = event_start
            window_end = event_start + 4.0

            # Ensure the window does not exceed the available timestamps
            if window_end > block_data.timestamps[-1]:
                print(f"Window end time {window_end}s exceeds the available data in block '{block_name}'. Adjusting window.")
                window_end = block_data.timestamps[-1]

            # Find indices within the window
            window_mask = (block_data.timestamps >= window_start) & (block_data.timestamps <= window_end)

            if not np.any(window_mask):
                print(f"No data found in the window [{window_start}, {window_end}]s for block '{block_name}', bout '{bout}'. Skipping.")
                continue

            # Extract zscore data within the window
            zscore_window = block_data.zscore[window_mask]

            # Calculate the mean z-scored ΔF/F over the window
            mean_zscore = np.mean(zscore_window)

            # Store the results
            bout_durations[bout].append(event_duration)
            bout_zscores[bout].append(mean_zscore)

    # Compute average durations and z-scores per bout
    avg_durations = {}
    avg_zscores = {}
    n_events = {}

    for bout in bouts:
        durations = bout_durations[bout]
        zscores = bout_zscores[bout]
        if len(durations) > 0:
            avg_durations[bout] = np.mean(durations)
            avg_zscores[bout] = np.mean(zscores)
            n_events[bout] = len(durations)
        else:
            avg_durations[bout] = np.nan
            avg_zscores[bout] = np.nan
            n_events[bout] = 0
            print(f"No valid '{behavior_name}' events for bout '{bout}' across all blocks.")

    # Prepare data for plotting
    plot_bouts = [bout for bout in bouts if not np.isnan(avg_durations[bout]) and not np.isnan(avg_zscores[bout])]
    plot_durations = [avg_durations[bout] for bout in plot_bouts]
    plot_zscores = [avg_zscores[bout] for bout in plot_bouts]
    plot_n = [n_events[bout] for bout in plot_bouts]

    # Check if there are valid data points
    if len(plot_bouts) < 2:
        print("Not enough valid data points for correlation and plotting.")
        return

    # Calculate Pearson correlation
    r, p = stats.pearsonr(plot_zscores, plot_durations)
    r_squared = r ** 2  # Calculate r-squared

    # Step 3: Plotting the scatter plot
    plt.figure(figsize=(16, 9))

    # Plot each bout separately
    for bout, duration, zscore in zip(plot_bouts, plot_durations, plot_zscores):
        plt.scatter(
            zscore, 
            duration,
            color=bout_colors.get(bout, '#000000'),  # Default to black if bout color not found
            label=legend_names.get(bout, bout), 
            alpha=1, 
            s=600, 
            edgecolor='black', 
            linewidth=2
        )

    # Adding the regression line with a consistent dashed style
    slope, intercept = np.polyfit(plot_zscores, plot_durations, 1)
    regression_x = np.linspace(min(plot_zscores), max(plot_zscores), 100)
    regression_y = slope * regression_x + intercept
    plt.plot(regression_x, regression_y, color='black', linestyle='--', linewidth=4, label='Regression Line')

    # Add labels and title with larger font sizes
    plt.xlabel('Mean Z-scored ΔF/F', fontsize=40, labelpad=20)
    plt.ylabel(f'Average {behavior_name.capitalize()} Duration (s)', fontsize=40, labelpad=20)
    plt.title(f'Average {behavior_name.capitalize()} Duration vs. Mean Z-scored ΔF/F', fontsize=45, pad=30)

    # Modify x-ticks and y-ticks to be larger
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)

    # Display Pearson correlation, r², and number of events in the legend
    correlation_text = f'r = {r:.3f}\nr² = {r_squared:.3f}\nn = {len(plot_bouts)} events'
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=bout_colors.get(bout, '#000000'), markersize=20, markeredgecolor='black') 
        for bout in plot_bouts
    ]
    custom_lines.append(Line2D([0], [0], linestyle='--', color='black', linewidth=4))
    legend_labels = [legend_names.get(bout, bout) for bout in plot_bouts] + [correlation_text]

    # Add a legend with bout names and correlation inside the plot at the top left
    plt.legend(custom_lines, legend_labels, title='Bout', loc='upper left', fontsize=24, title_fontsize=28)

    # Remove top and right spines and increase the linewidth of the remaining spines
    sns.despine()
    ax = plt.gca()
    ax.spines['left'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)
    ax.spines['top'].set_linewidth(5)

    plt.tight_layout()
    plt.savefig('Scatter_Average_Investigation_vs_Zscore.png', transparent=True, bbox_inches='tight', pad_inches=0.1)
    plt.show()