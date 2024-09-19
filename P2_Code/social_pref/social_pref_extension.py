import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def sp_extract_intruder_events(self, behavior_csv_path, cup_assignment_csv_path):
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



            


# Regular functions
def sp_extract_total_investigation_time(group_data, bouts=['Novel', 'Long_Term', 'Short_Term', 'Nothing']):
    """
    Extracts the total investigation time for each bout across all blocks in the group.
    """
    # Initialize an empty list to hold the data for each subject
    data_list = []

    # Loop through each block in the group data
    for block_data in group_data.blocks.values():
        # Use the subject name as the row identifier
        subject_data_dict = {'Subject': block_data.subject_name}
        
        # Initialize total investigation times for each bout to zero
        for bout in bouts:
            subject_data_dict[bout] = 0.0

        # Loop over agents (which are the bouts)
        for agent in block_data.behavior_event_dict:
            if agent in bouts:
                # Sum up the durations of 'investigation' events
                total_investigation_time = sum(
                    event['Duration'] for event in block_data.behavior_event_dict[agent]['investigation']
                )
                subject_data_dict[agent] = total_investigation_time

        # Append the subject's data to the data_list
        data_list.append(subject_data_dict)

    # Convert the data_list into a DataFrame
    investigation_time_df = pd.DataFrame(data_list)

    # Set the index to 'Subject'
    investigation_time_df.set_index('Subject', inplace=True)

    return investigation_time_df



def sp_extract_nth_investigation_mean_da_corrected(group_data, n=1, max_duration=5.0):
    """
    Extracts the mean DA during the n-th 'investigation' event for each agent across all blocks in the group,
    limiting the analysis to a maximum of max_duration seconds. Returns the data in a DataFrame.

    Parameters:
    group_data (object): The object containing multiple blocks of behavior event data.
    n (int): The occurrence number of the 'investigation' event to extract (default is 1).
    max_duration (float): The maximum duration in seconds to limit DA analysis for each behavior event (default is 5.0 seconds).

    Returns:
    pd.DataFrame: A DataFrame where each row represents a subject,
                  and each column represents an agent,
                  with the mean DA during the n-th 'investigation' event,
                  limited to max_duration seconds.
    """
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

            # Get the list of 'investigation' events for this agent
            investigation_events = block_data.behavior_event_dict[agent]['investigation']

            # Ensure the requested n-th occurrence exists
            if len(investigation_events) >= n:
                nth_behavior = investigation_events[n - 1]  # Get the n-th occurrence
                event_start = nth_behavior['Start Time']
                event_end = nth_behavior['End Time']

                # Limit the analysis to max_duration seconds
                analysis_end_time = min(event_start + max_duration, event_end)

                # Get the z-score signal during the limited event window
                zscore_indices = (block_data.timestamps >= event_start) & (block_data.timestamps <= analysis_end_time)
                mean_da_nth_behavior = np.mean(block_data.zscore[zscore_indices])  # Compute mean DA
            else:
                mean_da_nth_behavior = np.nan  # If fewer than n occurrences, assign NaN

            # Store the mean DA for this agent in the subject's data dict
            subject_data_dict[agent] = mean_da_nth_behavior

        # Append the subject's data to the data_list
        data_list.append(subject_data_dict)

    # Convert the data_list into a DataFrame
    behavior_mean_df = pd.DataFrame(data_list)

    # Set the index to 'Subject'
    behavior_mean_df.set_index('Subject', inplace=True)

    return behavior_mean_df
