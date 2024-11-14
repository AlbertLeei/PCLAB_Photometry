import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def sp_extract_intruder_events(behavior_csv_path, cup_assignments, subject_name):
    """
    Extracts behaviors from the behavior CSV and maps them to the correct agent in each cup based on the 
    cup assignments DataFrame. Keeps 'sniff' and 'chew' behaviors separate.
    
    Parameters:
        behavior_csv_path (str): Path to the behavior CSV file.
        cup_assignments (DataFrame): DataFrame with cup assignments for each subject.
        subject_name (str): Name of the subject to match with cup assignments.
    
    Returns:
        dict: A dictionary with events organized by agents and behaviors.
    """
    # Load behavior CSV
    behavior_data = pd.read_csv(behavior_csv_path)

    # Make behavior names lowercase for consistency
    behavior_data['Behavior'] = behavior_data['Behavior'].str.lower()

    # Initialize a dictionary to store events based on agents (bouts)
    behavior_event_dict = {}

    # Loop through each row in the behavior data
    for index, row in behavior_data.iterrows():
        behavior_type = row['Behavior']

        # Check if the behavior involves a cup (e.g., 'sniff cup 3' or 'chew cup 4')
        if 'cup' in behavior_type:
            try:
                # Extract the cup number
                cup_number = int(behavior_type.split('cup')[-1].strip())

                # Ensure the subject exists in the cup assignments
                if subject_name not in cup_assignments['Subject'].values:
                    print(f"Skipping row: subject {subject_name} not found in cup assignments")
                    continue

                # Ensure the cup number is within valid bounds (1 to 4)
                if cup_number < 1 or cup_number > 4:
                    print(f"Skipping row: invalid cup number {cup_number}")
                    continue

                # Find the corresponding agent (bout) from the cup assignment CSV based on subject and cup number
                agent = cup_assignments.loc[cup_assignments['Subject'] == subject_name, f'Cup {cup_number}'].iloc[0]

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

    return behavior_event_dict



def process_all_behavior_files(all_csvs_folder, cup_assignment_excel_path):
    # Read the cup assignment sheet from the Excel file
    cup_assignments = pd.read_excel(cup_assignment_excel_path, sheet_name='Sheet1')
    
    # Dictionary to hold behavior data for each subject
    subject_behavior_dict = {}

    # Iterate over all CSV files in the folder
    for filename in os.listdir(all_csvs_folder):
        if filename.endswith('.csv'):
            # Define the path to the current behavior CSV
            behavior_csv_path = os.path.join(all_csvs_folder, filename)
            
            # Extract the subject name from the filename using the hyphen separator
            subject_name = filename.split('-')[0]  # Adjust this if the naming format changes
            
            # Apply the function to extract behavior events for the current subject
            behavior_event_dict = sp_extract_intruder_events(behavior_csv_path, cup_assignments, subject_name)
            
            # Store the result in the dictionary
            subject_behavior_dict[subject_name] = behavior_event_dict
    
    return subject_behavior_dict


# Plotting
def calculate_average_sniff_duration_per_bout(behavior_dict, bouts=['novel', 'long_term', 'short_term', 'empty']):
    """
    Calculates the average duration of the 'sniff' behavior for each bout across all subjects.

    Parameters:
        behavior_dict (dict): Dictionary where each key is a subject and each value is a behavior event dictionary.
        bouts (list): List of bout names to calculate average sniff durations for.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a subject,
                      and each column represents a bout,
                      containing the average duration of 'sniff' behavior.
    """
    # Mapping to rename 'nothing' to 'empty' in the DataFrame
    bout_mapping = {'empty': 'nothing', 'novel': 'novel', 'long_term': 'long_term', 'short_term': 'short_term'}

    # Initialize an empty list to hold the data for each subject
    data_list = []

    # Loop through each subject in the behavior dictionary
    for subject, agent_data in behavior_dict.items():
        # Use the subject name as the row identifier
        subject_data_dict = {'Subject': subject}

        # Loop through each specified bout
        for bout in bouts:
            # Initialize average duration to NaN
            average_duration = np.nan

            # Map the bout name to the actual key in the behavior dictionary
            actual_bout = bout_mapping[bout]

            # Check if the bout exists in the subject's agent data
            if actual_bout in agent_data:
                # Access the behaviors within the bout
                behaviors_dict = agent_data[actual_bout]

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
                # Bout does not exist in the subject's agent data
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


def plot_y_across_bouts_colors(df, title='Mean Across Bouts', ylabel='Mean Value', custom_xtick_labels=None, custom_xtick_colors=None, ylim=None, 
                               bar_color='#00B7D7', yticks_increment=None, xlabel='intruder', figsize=(12, 7), pad_inches=0.1):
    """
    Plots the mean values across bouts with error bars for SEM and individual subject lines connecting the bouts.

    Parameters:
    - df (DataFrame): A DataFrame where rows are subjects, and columns are bouts.
                      Values should represent measurements (e.g., investigation times) for each subject and bout.
    - title (str): The title for the plot.
    - ylabel (str): The label for the y-axis.
    - custom_xtick_labels (list): A list of custom x-tick labels. If not provided, defaults to the column names.
    - custom_xtick_colors (list): A list of colors for the x-tick labels. Must match `custom_xtick_labels` length.
    - ylim (tuple): A tuple (min, max) to set the y-axis limits. If None, limits are set automatically based on data.
    - bar_color (str or list): A color or list of colors to use for the bars. Defaults to '#00B7D7'.
    - yticks_increment (float): Increment amount for the y-axis ticks.
    - xlabel (str): The label for the x-axis.
    """

    # Calculate the mean and SEM for each bout (across all subjects)
    mean_values = df.mean()
    sem_values = df.sem()

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the bar plot with error bars (mean and SEM)
    bars = ax.bar(
        df.columns, 
        mean_values, 
        yerr=sem_values, 
        capsize=6,  
        color=bar_color,  
        edgecolor='black', 
        linewidth=2,  
        width=0.7,
        error_kw=dict(elinewidth=2, capthick=2, zorder=5) 
    )

    # Plot individual data points and connecting lines for each subject
    for i, subject in enumerate(df.index):
        # Plot connecting line for each subject across bouts
        ax.plot(df.columns, df.loc[subject], linestyle='-', color='gray', alpha=0.5, linewidth=2, zorder=1)

        # Plot individual data points in gray
        ax.scatter(df.columns, df.loc[subject], color='gray', s=100, alpha=0.8, edgecolor='black', linewidth=1, zorder=2)

    # Set axis labels and title
    ax.set_ylabel(ylabel, fontsize=24)
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_title(title, fontsize=24)

    # Set x-ticks and x-tick labels
    if custom_xtick_labels:
        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_xticklabels(custom_xtick_labels, fontsize=18)
        if custom_xtick_colors:
            for tick, color in zip(ax.get_xticklabels(), custom_xtick_colors):
                tick.set_color(color)
    else:
        ax.set_xticklabels(df.columns, fontsize=18)

    # Set y-tick label size
    ax.tick_params(axis='y', labelsize=18)

    # Adjust y-limits based on the data range if ylim is not provided
    if ylim is None:
        all_values = np.concatenate([df.values.flatten(), mean_values.values.flatten()])
        min_val, max_val = np.nanmin(all_values), np.nanmax(all_values)
        lower_ylim = 0 if min_val > 0 else min_val * 1.1
        upper_ylim = max_val * 1.1
        ax.set_ylim(lower_ylim, upper_ylim)
        if lower_ylim < 0:
            ax.axhline(0, color='black', linestyle='--', linewidth=1)
    else:
        ax.set_ylim(ylim)
        if ylim[0] < 0:
            ax.axhline(0, color='black', linestyle='--', linewidth=1)

    # Set y-ticks based on yticks_increment
    if yticks_increment:
        y_min, y_max = ax.get_ylim()
        y_ticks = np.arange(np.floor(y_min), np.ceil(y_max) + yticks_increment, yticks_increment)
        ax.set_yticks(y_ticks)

    # Remove the right and top spines for a cleaner look
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()

