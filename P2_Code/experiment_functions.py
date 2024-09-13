import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# These functions are used for all experiments

def plot_y_across_bouts(df, title='Mean Across Bouts', ylabel='Mean Value'):
    """
    Plots the mean values during investigations or other events across bouts with error bars for SEM
    and individual subject lines connecting the bouts.

    Parameters:
    - df (DataFrame): A DataFrame where rows are subjects, and bouts are bouts.
                      Values should represent the mean values (e.g., mean DA, investigation times)
                      for each subject and bout.
    - title (str): The title for the plot.
    - ylabel (str): The label for the y-axis.
    """

    # Calculate the mean and SEM for each bout (across all subjects)
    mean_values = df.mean()
    sem_values = df.sem()

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot the bar plot with error bars (mean and SEM)
    bars = ax.bar(df.columns, mean_values, yerr=sem_values, capsize=5, color='skyblue', edgecolor='black', label='Mean')

    # Plot each individual's mean value and connect the dots between bars
    colors = plt.cm.get_cmap('tab10', len(df))  # Use a colormap with enough unique colors for each subject

    for i, subject in enumerate(df.index):
        ax.plot(df.columns, df.loc[subject], marker='o', linestyle='-', color=colors(i), alpha=0.7, label=subject)

    # Add labels, title, and format
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Bouts', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Set x-ticks to match the bout labels
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns, fontsize=12)

    # Automatically set the y-limits based on the data range
    all_values = np.concatenate([df.values.flatten(), mean_values.values.flatten()])
    min_val = np.nanmin(all_values)
    max_val = np.nanmax(all_values)

    # Set lower y-limit to 0 if all values are above 0, otherwise set to the minimum value
    lower_ylim = 0 if min_val > 0 else min_val * 1.1
    upper_ylim = max_val * 1.1  # Adding a bit of space above the highest value
    
    ax.set_ylim(lower_ylim, upper_ylim)

    # Add the legend on the right side, outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Subjects")

    # Display the plot
    plt.tight_layout()
    plt.show()


def extract_average_behavior_durations(group_data, bouts, behavior='Investigation'):
    """
    Extracts the mean durations for the specified behavior (e.g., 'Investigation') 
    for each subject and bout, and returns the data in a DataFrame.

    Parameters:
    group_data (object): The object containing bout data for each subject.
    bouts (list): A list of bout names to process.
    behavior (str): The behavior of interest to calculate mean durations for (default is 'Investigation').

    Returns:
    pd.DataFrame: A DataFrame where each row represents a subject, 
                  and each column represents the mean duration of the specified behavior for a specific bout.
    """
    # Initialize an empty list to hold the data for each subject
    data_list = []

    # Populate the data_list from the group_data.blocks
    for block_data in group_data.blocks.values():
        if hasattr(block_data, 'bout_dict') and block_data.bout_dict:  # Ensure bout_dict exists and is populated
            # Use the subject name from the TDTData object
            block_data_dict = {'Subject': block_data.subject_name}

            for bout in bouts:  # Only process bouts in the given list of bouts
                if bout in block_data.bout_dict and behavior in block_data.bout_dict[bout]:
                    # Collect the mean duration for the specified behavior for this subject and bout
                    total_duration = np.nanmean([event['Total Duration'] for event in block_data.bout_dict[bout][behavior]])
                    block_data_dict[bout] = total_duration
                else:
                    block_data_dict[bout] = np.nan  # If no data, assign NaN

            # Append the block's data to the data_list
            data_list.append(block_data_dict)

    # Convert the data_list into a DataFrame
    behavior_duration_df = pd.DataFrame(data_list)

    # Set the index to 'Subject'
    behavior_duration_df.set_index('Subject', inplace=True)

    return behavior_duration_df


def extract_total_behavior_durations(group_data, bouts, behavior='Investigation'):
    """
    Extracts the total durations for the specified behavior (e.g., 'Investigation') 
    for each subject and bout, and returns a DataFrame.

    Parameters:
    group_data (object): The object containing bout data for each subject.
    bouts (list): A list of bout names to process.
    behavior (str): The behavior of interest to calculate total durations for (default is 'Investigation').

    Returns:
    pd.DataFrame: A DataFrame where each row represents a subject, 
                  and each column represents the total duration of the specified behavior for a specific bout.
    """
    # Initialize an empty list to hold the data for each subject
    data_list = []

    # Populate the data_list from the group_data.blocks
    for block_data in group_data.blocks.values():
        if hasattr(block_data, 'bout_dict') and block_data.bout_dict:  # Ensure bout_dict exists and is populated
            # Use the subject name from the TDTData object
            block_data_dict = {'Subject': block_data.subject_name}

            for bout in bouts:  # Only process bouts in the given list of bouts
                if bout in block_data.bout_dict and behavior in block_data.bout_dict[bout]:
                    # Collect the total duration for the specified behavior for this subject and bout
                    total_duration = np.nansum([event['Total Duration'] for event in block_data.bout_dict[bout][behavior]])
                    block_data_dict[bout] = total_duration
                else:
                    block_data_dict[bout] = np.nan  # If no data, assign NaN

            # Append the block's data to the data_list
            data_list.append(block_data_dict)

    # Convert the data_list into a DataFrame
    behavior_duration_df = pd.DataFrame(data_list)

    # Set the index to 'Subject'
    behavior_duration_df.set_index('Subject', inplace=True)

    return behavior_duration_df


def extract_nth_behavior_mean_da(group_data, bouts, behavior='Investigation', n=1):
        """
        Extracts the mean DA during the n-th occurrence of the specified behavior (e.g., 'Investigation') 
        for each subject and bout, and returns the data in a DataFrame.

        Parameters:
        group_data (object): The object containing bout data for each subject.
        bouts (list): A list of bout names to process.
        behavior (str): The behavior of interest to extract mean DA for the n-th occurrence (default is 'Investigation').
        n (int): The occurrence number of the behavior to extract (default is the value of self.n_behavior_occurrence).

        Returns:
        pd.DataFrame: A DataFrame where each row represents a subject, 
                      and each column represents the mean DA during the n-th occurrence of the specified behavior 
                      for a specific bout.
        """

        # Initialize an empty list to hold the data for each subject
        data_list = []

        # Populate the data_list from the group_data.blocks
        for block_data in group_data.blocks.values():
            if hasattr(block_data, 'bout_dict') and block_data.bout_dict:  # Ensure bout_dict exists and is populated
                # Use the subject name from the TDTData object
                block_data_dict = {'Subject': block_data.subject_name}

                for bout in bouts:  # Only process bouts in the given list of bouts
                    if bout in block_data.bout_dict and behavior in block_data.bout_dict[bout]:
                        # Ensure the requested n-th occurrence exists
                        if len(block_data.bout_dict[bout][behavior]) >= n:
                            nth_behavior = block_data.bout_dict[bout][behavior][n - 1]  # Get the n-th occurrence
                            if 'Mean zscore' in nth_behavior:
                                mean_da_nth_behavior = nth_behavior['Mean zscore']
                            else:
                                mean_da_nth_behavior = np.nan  # If no z-score data, assign NaN
                        else:
                            mean_da_nth_behavior = np.nan  # If fewer than n occurrences, assign NaN

                        block_data_dict[bout] = mean_da_nth_behavior
                    else:
                        block_data_dict[bout] = np.nan  # If no data, assign NaN

                # Append the block's data to the data_list
                data_list.append(block_data_dict)

        # Convert the data_list into a DataFrame
        behavior_mean_df = pd.DataFrame(data_list)

        # Set the index to 'Subject'
        behavior_mean_df.set_index('Subject', inplace=True)

        return behavior_mean_df


def plot_approach_vs_aggression(group_data, min_duration=0):
    """
    Separates 'Approach' behaviors that are immediately followed by 'Aggression' (<1s) 
    from those that are not, and plots the mean DA (z-scored ΔF/F) as bar graphs.
    
    Parameters:
    group_data (object): The object containing bout data for each subject.
    min_duration (float): The minimum duration of 'Approach' behaviors to include in the plot.
    """
    approach_followed_by_aggression = []
    approach_not_followed_by_aggression = []
    subject_names_aggression = []
    subject_names_no_aggression = []

    # Loop through each block in group_data.blocks
    for block_name, block_data in group_data.blocks.items():
        if block_data.bout_dict:  # Ensure bout_dict is populated
            for bout, behavior_data in block_data.bout_dict.items():
                if 'Approach' in behavior_data:
                    # Loop through all 'Approach' events in this bout
                    for i, approach_event in enumerate(behavior_data['Approach']):
                        duration = approach_event['Total Duration']
                        if duration < min_duration:  # Only include events longer than min_duration
                            continue

                        approach_offset = approach_event['End Time']

                        # Check if there is an 'Aggression' behavior immediately after the 'Approach'
                        followed_by_aggression = False
                        if 'Aggression' in behavior_data:
                            for aggression_event in behavior_data['Aggression']:
                                if aggression_event['Start Time'] - approach_offset < 1:  # Check if it's within 1 second
                                    followed_by_aggression = True
                                    break

                        # Separate into the appropriate list
                        if followed_by_aggression:
                            approach_followed_by_aggression.append(approach_event['Mean zscore'])
                            subject_names_aggression.append(block_name)
                        else:
                            approach_not_followed_by_aggression.append(approach_event['Mean zscore'])
                            subject_names_no_aggression.append(block_name)

    # Ensure lists are only flat arrays of valid numbers
    approach_followed_by_aggression = [x for x in approach_followed_by_aggression if isinstance(x, (float, int))]
    approach_not_followed_by_aggression = [x for x in approach_not_followed_by_aggression if isinstance(x, (float, int))]

    # Convert lists to numpy arrays for calculations
    approach_followed_by_aggression = np.array(approach_followed_by_aggression, dtype=np.float64)
    approach_not_followed_by_aggression = np.array(approach_not_followed_by_aggression, dtype=np.float64)

    # Calculate the mean and SEM for both categories
    mean_with_aggression = np.nanmean(approach_followed_by_aggression) if len(approach_followed_by_aggression) > 0 else np.nan
    sem_with_aggression = np.nanstd(approach_followed_by_aggression) / np.sqrt(len(approach_followed_by_aggression)) if len(approach_followed_by_aggression) > 0 else np.nan

    mean_without_aggression = np.nanmean(approach_not_followed_by_aggression) if len(approach_not_followed_by_aggression) > 0 else np.nan
    sem_without_aggression = np.nanstd(approach_not_followed_by_aggression) / np.sqrt(len(approach_not_followed_by_aggression)) if len(approach_not_followed_by_aggression) > 0 else np.nan

    # Bar plot
    categories = ['Approach w/ Aggression', 'Approach w/o Aggression']
    means = np.array([mean_with_aggression, mean_without_aggression])
    sems = [sem_with_aggression, sem_without_aggression]

    plt.figure(figsize=(8, 6))
    
    # Create bar graph with error bars (mean ± SEM)
    plt.bar(categories, means, yerr=sems, capsize=5, color=['lightcoral', 'skyblue'], edgecolor='black')

    # Add labels and title
    plt.ylabel('Mean Z-scored ΔF/F')
    plt.title('Mean DA for Approach Behaviors With and Without Aggression')

    plt.tight_layout()
    plt.show()





