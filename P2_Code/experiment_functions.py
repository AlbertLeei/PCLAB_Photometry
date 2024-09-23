import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress
from scipy.optimize import curve_fit



# These functions are used for all experiments
custom_palette = ['#FF9F1C', '#0077B6', '#D1E8E2', '#55A630', '#E07A5F', '#FFADAD', '#2C2C54', '#792910']

def plot_y_across_bouts(df, title='Mean Across Bouts', ylabel='Mean Value', colors=custom_palette, custom_xtick_labels=None, ylim=None, bar_color='#00B7D7'):
    """
    Plots the mean values during investigations or other events across bouts with error bars for SEM
    and individual subject lines connecting the bouts.

    Parameters:
    - df (DataFrame): A DataFrame where rows are subjects, and bouts are columns.
                      Values should represent the mean values (e.g., mean DA, investigation times)
                      for each subject and bout.
    - title (str): The title for the plot.
    - ylabel (str): The label for the y-axis.
    - colors (list): A list of colors to use for individual subject lines and markers.
    - custom_xtick_labels (list): A list of custom x-tick labels. If not provided, defaults to the column names.
    - ylim (tuple): A tuple (min, max) to set the y-axis limits. If None, the limits are set automatically based on the data.
    - bar_colors (str or list): A color or list of colors to use for the bars. Defaults to '#00B7D7'.
    """

    # Calculate the mean and SEM for each bout (across all subjects)
    mean_values = df.mean()
    sem_values = df.sem()

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot the bar plot with error bars (mean and SEM) without adding it to the legend
    bars = ax.bar(
        df.columns, 
        mean_values, 
        yerr=sem_values, 
        capsize=6,  # Increase capsize for larger error bars
        color=bar_color,  # Custom bar colors
        edgecolor='black', 
        linewidth=2,  # Thicker and darker bar outlines
        width=0.6,
        error_kw=dict(elinewidth=2.5, capthick=2.5, zorder=5)  # Thicker error bars and make them appear above circles
    )

    # Plot the lines first with a reduced linewidth
    for i, subject in enumerate(df.index):
        color_idx = i % len(colors)  # Use modulo to loop over the colors if there are more subjects than colors
        ax.plot(df.columns, df.loc[subject], linestyle='-', color=colors[color_idx], alpha=0.5, linewidth=2, zorder=1)

    # Then plot the unfilled circle markers with larger size
    for i, subject in enumerate(df.index):
        color_idx = i % len(colors)  # Use modulo to loop over the colors if there are more subjects than colors
        ax.scatter(df.columns, df.loc[subject], facecolors='none', edgecolors=colors[color_idx], s=120, alpha=0.6, linewidth=2, label=subject, zorder=2)

    # Add labels, title, and format
    ax.set_ylabel(ylabel, fontsize=24)
    ax.set_xlabel('Agent', fontsize=24, labelpad=12)
    ax.set_title(title, fontsize=20)
    
    # Set x-ticks to match the bout labels
    ax.set_xticks(np.arange(len(df.columns)))

    # Use custom x-tick labels if provided, otherwise use the column names
    if custom_xtick_labels is not None:
        ax.set_xticklabels(custom_xtick_labels, fontsize=20)
    else:
        ax.set_xticklabels(df.columns, fontsize=20)

    # Increase the font size of y-axis tick numbers
    ax.tick_params(axis='y', labelsize=22)  # Increase y-axis number size
    ax.tick_params(axis='x', labelsize=20)  # Optional: also increase x-axis number size

    # Automatically set the y-limits based on the data range if ylim is not provided
    if ylim is None:
        # Collect all values to determine the y-limits
        all_values = np.concatenate([df.values.flatten(), mean_values.values.flatten()])
        min_val = np.nanmin(all_values)
        max_val = np.nanmax(all_values)

        # Set lower y-limit to 0 if all values are above 0, otherwise set to the minimum value
        lower_ylim = 0 if min_val > 0 else min_val * 1.1
        upper_ylim = max_val * 1.1  # Adding a bit of space above the highest value
        
        ax.set_ylim(lower_ylim, upper_ylim)
    else:
        # If ylim is provided, set the limits to the specified values
        ax.set_ylim(ylim)
        if(ylim[0] < 0):
            ax.axhline(0, color='gray', linestyle='--', linewidth=2, zorder=3)

    # Add the legend on the right side, outside the plot (but only for individual subjects)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Subjects")

    # Add a dashed line at y=0 if it exists in the y-limits and auto-adjusted
    if ylim is None and lower_ylim < 0:
        ax.axhline(0, color='gray', linestyle='--', linewidth=2, zorder=3)

    # Remove the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_y_across_bouts_gray(df, title='Mean Across Bouts', ylabel='Mean Value', custom_xtick_labels=None, ylim=None, bar_color='#00B7D7', yticks_increment=None, xlabel = 'Agent'):
    """
    Plots the mean values during investigations or other events across bouts with error bars for SEM
    and individual subject lines connecting the bouts. All subjects are plotted in gray.

    Parameters:
    - df (DataFrame): A DataFrame where rows are subjects, and bouts are columns.
                      Values should represent the mean values (e.g., mean DA, investigation times)
                      for each subject and bout.
    - title (str): The title for the plot.
    - ylabel (str): The label for the y-axis.
    - custom_xtick_labels (list): A list of custom x-tick labels. If not provided, defaults to the column names.
    - ylim (tuple): A tuple (min, max) to set the y-axis limits. If None, the limits are set automatically based on the data.
    - bar_color (str): The color to use for the bars (default is cyan).
    - yticks_increment (float): Increment amount for the y-axis ticks.
    """

    # Calculate the mean and SEM for each bout (across all subjects)
    mean_values = df.mean()
    sem_values = df.sem()

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))  #10,7

    # Plot the bar plot with error bars (mean and SEM) without adding it to the legend
    bars = ax.bar(
        df.columns, 
        mean_values, 
        yerr=sem_values, 
        capsize=6,  # Increase capsize for larger error bars
        color=bar_color,  # Customizable bar color
        edgecolor='black', 
        linewidth=2,  # Thicker and darker bar outlines
        width=0.6,
        error_kw=dict(elinewidth=2.5, capthick=2.5, zorder=5)  # Thicker error bars and make them appear above circles
    )

    # Plot all subject lines and markers in gray
    for i, subject in enumerate(df.index):
        ax.plot(df.columns, df.loc[subject], linestyle='-', color='gray', alpha=0.5, linewidth=2, zorder=1)

    # Plot unfilled circle markers with larger size, in gray
    for i, subject in enumerate(df.index):
        ax.scatter(df.columns, df.loc[subject], facecolors='none', edgecolors='gray', s=120, alpha=0.6, linewidth=2, zorder=2)

    # Add labels, title, and format
    ax.set_ylabel(ylabel, fontsize=30)  # Larger y-axis label
    ax.set_xlabel(xlabel, fontsize=30, labelpad=12)
    ax.set_title(title, fontsize=16)

    # Set x-ticks to match the bout labels
    ax.set_xticks(np.arange(len(df.columns)))

    # Use custom x-tick labels if provided, otherwise use the column names
    if custom_xtick_labels is not None:
        ax.set_xticklabels(custom_xtick_labels, fontsize=20)
    else:
        ax.set_xticklabels(df.columns, fontsize=20)

    # Increase the font size of y-axis tick numbers
    ax.tick_params(axis='y', labelsize=30)  # Increase y-axis number size
    ax.tick_params(axis='x', labelsize=30)  # Optional: also increase x-axis number size

    # Automatically set the y-limits based on the data range if ylim is not provided
    if ylim is None:
        # Collect all values to determine the y-limits
        all_values = np.concatenate([df.values.flatten(), mean_values.values.flatten()])
        min_val = np.nanmin(all_values)
        max_val = np.nanmax(all_values)

        # Set lower y-limit to 0 if all values are above 0, otherwise set to the minimum value
        lower_ylim = 0 if min_val > 0 else min_val * 1.1
        upper_ylim = max_val * 1.1  # Adding a bit of space above the highest value
        
        ax.set_ylim(lower_ylim, upper_ylim)
    else:
        # If ylim is provided, set the limits to the specified values
        ax.set_ylim(ylim)
        if ylim[0] < 0:
            ax.axhline(0, color='black', linestyle='--', linewidth=2, zorder=1)

    # Set y-ticks based on yticks_increment
    if yticks_increment is not None:
        y_min, y_max = ax.get_ylim()
        y_ticks = np.arange(np.floor(y_min), np.ceil(y_max) + yticks_increment, yticks_increment)
        ax.set_yticks(y_ticks)

    # Remove the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Display the plot without legend
    plt.tight_layout()
    plt.show()


def plot_y_across_bouts_colors(df, title='Mean Across Bouts', ylabel='Mean Value', custom_xtick_labels=None, ylim=None, bar_color='#00B7D7'):
    """
    Plots the mean values during investigations or other events across bouts with error bars for SEM
    and individual subject lines connecting the bouts.

    Parameters:
    - df (DataFrame): A DataFrame where rows are subjects, and bouts are columns.
                      Values should represent the mean values (e.g., mean DA, investigation times)
                      for each subject and bout.
    - title (str): The title for the plot.
    - ylabel (str): The label for the y-axis.
    - custom_xtick_labels (list): A list of custom x-tick labels. If not provided, defaults to the column names.
    - ylim (tuple): A tuple (min, max) to set the y-axis limits. If None, the limits are set automatically based on the data.
    - bar_color (str): A color or list of colors to use for the bars. Defaults to '#00B7D7'.
    """

    # Calculate the mean and SEM for each bout (across all subjects)
    mean_values = df.mean()
    sem_values = df.sem()

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot the bar plot with error bars (mean and SEM) without adding it to the legend
    bars = ax.bar(
        df.columns, 
        mean_values, 
        yerr=sem_values, 
        capsize=6,  # Increase capsize for larger error bars
        color=bar_color,  # Custom bar colors
        linewidth=2,  # Thicker and darker bar outlines
        width=0.6,
        hatch='/',  # Add diagonal dashes
        edgecolor = 'black',
        error_kw=dict(elinewidth=2.5, capthick=2.5, zorder=5)  # Thicker error bars and make them appear above circles
    )


    # Plot the lines first with a reduced linewidth
    for i, subject in enumerate(df.index):
        ax.plot(df.columns, df.loc[subject], linestyle='-', color='gray', alpha=0.5, linewidth=2, zorder=1)

    # Then plot the unfilled circle markers with larger size and custom color based on subject name
    for i, subject in enumerate(df.index):
        # Determine color based on subject name
        if subject.startswith('n'):
            marker_color = '#15616F'
        elif subject.startswith('p'):
            marker_color = '#FFAF00'
        else:
            marker_color = 'gray'  # Default color if the subject name doesn't match the criteria

        ax.scatter(df.columns, df.loc[subject], color=marker_color, s=120, alpha=1, linewidth=2, label=subject, zorder=2)


    # Add labels, title, and format
    ax.set_ylabel(ylabel, fontsize=30)
    ax.set_xlabel('Agent', fontsize=30, labelpad=12)
    ax.set_title(title, fontsize=20)
    
    # Set x-ticks to match the bout labels
    ax.set_xticks(np.arange(len(df.columns)))

    # Use custom x-tick labels if provided, otherwise use the column names
    if custom_xtick_labels is not None:
        ax.set_xticklabels(custom_xtick_labels, fontsize=30)
    else:
        ax.set_xticklabels(df.columns, fontsize=30)

    # Increase the font size of y-axis tick numbers
    ax.tick_params(axis='y', labelsize=30)  # Increase y-axis number size
    ax.tick_params(axis='x', labelsize=30)  # Optional: also increase x-axis number size


    # Automatically set the y-limits based on the data range if ylim is not provided
    if ylim is None:
        # Collect all values to determine the y-limits
        all_values = np.concatenate([df.values.flatten(), mean_values.values.flatten()])
        min_val = np.nanmin(all_values)
        max_val = np.nanmax(all_values)

        # Set lower y-limit to 0 if all values are above 0, otherwise set to the minimum value
        lower_ylim = 0 if min_val > 0 else min_val * 1.1
        upper_ylim = max_val * 1.1  # Adding a bit of space above the highest value
        
        ax.set_ylim(lower_ylim, upper_ylim)
    else:
        # If ylim is provided, set the limits to the specified values
        ax.set_ylim(ylim)
        if ylim[0] < 0:
            ax.axhline(0, color='gray', linestyle='--', linewidth=2, zorder=3)

    # Add the legend on the right side, outside the plot (but only for individual subjects)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Subjects")

    # Add a dashed line at y=0 if it exists in the y-limits and auto-adjusted
    if ylim is None and lower_ylim < 0:
        ax.axhline(0, color='gray', linestyle='--', linewidth=2, zorder=3)

    # Remove the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

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


def extract_nth_behavior_mean_da_corrected(group_data, bouts, behavior='Investigation', n=1, max_duration=5.0):
    """
    Extracts the mean DA during the n-th occurrence of the specified behavior (e.g., 'Investigation') 
    for each subject and bout, and limits the analysis to a maximum of max_duration seconds.
    Returns the data in a DataFrame.

    Parameters:
    group_data (object): The object containing bout data for each subject.
    bouts (list): A list of bout names to process.
    behavior (str): The behavior of interest to extract mean DA for the n-th occurrence (default is 'Investigation').
    n (int): The occurrence number of the behavior to extract (default is 1).
    max_duration (float): The maximum duration in seconds to limit DA analysis for each behavior event (default is 5.0 seconds).

    Returns:
    pd.DataFrame: A DataFrame where each row represents a subject, 
                  and each column represents the mean DA during the n-th occurrence of the specified behavior 
                  for a specific bout, limited to max_duration seconds.
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
                            event_start = nth_behavior['Start Time']
                            event_end = nth_behavior['End Time']
                            event_duration = nth_behavior['Total Duration']

                            # Limit the analysis to max_duration seconds
                            analysis_end_time = min(event_start + max_duration, event_end)

                            # Get the z-score signal during the limited event window
                            zscore_indices = (block_data.timestamps >= event_start) & (block_data.timestamps <= analysis_end_time)
                            mean_da_nth_behavior = np.mean(block_data.zscore[zscore_indices])  # Compute mean DA
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


def extract_nth_behavior_mean_baseline_peth(group_data, bouts, behavior='Investigation', n=1, windows=[(0, 3)], pre_time=5, post_time=5):
    """
    Extracts the mean of the peri-event time histogram (PETH) data during the n-th occurrence of the specified behavior,
    using baseline z-scoring based on the pre-event period.
    Returns the data in a DataFrame.

    Parameters:
    group_data (object): The object containing bout data for each subject.
    bouts (list): A list of bout names to process.
    behavior (str): The behavior of interest to extract mean PETH for the n-th occurrence (default is 'Investigation').
    n (int): The occurrence number of the behavior to extract (default is 1).
    windows (list of tuples): List of time windows (start, end) in seconds to calculate mean PETH (default is [(0, 3)]).
    pre_time (float): The time in seconds to include before the behavior starts.
    post_time (float): The time in seconds to include after the behavior starts.

    Returns:
    pd.DataFrame: A DataFrame where each row represents a subject, and each column represents the mean PETH during the n-th occurrence of the specified behavior
                  for a specific bout, limited to the specified time windows.
    """
    # Initialize an empty list to hold the data for each subject
    data_list = []

    # Populate the data_list from the group_data.blocks
    for block_name, block_data in group_data.blocks.items():
        if hasattr(block_data, 'bout_dict') and block_data.bout_dict:  # Ensure bout_dict exists and is populated
            # Use the subject name from the TDTData object
            block_data_dict = {'Subject': block_data.subject_name}

            for bout in bouts:  # Only process bouts in the given list of bouts
                if bout in block_data.bout_dict and behavior in block_data.bout_dict[bout]:
                    # Ensure the requested n-th occurrence exists
                    if len(block_data.bout_dict[bout][behavior]) >= n:
                        nth_behavior = block_data.bout_dict[bout][behavior][n - 1]  # Get the n-th occurrence

                        # Compute the peri-event data for this specific event
                        peri_event_data = block_data.compute_nth_bout_baseline_peth(
                            bout_name=bout,
                            behavior_name=behavior,
                            nth_event=n,
                            pre_time=pre_time,
                            post_time=post_time
                        )

                        # If peri_event_data is None, skip to next iteration
                        if peri_event_data is None:
                            print(f"Peri-event data not available for {block_data.subject_name} in {bout}.")
                            continue

                        # Extract the z-score and time axis from peri_event_data
                        peri_event_zscore = peri_event_data['zscore']
                        time_axis = peri_event_data['time_axis']

                        # Calculate mean PETH for each window
                        for start, end in windows:
                            # Find indices corresponding to the current window
                            window_indices = (time_axis >= start) & (time_axis <= end)
                            
                            if np.any(window_indices):  # Check if there are valid indices
                                mean_peth_window = np.nanmean(peri_event_zscore[window_indices])
                            else:
                                mean_peth_window = np.nan  # If no data, assign NaN
                            
                            # Store the result in the dictionary
                            block_data_dict[f'{bout}_{start}s_to_{end}s'] = mean_peth_window

                    else:
                        # If fewer than n occurrences, assign NaN
                        for start, end in windows:
                            block_data_dict[f'{bout}_{start}s_to_{end}s'] = np.nan

                else:
                    # If bout or behavior not found, assign NaN
                    for start, end in windows:
                        block_data_dict[f'{bout}_{start}s_to_{end}s'] = np.nan

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


def extract_nth_to_mth_behavior_mean_da(group_data, bouts, behavior='Investigation', n_start=1, n_end=5):
    """
    Extracts the mean DA during the n-th to m-th occurrences of the specified behavior (e.g., 'Investigation')
    for each subject and bout, and returns the data in a DataFrame.

    Parameters:
    group_data (object): The object containing bout data for each subject.
    bouts (list): A list of bout names to process.
    behavior (str): The behavior of interest to extract mean DA for (default is 'Investigation').
    n_start (int): The starting occurrence number of the behavior to extract.
    n_end (int): The ending occurrence number of the behavior to extract.

    Returns:
    pd.DataFrame: A DataFrame where each row represents a subject,
                  and each column represents the mean DA during the n-th to m-th occurrences of the specified behavior 
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
                mean_da_values = []
                if bout in block_data.bout_dict and behavior in block_data.bout_dict[bout]:
                    # Collect the mean DA for the n-th to m-th occurrences
                    for n in range(n_start, n_end + 1):
                        if len(block_data.bout_dict[bout][behavior]) >= n:
                            nth_behavior = block_data.bout_dict[bout][behavior][n - 1]  # Get the n-th occurrence
                            if 'Mean zscore' in nth_behavior:
                                mean_da_nth_behavior = nth_behavior['Mean zscore']
                            else:
                                mean_da_nth_behavior = np.nan  # If no z-score data, assign NaN
                        else:
                            mean_da_nth_behavior = np.nan  # If fewer than n occurrences, assign NaN
                        mean_da_values.append(mean_da_nth_behavior)

                block_data_dict[bout] = mean_da_values

            # Append the block's data to the data_list
            data_list.append(block_data_dict)

    # Convert the data_list into a DataFrame
    behavior_mean_df = pd.DataFrame(data_list)

    # Set the index to 'Subject'
    behavior_mean_df.set_index('Subject', inplace=True)

    return behavior_mean_df

# Exponential decay model
def exp_decay(t, A, k):
    return A * np.exp(-k * t)

def plot_meanDA_across_investigations(mean_da_df, bouts, max_investigations=5, metric_type='slope', colors=custom_palette, custom_xtick_labels=None):
    """
    Plots the mean DA from the 1st to 5th investigations across bouts and calculates either the slope or decay constant.
    
    Parameters:
    mean_da_df (pd.DataFrame): A DataFrame where each row represents a subject, and each column represents a list of mean DA
                               values during the investigations for a specific bout.
    bouts (list): A list of bout names to plot.
    max_investigations (int): Maximum number of investigations to consider (default is 5).
    metric_type (str): Whether to compute 'slope' or 'decay' (default is 'slope').
    colors (list): A list of colors to use for different bouts (default is custom_palette).
    custom_xtick_labels (list): Custom labels for the x-ticks, if provided. Otherwise, defaults to investigation numbers.
    
    Returns:
    None. Displays the line plot and prints the slope or decay constant for each bout.
    """
    # Create a plot for each bout
    fig, ax = plt.subplots(figsize=(12, 6))

    # Dictionary to store slopes or decay constants for each bout
    metrics = {}

    for i, bout in enumerate(bouts):
        # Extract data for the bout and truncate to the first 'max_investigations' investigations
        bout_data = mean_da_df[bout].apply(lambda x: x[:max_investigations] + [np.nan] * (max_investigations - len(x)) if len(x) < max_investigations else x[:max_investigations])
        bout_data = np.array(bout_data.tolist())  # Extract the list of mean DA values for each subject

        # Average across subjects for each investigation (column-wise mean)
        mean_across_investigations = np.nanmean(bout_data, axis=0)

        # Define time points (investigation numbers)
        x_values = np.arange(1, len(mean_across_investigations) + 1)  # [1, 2, 3, 4, 5] for investigations

        if metric_type == 'slope':
            # Calculate slope using linear regression (linregress)
            slope, intercept, r_value, p_value, std_err = linregress(x_values, mean_across_investigations)
            metrics[bout] = slope

            # Plot the line for the bout with slope in the label and custom colors
            ax.plot(x_values, mean_across_investigations, marker='o', linestyle='-', color=colors[i % len(colors)], label=f'{bout} (slope: {slope:.2f})')

        elif metric_type == 'decay':
            # Fit the exponential decay model to the data
            try:
                popt, _ = curve_fit(exp_decay, x_values, mean_across_investigations, p0=(mean_across_investigations[0], 0.1))
                A, k = popt  # A is the initial value, k is the decay constant
                metrics[bout] = k

                # Generate fitted decay curve for plotting
                fitted_curve = exp_decay(x_values, *popt)

                # Plot the line for the bout with decay constant in the label and custom colors
                ax.plot(x_values, fitted_curve, marker='o', linestyle='-', color=colors[i % len(colors)], label=f'{bout} (decay: {k:.2f})')
            except RuntimeError:
                metrics[bout] = np.nan  # If fitting fails, store NaN

        else:
            raise ValueError("Invalid metric_type. Use 'slope' or 'decay'.")

    # Add labels, title, and legend
    ax.set_xlabel('Investigation Number', fontsize=20)
    ax.set_ylabel('Mean DA (z-scored dFF)', fontsize=20)
    ax.set_title(f'Mean DA during 1st to {max_investigations} Investigation Bouts Per Agent({metric_type.capitalize()})', fontsize=20)

    # Customize tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Set custom x-tick labels if provided, otherwise default to investigation numbers
    if custom_xtick_labels is not None:
        ax.set_xticks(np.arange(1, len(custom_xtick_labels) + 1))
        ax.set_xticklabels(custom_xtick_labels, fontsize=20)
    else:
        ax.set_xticks(np.arange(1, len(x_values) + 1))
        ax.set_xticklabels(x_values, fontsize=20)

    # Remove the top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Add a legend
    ax.legend(fontsize=15)

    # Display the plot
    plt.tight_layout()
    plt.show()

    # Print the slopes or decay constants
    for bout, metric in metrics.items():
        if metric_type == 'slope':
            print(f'Slope for {bout}: {metric:.2f}')
        elif metric_type == 'decay':
            print(f'Decay constant for {bout}: {metric:.4f}')



def compute_mean_da_across_trials(group_data, n=15, pre_time=5, post_time=5, bin_size=0.1, mean_window=4):
    """
    Processes the data to compute the mean DA signal across all trials for each of the first n sound cues.
    
    Parameters:
    - group_data: The GroupTDTData object containing the data blocks.
    - n: Number of sound cues to process.
    - pre_time: Time before port entry onset to include in PETH (seconds).
    - post_time: Time after port entry onset to include in PETH (seconds).
    - bin_size: Bin size for PETH (seconds).
    - mean_window: The time window (in seconds) from 0 to mean_window to compute the mean DA signal.
    
    Returns:
    - df: A pandas DataFrame containing trial numbers, mean DA signals, and SEMs.
    """
    # Initialize data structures
    peri_event_signals = [[] for _ in range(n)]  # List to collect signals for each of the first n port entries
    common_time_axis = np.arange(-pre_time, post_time + bin_size, bin_size)
    
    # Iterate over all blocks in group_data
    for block_name, block_data in group_data.blocks.items():
        print(f"Processing block: {block_name}")
    
        # Extract sound cue onsets and port entry onsets from the block
        sound_cue_onsets = np.array(block_data.behaviors['sound cues'].onset)
        port_entry_onsets = np.array(block_data.behaviors['port entries'].onset)
        
        # Limit to the first n sound cues
        sound_cue_onsets = sound_cue_onsets[:n]
        
        # For each sound cue
        for sc_index, sc_onset in enumerate(sound_cue_onsets):
            # Find the first port entry after the sound cue onset
            pe_indices = np.where(port_entry_onsets > sc_onset)[0]
            if len(pe_indices) == 0:
                print(f"No port entries found after sound cue at {sc_onset} seconds in block {block_name}.")
                continue
            first_pe_index = pe_indices[0]
            pe_onset = port_entry_onsets[first_pe_index]
            
            # Define time window around the port entry onset
            start_time = pe_onset - pre_time
            end_time = pe_onset + post_time
            
            # Get indices of DA signal within this window
            indices = np.where((block_data.timestamps >= start_time) & (block_data.timestamps <= end_time))[0]
            if len(indices) == 0:
                print(f"No DA data found for port entry at {pe_onset} seconds in block {block_name}.")
                continue
            
            # Extract DA signal and timestamps
            da_segment = block_data.zscore[indices]
            time_segment = block_data.timestamps[indices] - pe_onset  # Align time to port entry onset
            
            # Interpolate DA signal onto the common time axis
            interpolated_da = np.interp(common_time_axis, time_segment, da_segment)
            
            # Collect the interpolated DA signal
            peri_event_signals[sc_index].append(interpolated_da)

    # Compute the mean DA signal and SEM across all trials for each port entry number
    trial_mean_da = []
    trial_sem_da = []
    mean_indices = np.where((common_time_axis >= 0) & (common_time_axis <= mean_window))[0]
    
    for event_signals in peri_event_signals:
        if event_signals:
            # Convert list of signals to numpy array
            event_signals = np.array(event_signals)
            # Compute mean PETH across all trials for this event
            mean_peth = np.mean(event_signals, axis=0)
            # Compute mean DA in the specified window
            mean_da = np.mean(mean_peth[mean_indices])
            sem_da = np.std(mean_peth[mean_indices]) / np.sqrt(len(event_signals))
            trial_mean_da.append(mean_da)
            trial_sem_da.append(sem_da)
        else:
            trial_mean_da.append(np.nan)  # Handle cases where no data is available
            trial_sem_da.append(np.nan)  # Handle cases where no data is available

    # Create a DataFrame to store the results
    df = pd.DataFrame({
        'Trial': np.arange(1, n + 1),
        'Mean_DA': trial_mean_da,
        'SEM_DA': trial_sem_da
    })
    
    return df

def plot_linear_fit_with_error_bars(df, color='blue'):
    """
    Plots the mean DA values with SEM error bars, fits a line of best fit,
    and computes the Pearson correlation coefficient.
    
    Parameters:
    - df: A pandas DataFrame containing trial numbers, mean DA signals, and SEMs.
    
    Returns:
    - slope: The slope of the line of best fit.
    - intercept: The intercept of the line of best fit.
    - r_value: The Pearson correlation coefficient.
    - p_value: The p-value for the correlation coefficient.
    """
    # Sort the DataFrame by Trial
    df_sorted = df.sort_values('Trial')
    
    # Extract trial numbers, mean DA values, and SEMs
    x_data = df_sorted['Trial'].values
    y_data = df_sorted['Mean_DA'].values
    y_err = df_sorted['SEM_DA'].values
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
    y_fitted = intercept + slope * x_data
    
    # Plot the data with error bars and the fitted line
    plt.figure(figsize=(10, 6))
    plt.errorbar(x_data, y_data, yerr=y_err, fmt='o', label='Mean DA per Trial', color=color, capsize=5)
    plt.plot(x_data, y_fitted, 'r--', label=f'Best Fit Line\n$R^2$ = {r_value**2:.2f}, p = {p_value:.2e}')
    plt.xlabel('Tone Number', fontsize=24)
    plt.ylabel('Mean DA Signal (z-score)', fontsize=24)
    plt.title('Mean DA Signal Over Trials with Best Fit Line', fontsize=10)
    plt.legend(fontsize=12)
    
    # Set custom x-ticks from 2 to 16 (whole numbers)
    plt.xticks(np.arange(1, 17, 2), fontsize=20)

    # Remove the top and right spines
    ax = plt.gca()  # Get current axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_ylim(0,0.8)
    # Optionally, adjust tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=20)

    plt.tight_layout()
    plt.show()
    
    print(f"Slope: {slope:.4f}, Intercept: {intercept:.4f}")
    print(f"Pearson correlation coefficient (R): {r_value:.4f}, p-value: {p_value:.4e}")
    
    return slope, intercept, r_value, p_value

# Example usage:






