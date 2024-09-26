import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress
from scipy.optimize import curve_fit
import os


def ag_extract_aggression_events(self, behavior_csv_path):
    """
    Extracts aggression behaviors from the behavior CSV file and stores them in the bout_dict.
    
    Parameters:
    - behavior_csv_path (str): The file path to the CSV file containing behavior data.
    
    Returns:
    - None. Updates the self.bout_dict attribute with aggression events.
    """
    # Load the CSV file into a DataFrame
    data = pd.read_csv(behavior_csv_path)
    
    # Ensure the 'Behavior' column exists
    if 'Behavior' not in data.columns:
        print(f"'Behavior' column not found in {behavior_csv_path}.")
        return
    
    # Filter rows for aggression behaviors (case-insensitive)
    aggression_events = data[data['Behavior'].str.lower().str.contains('aggression')]
    
    if aggression_events.empty:
        print(f"No aggression events found in {behavior_csv_path}.")
        return
    
    # Initialize a dictionary to store aggression events
    # Assuming you want to store all aggression events under a single bout or category
    aggression_bout_key = 'aggression_bout'  # You can name this key as per your preference
    self.bout_dict[aggression_bout_key] = {}
    self.bout_dict[aggression_bout_key]['Aggression'] = []
    
    # Iterate through each aggression event and store its details
    for index, row in aggression_events.iterrows():
        try:
            onset = row['Start (s)']
            offset = row['Stop (s)']
            duration = row['Duration (s)']
            
            # Compute the mean z-score during the aggression event
            # Ensure that zscore and timestamps are available
            if self.zscore is None or self.timestamps is None:
                print("Z-score or timestamps data is missing. Cannot compute mean z-score for aggression events.")
                mean_zscore = np.nan
            else:
                zscore_indices = (self.timestamps >= onset) & (self.timestamps <= offset)
                if not np.any(zscore_indices):
                    print(f"No z-score data found for aggression event at index {index}.")
                    mean_zscore = np.nan
                else:
                    mean_zscore = np.mean(self.zscore[zscore_indices])
            
            # Create an event dictionary
            event_dict = {
                'Start Time': onset,
                'End Time': offset,
                'Duration': duration,
                'Mean zscore': mean_zscore
            }
            
            # Append the event to the aggression list
            self.bout_dict[aggression_bout_key]['Aggression'].append(event_dict)
        
        except KeyError as e:
            print(f"Missing expected column in CSV: {e}. Skipping row {index}.")
            continue
        except Exception as e:
            print(f"Error processing row {index}: {e}. Skipping.")
            continue
    
    # Optional: Print summary of extracted aggression events
    total_aggression_events = len(self.bout_dict[aggression_bout_key]['Aggression'])
    print(f"Extracted {total_aggression_events} aggression events from {behavior_csv_path}.")


def ag_proc_processing_all_blocks(self, behavior_csv_paths):
    """
    Processes multiple CSV files for the Social Defeat experiment, extracting bouts and aggression events.
    
    Parameters:
    - behavior_csv_paths (list of str): List of file paths to the CSV files.
    """
    for csv_path in behavior_csv_paths:
        if os.path.exists(csv_path):
            print(f"Processing {csv_path}...")
            self.ag_extract_aggression_events(csv_path)
        else:
            print(f"File not found: {csv_path}. Skipping.")



def ag_extract_average_aggression_durations(group_data, bouts, behavior='Aggression'):
    """
    Extracts the mean durations for the specified behavior (e.g., 'Aggression') 
    for each subject and bout, and returns the data in a DataFrame.

    Parameters:
    - group_data (object): The object containing bout data for each subject.
                            It should have a 'blocks' attribute, which is a dictionary 
                            where each value represents a block containing 'bout_dict' 
                            and 'subject_name'.
    - bouts (list): A list of bout names to process.
    - behavior (str): The behavior of interest to calculate mean durations for (default is 'Aggression').

    Returns:
    - pd.DataFrame: A DataFrame where each row represents a subject, 
                    and each column represents the mean duration of the specified behavior for a specific bout.
    """
    # Initialize an empty list to hold the data for each subject
    data_list = []

    # Iterate through each block in group_data
    for block_key, block_data in group_data.blocks.items():
        # Check if 'bout_dict' exists and is not empty
        if hasattr(block_data, 'bout_dict') and isinstance(block_data.bout_dict, dict) and block_data.bout_dict:
            # Initialize a dictionary for the current block/subject
            block_data_dict = {'Subject': block_data.subject_name}

            # Iterate through each specified bout
            for bout in bouts:
                # Check if the bout exists in the bout_dict
                if bout in block_data.bout_dict:
                    # Check if the specified behavior exists within the bout
                    if behavior in block_data.bout_dict[bout]:
                        # Extract all durations for the specified behavior
                        durations = [event.get('Duration', np.nan) for event in block_data.bout_dict[bout][behavior]]
                        
                        # Filter out any NaN durations
                        valid_durations = [dur for dur in durations if not pd.isna(dur)]
                        
                        if valid_durations:
                            # Compute the mean duration, ignoring NaN values
                            mean_duration = np.nanmean(valid_durations)
                        else:
                            # Assign NaN if no valid durations are available
                            mean_duration = np.nan
                        
                        # Assign the mean duration to the current bout
                        block_data_dict[bout] = mean_duration
                    else:
                        # Assign NaN if the behavior does not exist in the bout
                        block_data_dict[bout] = np.nan
                else:
                    # Assign NaN if the bout does not exist in the bout_dict
                    block_data_dict[bout] = np.nan

            # Append the current block's data to the data_list
            data_list.append(block_data_dict)
        else:
            print(f"Block '{block_key}' does not have a valid 'bout_dict'. Skipping.")
            continue

    # Convert the list of dictionaries to a Pandas DataFrame
    aggression_duration_df = pd.DataFrame(data_list)

    # Set 'Subject' as the index of the DataFrame
    if 'Subject' in aggression_duration_df.columns:
        aggression_duration_df.set_index('Subject', inplace=True)

    return aggression_duration_df



def compute_nth_bout_peth_all_blocks_standard_zscore(
    self, 
    behavior_name='Aggression', 
    nth_occurrence=1, 
    bouts=None, 
    pre_time=5, 
    post_time=5, 
    bin_size=0.1
):
    """
    Computes the peri-event time histogram (PETH) data for the nth occurrence of a given behavior in each bout using precomputed z-score.
    Stores the peri-event data (zscore and time axis) for each bout as a class variable.

    Parameters:
    - behavior_name (str): The name of the behavior to generate the PETH for (e.g., 'Aggression').
    - nth_occurrence (int): The occurrence number of the behavior to analyze (1 for first occurrence, 2 for second, etc.).
    - bouts (list): A list of bout names to process. If None, defaults to ['Short_Term_1', 'Short_Term_2', 'Novel_1', 'Long_Term_1'].
    - pre_time (float): The time in seconds to include before the event.
    - post_time (float): The time in seconds to include after the event.
    - bin_size (float): The size of each bin in the histogram (in seconds).

    Returns:
    - None. Stores peri-event data for all blocks and bouts in self.peri_event_data_all_blocks.
    """
    if bouts is None:
        bouts = ['Short_Term_1', 'Short_Term_2', 'Novel_1', 'Long_Term_1']  # Default bouts

    peri_event_data_all_blocks = {}  # Temporary dictionary to store PETH data

    # Initialize to track the minimum number of bins across all PETHs
    min_num_bins = float('inf')

    # Iterate through each block in self.blocks
    for block_name, block_data in self.blocks.items():
        # Initialize dictionary for the current block
        peri_event_data_all_blocks[block_name] = {}

        # Validate required attributes
        if not hasattr(block_data, 'zscore') or block_data.zscore is None:
            print(f"Block '{block_name}' is missing 'zscore' data. Skipping.")
            continue
        if not hasattr(block_data, 'timestamps') or block_data.timestamps is None:
            print(f"Block '{block_name}' is missing 'timestamps' data. Skipping.")
            continue
        if not hasattr(block_data, 'bout_dict') or not isinstance(block_data.bout_dict, dict):
            print(f"Block '{block_name}' is missing a valid 'bout_dict'. Skipping.")
            continue

        # Iterate through each specified bout
        for bout in bouts:
            # Check if the bout and behavior exist in the current block
            if bout in block_data.bout_dict and behavior_name in block_data.bout_dict[bout]:
                events = block_data.bout_dict[bout][behavior_name]
                
                # Check if the desired nth occurrence exists
                if len(events) >= nth_occurrence:
                    # Get the nth occurrence (1-based index)
                    event = events[nth_occurrence - 1]
                    event_time = event['Start Time']

                    # Define PETH window around the event
                    window_start = event_time - pre_time
                    window_end = event_time + post_time

                    # Adjust window_start if it goes below 0
                    if window_start < 0:
                        print(f"Block '{block_name}', Bout '{bout}': Window start time {window_start}s is less than 0. Adjusting to 0.")
                        window_start = 0

                    # Adjust window_end if it exceeds the signal duration
                    if window_end > block_data.timestamps[-1]:
                        print(f"Block '{block_name}', Bout '{bout}': Window end time {window_end}s exceeds signal duration. Adjusting to {block_data.timestamps[-1]}s.")
                        window_end = block_data.timestamps[-1]

                    # Extract z-score signal within the PETH window
                    window_mask = (block_data.timestamps >= window_start) & (block_data.timestamps <= window_end)
                    window_zscore = block_data.zscore[window_mask]
                    window_time = block_data.timestamps[window_mask] - event_time  # Align time to event

                    if len(window_zscore) == 0:
                        print(f"Block '{block_name}', Bout '{bout}': No z-score data found in the window [{window_start}, {window_end}]s. Skipping.")
                        continue

                    # Define the number of bins
                    num_bins = int(np.ceil((pre_time + post_time) / bin_size))
                    binned_signal = np.empty(num_bins)
                    binned_signal[:] = np.nan  # Initialize with NaNs

                    # Define bin edges
                    bin_edges = np.linspace(-pre_time, post_time, num_bins + 1)

                    # Bin the z-score signal
                    for i in range(num_bins):
                        bin_start = bin_edges[i]
                        bin_end = bin_edges[i + 1]
                        bin_mask = (window_time >= bin_start) & (window_time < bin_end)
                        if np.any(bin_mask):
                            binned_signal[i] = np.mean(window_zscore[bin_mask])
                        else:
                            binned_signal[i] = np.nan  # Assign NaN if no data in bin

                    # Define the time axis for PETH
                    binned_time = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoint of bins

                    # Update the minimum number of bins if necessary
                    if num_bins < min_num_bins:
                        min_num_bins = num_bins

                    # Store the PETH data
                    peri_event_data_all_blocks[block_name][bout] = {
                        'time_axis': binned_time.tolist(),
                        'zscore': binned_signal.tolist()
                    }
                else:
                    print(f"Block '{block_name}', Bout '{bout}': Less than {nth_occurrence} occurrences of '{behavior_name}'. Skipping.")
            else:
                print(f"Block '{block_name}', Bout '{bout}': Behavior '{behavior_name}' not found. Skipping.")

    # After processing all blocks and bouts, truncate all PETHs to the minimum number of bins
    for block_name, bouts_data in peri_event_data_all_blocks.items():
        for bout_name, peth_data in bouts_data.items():
            # Truncate the PETH data to min_num_bins
            if len(peth_data['time_axis']) > min_num_bins:
                peth_data['time_axis'] = peth_data['time_axis'][:min_num_bins]
                peth_data['zscore'] = peth_data['zscore'][:min_num_bins]

    # Store the computed PETH data in the class attribute
    self.peri_event_data_all_blocks = peri_event_data_all_blocks