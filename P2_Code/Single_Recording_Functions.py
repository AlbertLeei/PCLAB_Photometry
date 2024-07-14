import tdt
import numpy as np
import pandas as pd
import matplotlib as plt

'''Boris Feature Extraction'''

def extract_single_behavior(tdt_recording, behavior_name, bout_aggregated_df):
    '''
    This function extracts single behavior events from the DataFrame and adds them to the TDT recording.

    Parameters:
    tdt_recording: The TDT recording object where the extracted events will be added.
    behavior_name: The name of the behavior to extract.
    bout_aggregated_df: The DataFrame containing the behavior data.
    '''
    
    # Filter the DataFrame to get rows corresponding to the specific behavior
    behavior_df = bout_aggregated_df[bout_aggregated_df['Behavior'] == behavior_name]
    
    # Extract onset and offset times as lists
    onset_times = behavior_df['Start (s)'].values.tolist()
    offset_times = behavior_df['Stop (s)'].values.tolist()
    
    # Define the event name by appending '_event' to the behavior name
    event_name = behavior_name + '_event'
    
    # Create a data array filled with 1s, with the same length as onset_times
    data_arr = [1] * len(onset_times)

    # Create a dictionary to hold the event data
    EVENT_DICT = {
        "name": event_name,
        "onset": onset_times,
        "offset": offset_times,
        "type_str": tdt_recording.epocs.Cam1.type_str,  # Copy type_str from an existing epoc
        "data": data_arr
    }

    # Assign the new event dictionary to the tdt_recording.epocs structure
    tdt_recording.epocs[event_name] = tdt.StructType(EVENT_DICT)

def extract_manual_annotation_behaviors(tdt_recording, bout_aggregated_csv_path):
    '''
    This function processes all behaviors of type 'STATE' in the CSV file and extracts them
    into the TDT recording.

    Parameters:
    tdt_recording: The TDT recording object where the extracted events will be added.
    bout_aggregated_csv_path: The file path to the CSV containing the behavior data.
    '''
    
    # Load the CSV file into a DataFrame
    bout_aggregated_df = pd.read_csv(bout_aggregated_csv_path)

    # Filter the DataFrame to include only rows where the behavior type is 'STATE'
    state_behaviors_df = bout_aggregated_df[bout_aggregated_df['Behavior type'] == 'STATE']
    
    # Get a list of unique behaviors in the filtered DataFrame
    unique_behaviors = state_behaviors_df['Behavior'].unique()

    # Iterate over each unique behavior
    for behavior in unique_behaviors:
        # Filter the DataFrame for the current behavior
        behavior_df = state_behaviors_df[state_behaviors_df['Behavior'] == behavior]
        
        # Call the helper function to extract and add the behavior events to tdt_recording
        extract_single_behavior(tdt_recording, behavior, behavior_df)
