import tdt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def remove_intial_LED_artifact(tdt_data):
    '''
    This function removes the initial artifact caused by the onset of LEDs turning on.
    The artifact is assumed to occur within the first 't' seconds of the data.
    
    Parameters:
    tdt_data: The TDT data object containing the streams with data to be processed.
    '''
    t = 10

    # Calculate the timestamps using np.arange
    fs = tdt_data.streams['_465A'].fs
    time = np.arange(len(tdt_data.streams['_465A'].data)) / fs
    
    inds = np.where(time > t)
    ind = inds[0][0]
    # Remove the initial data affected by the artifact from the '_465A' stream (Dopamine data).
    tdt_data.streams['_465A'].data = tdt_data.streams['_465A'].data[ind:]
    # Remove the initial data affected by the artifact from the '_405A' stream (Isobestic data).
    tdt_data.streams['_405A'].data = tdt_data.streams['_405A'].data[ind:]

def plot_raw_trace(tdt_data):
    # Make some variables up here to so if they change in new recordings you won't have to change everything downstream
    ISOS = '_405A' # 405nm channel.
    DA = '_465A'
    # Make a time array based on the number of samples and sample freq of the demodulated streams
    time = np.linspace(1,len(tdt_data.streams[DA].data), len(tdt_data.streams[DA].data))/tdt_data.streams[DA].fs

    # Plot both unprocessed demodulated stream            
    fig1 = plt.figure(figsize=(18,6))
    ax0 = fig1.add_subplot(111)

    # Plotting the traces
    p1, = ax0.plot(time, tdt_data.streams[DA].data, linewidth=2, color='blue', label='mPFC')
    p2, = ax0.plot(time, tdt_data.streams[ISOS].data, linewidth=2, color='blueviolet', label='ISOS')

    ax0.set_ylabel('mV')
    ax0.set_xlabel('Seconds', fontsize=14)
    ax0.set_title('Raw Demodulated Responses', fontsize=14)
    ax0.legend(handles=[p1,p2], loc='upper right')

def downsample_data(tdt_data, N = 10):
    """
    Downsample the data by averaging every N samples and return the new time array

    Parameters:
    tdt_data: The TDT data object containing the streams with data to be processed.
    N (int): The number of samples to average into one value.

    Returns:
    time: The downsampled Time
    """
    ISOS = '_405A' # 405nm channel.
    DA = '_465A'
    F405 = []
    F465 = []

    for i in range(0, len(tdt_data.streams[DA].data), N):
        F465.append(np.mean(tdt_data.streams[DA].data[i:i+N-1])) # This is the moving window mean
    tdt_data.streams[DA].data = F465

    for i in range(0, len(tdt_data.streams[ISOS].data), N):
        F405.append(np.mean(tdt_data.streams[ISOS].data[i:i+N-1]))
    tdt_data.streams[ISOS].data = F405

    time = np.linspace(1, len(tdt_data.streams['_465A'].data), len(tdt_data.streams['_465A'].data)) / tdt_data.streams['_465A'].fs
    time = time[::N] # go from beginning to end of array in steps on N
    time = time[:len(tdt_data.streams[DA].data)]

    return time

def compute_dff(tdt_data):
    """
    Compute the delta F/F (dFF) using the ISOS data as a baseline for the DA data.

    Parameters:
    DA_data (list or array): The DA data stream.
    isos_data (list or array): The ISOS data stream used as the baseline.

    Returns:
    tuple: A tuple containing the dFF values and the standard deviation of the dFF.
    """
    ISOS = '_405A' # 405nm channel.
    DA = '_465A'

    x = np.array(tdt_data.streams[ISOS].data)
    y = np.array(tdt_data.streams[DA].data)
    
    # Fit a linear baseline to the DA data using the ISOS data
    bls = np.polyfit(x, y, 1)
    Y_fit_all = np.multiply(bls[0], x) + bls[1]
    Y_dF_all = y - Y_fit_all

    # Compute the dFF values
    dFF = np.multiply(100, np.divide(Y_dF_all, Y_fit_all))
    
    # Compute the standard deviation of the dFF values
    std_dFF = np.std(dFF)
    
    return dFF, std_dFF

def remove_time(tdt_data, start_time, end_time):
    """
    Removes a specified time range from all data streams in the tdt_data object and stitches the remaining data back together.

    Parameters:
    tdt_data: The TDT data object containing data streams.
    start_time (float): The start time of the range to be removed.
    end_time (float): The end time of the range to be removed.
    """
    for stream_name in tdt_data.streams:
        # Get the data and corresponding timestamps for the current stream
        data = tdt_data.streams[stream_name].data
        fs = tdt_data.streams[stream_name].fs
        timestamps = np.arange(len(data)) / fs

        # Find indices for the start and end time
        remove_indices = np.where((timestamps >= start_time) & (timestamps <= end_time))[0]

        # Create a mask for the indices to keep
        keep_indices = np.ones_like(timestamps, dtype=bool)
        keep_indices[remove_indices] = False

        # Remove the specified time range from data and timestamps
        data_trimmed = data[keep_indices]
        timestamps_trimmed = timestamps[keep_indices]

        # Adjust timestamps to stitch the remaining data
        timestamps_stitched = timestamps_trimmed - timestamps_trimmed[0]

        # Update the data stream in tdt_data with the trimmed data and adjusted timestamps
        tdt_data.streams[stream_name].data = data_trimmed
        tdt_data.streams[stream_name].timestamps = timestamps_stitched  # Assuming the TDT data structure supports storing timestamps

    print(f"Removed data from {start_time} to {end_time} and stitched the remaining data for all channels.")

