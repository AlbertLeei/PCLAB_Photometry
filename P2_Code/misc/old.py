    def smooth_signal(self, filter_window=100, filter_type='moving_average'):
        '''
        Smooths the signal using a specified filter type.

        Parameters:
        filter_window (int): The window size for the filter.
        filter_type (str): The type of filter to use. Options are 'moving_average' or 'lowpass'.
        '''
        for stream_name in ['DA', 'ISOS']:
            if stream_name in self.streams:
                data = self.streams[stream_name]

                if filter_type == 'moving_average':
                    # Moving average filter
                    b = np.ones(filter_window) / filter_window
                    a = 1
                elif filter_type == 'lowpass':
                    # Lowpass filter (Butterworth)
                    nyquist = 0.5 * self.fs
                    cutoff_freq = 1.0  # Set cutoff frequency in Hz (adjust as needed)
                    normal_cutoff = cutoff_freq / nyquist
                    b, a = ss.butter(N=filter_window, Wn=normal_cutoff, btype='low', analog=False)
                else:
                    raise ValueError("Invalid filter_type. Choose 'moving_average' or 'lowpass'.")

                smoothed_data = ss.filtfilt(b, a, data)
                self.streams[stream_name] = smoothed_data

        # Clear dFF and zscore since the raw data has changed
        self.dFF = None
        self.zscore = None

                
    def downsample_data(self, N=10):
        downsampled_timestamps = self.timestamps[::N]
        for stream_name in ['DA', 'ISOS']:
            if stream_name in self.streams:
                data = self.streams[stream_name]
                downsampled_data = [np.mean(data[i:i + N]) for i in range(0, len(data), N)]
                self.streams[stream_name] = downsampled_data
        self.timestamps = downsampled_timestamps

        # Clear dFF and zscore since the raw data has changed
        self.dFF = None
        self.zscore = None

    def remove_time(self, start_time, end_time):
        """
        Removes a segment of time from the data streams and timestamps and then verifies the signal length.
        
        Parameters:
        start_time (float): The start time of the segment to be removed (in seconds).
        end_time (float): The end time of the segment to be removed (in seconds).
        """
        # Find the indices corresponding to the start and end times
        start_index = np.where(self.timestamps >= start_time)[0][0]
        end_index = np.where(self.timestamps <= end_time)[0][-1]
        
        # Create an array of boolean values, keeping all indices outside the specified range
        keep_indices = np.ones_like(self.timestamps, dtype=bool)
        keep_indices[start_index:end_index+1] = False
        
        # Update the streams by applying the boolean mask
        for stream_name in ['DA', 'ISOS']:
            if stream_name in self.streams:
                self.streams[stream_name] = np.array(self.streams[stream_name])[keep_indices]  # Apply the mask
        
        # Update the timestamps by applying the boolean mask
        self.timestamps = self.timestamps[keep_indices]
        
        # Clear dFF and zscore since the raw data has changed
        self.dFF = None
        self.zscore = None
        
        # Verify the signal lengths to ensure consistency
        self.verify_signal()


            # def execute_controlFit_dff(self, control, signal, filter_window=100):
    #     """
    #     Fits the control channel to the signal channel and calculates delta F/F (dFF).

    #     Parameters:
    #     control (numpy.array): The control signal (e.g., isosbestic control signal).
    #     signal (numpy.array): The signal of interest (e.g., dopamine signal).
    #     filter_window (int): The window size for the moving average filter.

    #     Returns:
    #     norm_data (numpy.array): The normalized delta F/F signal.
    #     control_fit (numpy.array): The fitted control signal.
    #     """
    #     if filter_window > 1:
    #         # Smoothing both signals
    #         control_smooth = ss.filtfilt(np.ones(filter_window) / filter_window, 1, control)
    #         signal_smooth = ss.filtfilt(np.ones(filter_window) / filter_window, 1, signal)
    #     else:
    #         control_smooth = control
    #         signal_smooth = signal

    #     # Fitting the control signal to the signal of interest
    #     p = np.polyfit(control_smooth, signal_smooth, 1)
    #     control_fit = p[0] * control_smooth + p[1]

    #     # Calculating delta F/F (dFF)
    #     norm_data = 100 * (signal_smooth - control_fit) / control_fit

    #     return norm_data, control_fit

    # def compute_dff(self, filter_window=100):
    #     """
    #     Computes the delta F/F (dFF) signal by fitting the isosbestic control signal to the signal of interest.
        
    #     Parameters:
    #     filter_window (int): The window size for the moving average filter.
    #     """
    #     if 'DA' in self.streams and 'ISOS' in self.streams:
    #         signal = np.array(self.streams['DA'])
    #         control = np.array(self.streams['ISOS'])
            
    #         # Call the execute_controlFit_dff method
    #         self.dFF, self.control_fit = self.execute_controlFit_dff(control, signal, filter_window)
            
    #         # Calculate the standard deviation of dFF
    #         self.std_dFF = np.std(self.dFF)
    #     else:
    #         self.dFF = None
    #         self.std_dFF = None