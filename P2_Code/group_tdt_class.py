import os
import tdt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from single_tdt_class import *
import sys

# root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))  # Go up one directory to P2_Code
# # Add the root directory to sys.path
# sys.path.append(root_dir)

class GroupTDTData:
    def __init__(self, experiment_folder_path, csv_base_path):
        """
        Initializes the GroupTDTData object with paths.
        """
        self.experiment_folder_path = experiment_folder_path
        self.csv_base_path = csv_base_path
        self.blocks = {}
        self.group_psth = None

        self.load_blocks()

        # Hab Dishab
        self.hab_dishab_df = pd.DataFrame()
    
    from P2_Code.hab_dishab.hab_dishab_extension import hab_dishab_processing, hab_dishab_plot_individual_behavior, plot_investigation_vs_dff_all, plot_all_investigation_vs_dff_all,plot_investigation_mean_DA_boutwise, plot_investigation_durations_boutwise
    # from P2_Code.social_pref. import 
    from P2_Code.home_cage.home_cage_extension import hc_processing

    def load_blocks(self):
        """
        Loads each block folder as a TDTData object.
        """
        block_folders = [folder for folder in os.listdir(self.experiment_folder_path)
                         if os.path.isdir(os.path.join(self.experiment_folder_path, folder))]

        for block_folder in block_folders:
            block_path = os.path.join(self.experiment_folder_path, block_folder)
            block_data = tdt.read_block(block_path)
            self.blocks[block_folder] = TDTData(block_data, block_path)

    def get_block(self, block_name):
        """
        Retrieves a specific block by its folder name.
        """
        return self.blocks.get(block_name, None)

    def list_blocks(self):
        """
        Lists all the block names available in the group.
        """
        return list(self.blocks.keys())

    def batch_process(self, remove_led_artifact=True, t=20):
        """
        Batch processes the TDT data by extracting behaviors, removing LED artifacts, and computing z-score.
        """
        for block_folder, tdt_data_obj in self.blocks.items():
            csv_file_name = f"{block_folder}.csv"
            csv_file_path = os.path.join(self.csv_base_path, csv_file_name)
            if os.path.exists(csv_file_path):
                print(f"Processing {block_folder}...")
                tdt_data_obj.extract_manual_annotation_behaviors(csv_file_path)
                # tdt_data_obj.combine_consecutive_behaviors(behavior_name='all', bout_time_threshold=2, min_occurrences=1)
                if remove_led_artifact:
                    tdt_data_obj.remove_initial_LED_artifact(t=t)
                # tdt_data_obj.smooth_signal()
                tdt_data_obj.downsample_data(N=16)
                tdt_data_obj.verify_signal()
                tdt_data_obj.compute_dff()
                tdt_data_obj.compute_zscore()

    def compute_group_psth(self, behavior_name='Pinch', pre_time=5, post_time=5, signal_type='zscore'):
        """
        Computes the group-level PSTH for the specified behavior and stores it in self.group_psth.
        """
        psths = []
        for block_folder, tdt_data_obj in self.blocks.items():
            # Compute individual PSTH
            psth_df = tdt_data_obj.compute_psth(behavior_name, pre_time=pre_time, post_time=post_time, signal_type=signal_type)
            psths.append(psth_df.mean(axis=0).values)

        # Ensure all PSTHs have the same length by trimming or padding
        min_length = min(len(psth) for psth in psths)
        trimmed_psths = [psth[:min_length] for psth in psths]

        # Convert list of arrays to a DataFrame
        self.group_psth = pd.DataFrame(trimmed_psths).mean(axis=0)
        self.group_psth.index = psth_df.columns[:min_length]

    def plot_group_psth(self, behavior_name='Pinch', pre_time=5, post_time=5, signal_type='zscore'):
        """
        Plots the group-level PSTH for the specified behavior, including the variance (standard deviation) between trials.
        """
        if self.group_psth is None:
            raise ValueError("Group PSTH has not been computed. Call compute_group_psth first.")

        psths_mean = []
        psths_std = []
        for block_folder, tdt_data_obj in self.blocks.items():
            # Compute individual PSTH
            psth_df = tdt_data_obj.compute_psth(behavior_name, pre_time=pre_time, post_time=post_time, signal_type=signal_type)
            psths_mean.append(psth_df['mean'].values)
            psths_std.append(psth_df['std'].values)

        # Convert list of arrays to DataFrames
        psth_mean_df = pd.DataFrame(psths_mean)
        psth_std_df = pd.DataFrame(psths_std)

        # Calculate the mean and standard deviation across blocks
        group_psth_mean = psth_mean_df.mean(axis=0)
        group_psth_std = psth_std_df.mean(axis=0)

        # Ensure the index matches the time points
        time_points = psth_df.index

        plt.figure(figsize=(10, 6))
        plt.plot(time_points, group_psth_mean, label=f'Group {signal_type} Mean')
        plt.fill_between(time_points, group_psth_mean - group_psth_std, group_psth_mean + group_psth_std, 
                        color='gray', alpha=0.3, label='_nolegend_')  # Exclude from legend
        plt.xlabel('Time (s)')
        plt.ylabel(f'{signal_type}')
        plt.title(f'Group PSTH for {behavior_name}')
        plt.axvline(0, color='r', linestyle='--', label=f'{behavior_name} Onset')
        # Set x-ticks at each second
        plt.xticks(np.arange(int(time_points.min()), int(time_points.max())+1, 1))  # Ticks every second
        plt.legend()
        plt.show()




    '''********************************** PLOTTING **********************************'''
    def plot_individual_psths(self, behavior_name='Pinch', pre_time=5, post_time=5, signal_type='dFF'):
        """
        Plots the PSTH for each block individually.
        """
        rows = len(self.blocks)
        figsize = (18, 5 * rows)

        fig, axs = plt.subplots(rows, 1, figsize=figsize)
        axs = axs.flatten()
        plt.rcParams.update({'font.size': 16})

        for i, (block_folder, tdt_data_obj) in enumerate(self.blocks.items()):
            psth_df = tdt_data_obj.compute_psth(behavior_name, pre_time=pre_time, post_time=post_time, signal_type=signal_type)

            # Extract the time axis (which should be the columns of the DataFrame)
            time_axis = psth_df.index
            
            # Plot the mean PSTH with standard deviation shading
            psth_mean = psth_df['mean'].values
            psth_std = psth_df['std'].values
            
            axs[i].plot(time_axis, psth_mean, label=f'{tdt_data_obj.subject_name}', color='blue')
            axs[i].fill_between(time_axis, psth_mean - psth_std, psth_mean + psth_std, color='blue', alpha=0.3)
            
            axs[i].set_title(f'{tdt_data_obj.subject_name}: {signal_type.capitalize()} Signal with {behavior_name} Bouts')
            axs[i].set_ylabel(f'{signal_type}')
            axs[i].set_xlabel('Time (s)')
            axs[i].axvline(0, color='r', linestyle='--', label=f'{behavior_name} Onset')

            # Set x-ticks at each second for this subplot
            axs[i].set_xticks(np.arange(int(time_axis.min()), int(time_axis.max()) + 1, 1))
            axs[i].legend()

        plt.tight_layout()
        plt.show()



    def plot_individual_behavior(self, behavior_name='Pinch', plot_type='zscore', figsize=(18, 5)):
        """
        Plots the specified behavior and y-axis signal type for each processed block.
        
        Parameters:
        - behavior_name: The name of the behavior to plot.
        - plot_type: The type of signal to plot ('dFF', 'zscore', or 'raw').
        - figsize: The size of the figure.
        """
        # Determine the number of rows based on the number of blocks
        rows = len(self.blocks)
        figsize = (figsize[0], figsize[1] * rows)

        # Initialize the figure with the calculated size and adjust font size
        fig, axs = plt.subplots(rows, 1, figsize=figsize)
        axs = axs.flatten()
        plt.rcParams.update({'font.size': 16})

        # Loop over each block and plot
        for i, (block_folder, tdt_data_obj) in enumerate(self.blocks.items()):
            # Plot the behavior event using the plot_behavior_event method in the single block class
            tdt_data_obj.plot_behavior_event(behavior_name=behavior_name, plot_type=plot_type, ax=axs[i])
            subject_name = tdt_data_obj.subject_name
            axs[i].set_title(f'{subject_name}: {plot_type.capitalize()} Signal with {behavior_name} Bouts', fontsize=18)

        plt.tight_layout()
        plt.show()
