import os
import tdt
import matplotlib.pyplot as plt
from single_tdt_class import TDTData

class GroupTDTData:
    def __init__(self, experiment_folder_path, csv_base_path):
        """
        Initializes the GroupTDTData object with paths.
        """
        self.experiment_folder_path = experiment_folder_path
        self.csv_base_path = csv_base_path
        self.blocks = {}

        self.load_blocks()

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

    def batch_process(self, behavior_name='Pinch', remove_led_artifact=True, t=10):
        """
        Batch processes the TDT data by extracting behaviors, removing LED artifacts, and computing z-score.
        """
        for block_folder, tdt_data_obj in self.blocks.items():
            csv_file_name = f"{block_folder}.csv"
            csv_file_path = os.path.join(self.csv_base_path, csv_file_name)
            if os.path.exists(csv_file_path):
                print("found csv")
                tdt_data_obj.extract_manual_annotation_behaviors(csv_file_path)
                if remove_led_artifact:
                    tdt_data_obj.remove_initial_LED_artifact(t=t)
                tdt_data_obj.smooth_signal()
                tdt_data_obj.verify_signal()
                tdt_data_obj.compute_dff()
                tdt_data_obj.compute_zscore()

    def plot_behavior(self, behavior_name='Pinch', plot_type='zscore', figsize=(18, 5)):
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
            tdt_data_obj.plot_behavior_event(behavior_name=behavior_name, plot_type=plot_type)
            subject_name = tdt_data_obj.subject_name
            axs[i].set_title(f'{subject_name}: {plot_type.capitalize()} Signal with {behavior_name} Bouts', fontsize=18)

        plt.tight_layout()
        plt.show()
