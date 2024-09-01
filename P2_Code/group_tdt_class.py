import os
import tdt
from single_tdt_class import TDTData   

class GroupTDTData:
    def __init__(self, experiment_folder_path):
        self.experiment_folder_path = experiment_folder_path
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

# Example usage
experiment_folder = r"D:\Pilot_2\Cohort_2\Synapse\Tanks\5_6_24_Pinch_Test_P2-240506-084710"
group_data = GroupTDTData(experiment_folder)

# Access a specific block
block_name = 'n1-240506-101729'
block_data_obj = group_data.get_block(block_name)

# List all loaded blocks
print(group_data.list_blocks())