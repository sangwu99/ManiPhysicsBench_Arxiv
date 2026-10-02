import numpy as np
from PIL import Image

def install_release_packing(image_size):
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
    original = LeRobotSingleDataset._pack_sample
    size = tuple(image_size)

    def pack(self, data):
        sample = original(self, data)
        sample['image'] = [np.array(Image.fromarray(im).resize(size)) for im in sample['image']]
        return sample
    LeRobotSingleDataset._pack_sample = pack
