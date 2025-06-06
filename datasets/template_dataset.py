
import math
import os
import random

import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage

class TemplateDataset(Dataset):

    def __init__(self, root_dir, img_list, input_D, input_H, input_W, phase):
        with open(img_list, 'r') as f:
            self.img_list = [line.strip() for line in f]
        print("Processing {} datas".format(len(self.img_list)))
        self.root_dir = root_dir
        self.input_D = input_D
        self.input_H = input_H
        self.input_W = input_W
        self.phase = phase
    
    def __len__(self):
        return len(self.img_list)

# NOT DONE