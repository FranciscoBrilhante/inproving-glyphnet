import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path
from shutil import copy2
from PIL import Image, ImageFont, ImageDraw
import shutil

from torchvision import transforms
from torchvision.utils import save_image
from torchvision.io import read_image
from torch.nn import MSELoss
import torch.nn as nn
import torch 

from utils import load_image_to_tensor, ThresholdTransform, InvertTransform, get_mask, detect_poinsts,download_file,decompose_row
from thinning import get_skeleton


splits=['train','test','val']

rename_folder="/home/francisco/dataset/skeletons"
target_folder="/home/francisco/dataset/skeletonsAccurate"
original_folder="/home/francisco/mc-gan/datasets/Capitals64"

Path(target_folder).mkdir(parents=True, exist_ok=True)
for split in splits:
    font_names=os.listdir(os.path.join(original_folder,split))
    for i,font in tqdm(enumerate(font_names)):
        for k in range(26):
            Path(os.path.join(target_folder,split,str(k))).mkdir(parents=True, exist_ok=True)

            source_file=os.path.join(rename_folder,split,str(k),f'{i}.png')
            target_file=os.path.join(target_folder,split,str(k),font)
            command=f"cp \"{source_file}\" \"{target_file}\""
            os.system(command)
        
    
