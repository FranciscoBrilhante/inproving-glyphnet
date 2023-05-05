import matplotlib.pyplot as plt
import numpy as np
import torch
import requests
from torch import Tensor
from PIL import Image
from torchvision import transforms
from functools import lru_cache

from cases import endings,junctions, corners

def class_list_to_tensor(labels: list[int], device: str):
    return torch.Tensor([labels]).long().to(device)

def load_image_to_tensor(source_path:str, device:str, unsqueeze: bool=False, ):
    """
    :return: tensor width shape (channels,height, width)
    """
    image=Image.open(source_path)
    trans=transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])
    image=trans(image)

    if unsqueeze:
        image=torch.unsqueeze(image, 0)
    image=image.to(device)

    return image

def decompose_row(image: Tensor) -> Tensor:
    """
    decompose_row partitions a tensor containing multiple glyphs in a row(width) into individual indexable glyphs
    :param image: tensor with multiple glyphs in a row, expected shape: (channels,height, width)
    :return: tensor where each column is a single glyph, shape: (n_glyphs,height,width)
    """

    height = image.size(1)
    width = image.size(2)
    n_glyphs = int(width/height)

    glyphs = torch.Tensor()
    for i in range(n_glyphs):
        glyphs = torch.cat((glyphs, image[:, :, height*i:height*i+height]))
    return glyphs


def read_cvae_log(path: str) -> list:
    """
    read_cvae_log reads log file of cvae model into array of floats
    :param path: log file path
    :return: uni-dim array with float values
    """
    with open(path, 'r') as file:
        lines = file.readlines()
        lines = lines[1:]
        lines = [line.strip().split(',') for line in lines]
        lines = [(int(line[-2]), float(line[-1])) for line in lines]
        return lines


def download_file(url: str, target_filename: str,):
    """
    download_file downloads a file defined by its url and saves it in the desired path
    :param url: remote file path
    :param target_filename: path where file should be saved
    """
    response = requests.get(url)
    open(target_filename, "wb").write(response.content)


def plot_lines(lines: list, fig_size: tuple, smooth_strength: list, colors: list, legends: list, fig_name: str, dpi:int,xlabel, ylabel, title):
    """
    plot_lines plots multiple data in a line graph
    :param lines: list where each element is a list of float values in ordered sequence
    :param smooth_strength: list of int values for each line
    :param colors: list of colors for each line
    :param legends: list of legends for each line
    """

    plt.figure(figsize=fig_size)
    fig, ax = plt.subplots(figsize=fig_size)

    for i in range(lines):
        kernel_size = smooth_strength[i]
        kernel = np.ones(kernel_size) / kernel_size
        plt.plot([x[0]/9121 for x in lines[i]], np.convolve([x[1] for x in lines[i]], kernel, mode='same'), colors[i], label=legends[i])

    plt.grid(color=(0.01, 0.01, 0.01), alpha=0.1)
    plt.rcParams['grid.color'] = (0.5, 0.2, 0.2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(frameon=False)
    plt.savefig(fig_name, dpi=dpi)

    # Show the plot
    plt.show()


class ThresholdTransform(object):
    def __init__(self, thr_255):
        self.thr = thr_255 / 255.  # input threshold for [0..255] gray level, convert to [0..1]

    def __call__(self, x):
        return (x > self.thr).to(x.dtype)  # do not change the data type

class InvertTransform(object):
    def __call__(self,x):
        return (x<1).to(x.dtype)


def get_mask(font: Tensor, default_font: Tensor, apply_blur: bool, lower_limit:float):
    mask=torch.nn.functional.relu(torch.sub(default_font,font))
    if apply_blur:
        mask=transforms.GaussianBlur(9,2)(mask)
    
    mask=mask*(1-lower_limit)+lower_limit
    return  mask


def detect_poinsts(glyph: Tensor, apply_blur:bool, skeleton_value:float, lower_limit:float):
    """
    param glyph: tensor with skeleton with expected shape: (64x64)
    """
    if glyph.size()!=(64,64):
        raise Exception()

    mask=torch.zeros((70,70))
    glyph=transforms.Pad(1,0,'constant')(glyph)

    for x in range(0, 66):
        for y in range(0,66):
            zone=glyph[x:x+3,y:y+3]
            if is_area_of_interest(str(zone.int().tolist())):
                mask[x-3:x+6,y-3:y+6]=1
    mask=mask[3:67,3:67]
    glyph=glyph[1:65,1:65]
    mask=torch.add(mask,glyph*skeleton_value).unsqueeze(0)
    if apply_blur:
        mask=transforms.GaussianBlur(9,sigma=2)(mask)
    mask=mask*(1-lower_limit)+lower_limit
    return mask

@lru_cache
def is_area_of_interest(area):
    cases=junctions()+corners()+endings()
    for case in cases:
        if str(case)==area:
            return True
    return False