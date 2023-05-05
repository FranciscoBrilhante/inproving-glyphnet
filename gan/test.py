import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import random
import numpy
import string

# pytorch modules
import torch
from torch import load, nn, optim
from torchvision.utils import save_image
from torchvision import transforms
from torch.nn import MSELoss

from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *

# local modules
from configs import *
from utils import create_dirs_logs, compute_fid, compute_mse, save_list_table, analize_table, plot_histogram, draw_red_square
from data.common import unfold_image, fold_image, load_image_to_tensor
from models import glyphgan as model


def load_inputs(source_path: str, device, n_glyphs_visible: int, glyphs_visible: list[int] | None):
    """
    :return: tensor width shape (channels,height, width)
    """
    image = Image.open(source_path)
    trans = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])
    image = trans(image)
    image = image.to(device)
    image = fold_image(image)

    if glyphs_visible == None:
        idx_to_hide = random.sample(list(range(26)), 26-n_glyphs_visible)
    else:
        idx_to_hide = [i for i in range(26) if i not in glyphs_visible]

    for idx in idx_to_hide:
        image[idx, :, :] = 1

    image = image.unsqueeze(0).detach().to(device)
    return image


def test_default(config):
    create_dirs_logs(config)

    path_log = os.path.join(config['path_log'], config['logname'])
    path_data_test = config['data_path_test']
    path_checkpoint = os.path.join(path_log, "Checkpoints", f"last.pt")

    glyphs_visible = config.get('glyphs_visible', None)
    device = config['device']
    n_glyphs_visible = config['n_visible']

    model = load(path_checkpoint)
    model.eval()
    model.to(device)

    font_list = os.listdir(os.path.join(path_data_test))
    with torch.no_grad():
        for i in tqdm(range(len(font_list))):
            image = load_inputs(os.path.join(path_data_test, font_list[i]), device=device, n_glyphs_visible=n_glyphs_visible, glyphs_visible=glyphs_visible)
            output = model.forward(image)
            output = unfold_image(output[0])
            path_to_save_image = Path(os.path.join(path_log, 'test', font_list[i]))
            path_to_save_image.parent.mkdir(exist_ok=True, parents=True)
            save_image(output, path_to_save_image)


def test_with_metrics(config, text_file_name):
    create_dirs_logs(config)
    path_log = os.path.join(config['path_log'], config['logname'])
    path_data_test = config['data_path_test']
    path_checkpoint = os.path.join(path_log, "Checkpoints", f"last.pt")
    path_table = os.path.join(path_log, text_file_name)
    device = config['device']
    n_glyphs_visible = config['n_visible']
    glyphs_visible = config.get('glyphs_visible', None)

    model = load(path_checkpoint, map_location=torch.device(device))
    model.eval()

    font_list = os.listdir(os.path.join(path_data_test))
    data = []

    meanMSE = 0
    meanSSIM = 0
    for i in tqdm(range(len(font_list))):
        inputs = load_inputs(os.path.join(path_data_test, font_list[i]), device=device, n_glyphs_visible=n_glyphs_visible, glyphs_visible=glyphs_visible)
        output = model.forward(inputs)
        output = unfold_image(output[0])

        target = load_image_to_tensor(os.path.join(path_data_test, font_list[i]), unsqueeze=False, device=device)

        # mse
        mseLoss = MSELoss()
        mse = mseLoss(output, target).tolist()
        meanMSE += mse

        ssimLoss = SSIM(data_range=1.0)
        ssimLoss.update((output.unsqueeze(0), target.unsqueeze(0)))
        ssim = ssimLoss.compute()
        ssimLoss.reset()
        meanSSIM += ssim

        data.append([font_list[i][:-4], mse, ssim])

        path_to_save_image = Path(os.path.join(path_log, 'test', font_list[i]))
        path_to_save_image.parent.mkdir(exist_ok=True, parents=True)
        save_image(output, path_to_save_image)

    fid = compute_fid(config['data_path_test'], os.path.join(path_log, 'test'), batch_size=64, device=device)
    data = [[meanMSE/len(font_list), meanSSIM/len(font_list), fid]]+data
    save_list_table(path_table, data)



if __name__ == "__main__":
    pass
