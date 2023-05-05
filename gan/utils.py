from torch.utils.tensorboard import SummaryWriter

import os
import torch
import json
import random
from tqdm import tqdm
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

# pytorch modules
from torch.nn import MSELoss
from torchvision.utils import save_image

# local modules
from data.common import load_image_to_tensor


def create_dirs_logs(config: dict):
    log_dir = os.path.join(config['path_log'], config['logname'])
    for path_target in ["", 'test', 'Checkpoints']:
        Path(os.path.join(log_dir, path_target)).mkdir(exist_ok=True, parents=True)
    with open(f"{log_dir}/config.json", 'w') as fp:
        json.dump(config, fp,sort_keys=True, indent=2)


def update_telemetry(writer: SummaryWriter, t, step: int, loss_dict: dict, lr: float | None):
    loss_value = round(loss_dict['loss_gen'].item(), 5)
    t.set_description(f"Generator Loss: {loss_value}")
    writer.add_scalar('gen_loss/train', loss_dict['loss_gen'], step)
    writer.add_scalar('dis_loss/train', loss_dict['loss_disc'], step)
    if lr:
        writer.add_scalar('lr/train', lr, step)


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def compute_mse(real_folder: str, fake_folder: str, class_folders: bool = True, fake_suffix: bool = True) -> float:
    mean_mse = 0
    total = 0

    if class_folders:
        font_list = os.listdir(os.path.join(real_folder, "0"))
        for i in tqdm(range(len(font_list))):
            for j in range(26):
                real_image = load_image_to_tensor(os.path.join(real_folder, f'{j}', font_list[i]))
                fake_image = load_image_to_tensor(os.path.join(fake_folder, f'{j}', font_list[i]))

                loss = MSELoss()
                out = loss(fake_image, real_image)
                mean_mse += out
                total += 1
    else:
        font_list = os.listdir(real_folder)
        for i in tqdm(range(len(font_list))):
            real_path = os.path.join(real_folder, font_list[i])
            if fake_suffix:
                fake_path = os.path.join(fake_folder, font_list[i][:-4]+"_fake_B.png")
            else:
                fake_path = os.path.join(fake_folder, font_list[i])
            real_image = load_image_to_tensor(real_path)
            fake_image = load_image_to_tensor(fake_path)

            loss = MSELoss()
            out = loss(fake_image, real_image).tolist()
            mean_mse += out
            total += 1

    return mean_mse/total


def compute_fid(real_folder: str, fake_folder: str, batch_size: int, device:str) -> float:
    stream = os.popen(f'python -m pytorch_fid {real_folder} {fake_folder} --batch-size {batch_size} --device {device}')
    output = stream.read()
    return float(output.strip().split(':')[1].strip())


def save_list_table(path_file:str ,list:list):
    with open(path_file,'w') as f:
        for elem in list:
            for col in elem:
                f.write(f'{col}\t')
            f.write('\n')

def analize_table(path_log:str, filename:str, tails_size:int, original_dataset:str, generated_dataset:str):
    f=open(os.path.join(path_log,filename),'r')
    lines=f.readlines()
    data=[line.strip().split('\t') for line in lines]
    a=[line[0] for line in data]
    b=[float(line[1]) for line in data]

    image=torch.empty(size=(1,tails_size*2*64*3+2*64,64*26))

    a=np.array(a)
    b=np.array(b)
    
    print('-------------mean mse--------------')
    print(np.mean(b))
    
    indxs=np.argsort(b)
    j=0
    print('-------------Lowest Values--------------')
    for i,elem in enumerate(indxs[0:tails_size]):
        print(a[elem],end='\t')
        print(b[elem],end='\n')

        elem_a=load_image_to_tensor(os.path.join(original_dataset,f'{a[elem]}.png'),unsqueeze=False,device='cpu')
        elem_b=load_image_to_tensor(os.path.join(generated_dataset,f'{a[elem]}.png'),unsqueeze=False,device='cpu')
        image[0,j*64:j*64+64,:]=elem_a
        j+=1
        image[0,j*64:j*64+64,:]=elem_b
        j+=1
    
    image[0,j*64:j*64+64,:]=torch.zeros(size=(1,64,64*26))
    j+=1
    for i,elem in enumerate(indxs[round(len(indxs)/2)-tails_size:round(len(indxs)/2)]):
        elem_a=load_image_to_tensor(os.path.join(original_dataset,f'{a[elem]}.png'),unsqueeze=False,device='cpu')
        elem_b=load_image_to_tensor(os.path.join(generated_dataset,f'{a[elem]}.png'),unsqueeze=False,device='cpu')
        image[0,j*64:j*64+64,:]=elem_a
        j+=1
        image[0,j*64:j*64+64,:]=elem_b
        j+=1

    image[0,j*64:j*64+64,:]=torch.zeros(size=(1,64,64*26))
    j+=1

    print('-------------Highest Values--------------')
    for i,elem in enumerate(indxs[-tails_size:]):
        print(a[elem],end='\t')
        print(b[elem],end='\n')

        elem_a=load_image_to_tensor(os.path.join(original_dataset,f'{a[elem]}.png'),unsqueeze=False,device='cpu')
        elem_b=load_image_to_tensor(os.path.join(generated_dataset,f'{a[elem]}.png'),unsqueeze=False,device='cpu')
        image[0,j*64:j*64+64,:]=elem_a
        j+=1
        image[0,j*64:j*64+64,:]=elem_b
        j+=1

    
    save_image(image,os.path.join(path_log,'visuals.png'))


def plot_histogram(path_plot:str, data:list[float], bins, title:str, ylegend:str, xlegend:str):
    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(data, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlegend)
    ax.set_ylabel(ylegend)

    fig.savefig(path_plot)


def draw_red_square(image, startX:int , startY:int, width:int):
    image[0,startX:startX+width,startY:startY+2]=1
    image[1,startX:startX+width,startY:startY+2]=0
    image[2,startX:startX+width,startY:startY+2]=0

    image[0,startX:startX+width,startY+width-2:startY+width]=1
    image[1,startX:startX+width,startY+width-2:startY+width]=0
    image[2,startX:startX+width,startY+width-2:startY+width]=0

    image[0,startX:startX+2,startY:startY+width]=1
    image[1,startX:startX+2,startY:startY+width]=0
    image[2,startX:startX+2,startY:startY+width]=0

    image[0,startX+width-2:startX+width,startY:startY+width]=1
    image[1,startX+width-2:startX+width,startY:startY+width]=0
    image[2,startX+width-2:startX+width,startY:startY+width]=0

    return image

