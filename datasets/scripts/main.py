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


def import_capitals64(source_folder: str, target_folder: str, class_folders: bool = True):
    Path(target_folder).mkdir(parents=True, exist_ok=True)

    splits = ["train", "test", "val"]
    for split_label in splits:
        split_path = os.path.join(source_folder, split_label)
        split_list = [os.path.join(split_path, filename) for filename in os.listdir(split_path)]
        for i, filename in tqdm(enumerate(split_list)):
            glyphs = load_image_to_tensor(filename)
            glyphs = decompose_row(glyphs)
            for j in range(glyphs.size(0)):
                if class_folders:
                    image_path = os.path.join(target_folder, split_label, str(j), f"{i}.png")
                else:
                    image_path = os.path.join(target_folder, split_label, f"{j}_{i}.png")

                Path(image_path).parent.mkdir(parents=True, exist_ok=True)
                save_image(glyphs[j, :], image_path)


def import_capitals64_as_is(source_folder: str, target_folder: str):
    Path(target_folder).mkdir(parents=True, exist_ok=True)

    splits = ["train", "test", "val"]
    for split_label in splits:
        split_path = os.path.join(source_folder, split_label)
        split_list = [os.path.join(split_path, filename) for filename in os.listdir(split_path)]
        for i, filename in tqdm(enumerate(split_list)):
            glyphs = load_image_to_tensor(filename)
            image_path = os.path.join(target_folder, split_label, '0', f"{i}.png")

            Path(image_path).parent.mkdir(parents=True, exist_ok=True)
            save_image(glyphs, image_path)


def import_capitals64_original(source_folder: str, target_folder: str):
    Path(target_folder).mkdir(parents=True, exist_ok=True)

    splits = ["train", "test", "val"]
    for split_label in splits:
        split_path = os.path.join(source_folder, split_label)
        split_list = [os.path.join(split_path, filename) for filename in os.listdir(split_path)]
        for i, filename in tqdm(enumerate(split_list)):
            glyphs = load_image_to_tensor(filename)
            image_path = os.path.join(target_folder, split_label, f"{i}.png")

            Path(image_path).parent.mkdir(parents=True, exist_ok=True)
            save_image(glyphs, image_path)


def unzip_dataset(source_folder: str, target_folder: str):
    splits = ["train", "test", "val"]
    for split_label in splits:
        total_fonts = len(os.listdir(os.path.join(source_folder, split_label, "0")))
        for i in tqdm(range(total_fonts)):
            for j in range(26):
                source_path = os.path.join(source_folder, split_label, f'{j}', f'{i}.png')
                target_path = os.path.join(target_folder, split_label, f'{j}_{i}.png')
                Path(target_path).parent.mkdir(parents=True, exist_ok=True)
                copy2(source_path, target_path)


def download_fonts(url_list: list, save_path: str):
    Path(save_path).mkdir(parents=True, exist_ok=True)

    total = len(url_list)
    error = 0
    success = 0
    for i, url in tqdm(enumerate(url_list)):
        extension = url.split('.')[-1].strip()
        target_filename = f"{save_path}/{i}.{extension}"
        try:
            if extension != "ttf":
                raise Exception()

            download_file(url=url, target_filename=target_filename)
            success += 1
        except Exception:
            error += 1
    print(f"Total:{total} Error:{error} Success:{success}")


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
            out = loss(fake_image, real_image)
            mean_mse += out
            total += 1

    return mean_mse.item()/total


def compute_fid(real_folder: str, fake_folder: str, batch_size: int) -> float:
    stream = os.popen(f'python -m pytorch_fid {real_folder} {fake_folder} --batch-size {batch_size}')
    output = stream.read()
    return output


def build_skeleton_masks(original_folder: str, target_folder: str, class_folders: bool = True):
    Path(target_folder).mkdir(parents=True, exist_ok=True)

    splits = ["train", "test", "val"]
    for split_label in splits:
        split_path = os.path.join(original_folder, split_label)
        split_list = [os.path.join(split_path, filename) for filename in os.listdir(split_path)]
        for i, filename in tqdm(enumerate(split_list)):
            glyphs = load_image_to_tensor(filename)
            glyphs = decompose_row(glyphs)
            for j in range(glyphs.size(0)):
                if class_folders:
                    image_path = os.path.join(target_folder, split_label, str(j), f"{i}.png")
                else:
                    image_path = os.path.join(target_folder, split_label, f"{j}_{i}.png")

                Path(image_path).parent.mkdir(parents=True, exist_ok=True)
                glyph = glyphs[j, :]
                skeleton = get_skeleton(glyph,False, lower_limit=0)
                #mask=detect_poinsts(skeleton, True, 0.4, 0.5)
                save_image(skeleton, image_path)


def test_skeleton(original_glyph: str, save_folder: str, apply_blur: bool, lower_limit: float):
    glyph = load_image_to_tensor(original_glyph)
    skeleton = get_skeleton(glyph, apply_blur, lower_limit)
    mask=detect_poinsts(skeleton, True, 0.4, 0.5)
    save_image(mask, os.path.join(save_folder,"mask.png"))
    save_image(skeleton, os.path.join(save_folder,"skeleton.png"))
    save_image(glyph, os.path.join(save_folder,"original.png"))


def test_original_mask(original_font: str, default_font: str, save_folder:str, apply_blur: bool, lower_limit=0.8):
    font = load_image_to_tensor(original_font)
    default_font=load_image_to_tensor(default_font)
    mask=get_mask(font,default_font, apply_blur,lower_limit)
    
    save_image(font, os.path.join(save_folder,"original.png"))
    save_image(mask, os.path.join(save_folder,"mask.png"))
    save_image(default_font, os.path.join(save_folder,"default_font.png"))


if __name__ == "__main__":
    # download google fonts
    """
    with open("/home/francisco/dataset/glyphazzn_urls.txt", "r") as file1:
        read_content = file1.readlines()
        lines=[line.strip().split(",")[-1].strip() for line in read_content]
    download_fonts(lines,"/home/francisco/dataset/glyphazzn")"""

    # import capitals64 dataset from original repo
    # import_capitals64("/home/francisco/mc-gan/datasets/Capitals64","/home/francisco/dataset/capitals64",True)

    # import capitals64 dataset from original repo in a flat folder structure
    # import_capitals64("/home/francisco/mc-gan/datasets/Capitals64","/home/francisco/dataset/capitals64Flat",False)

    #unzip_dataset("/home/francisco/dataset/capitals64", "/home/francisco/dataset/capitals64Unizp")

    #compute_mse("/home/francisco/dataset/capitals64/test", "/home/francisco/logs/cinvae_cond/Test")

    #value=compute_mse("/home/francisco/mc-gan/datasets/Capitals64/test", "/home/francisco/mc-gan/results/GlyphNet_3_10000/test_600/images",False, True)
    # print(value)

    #value=compute_fid("/home/francisco/mc-gan/datasets/Capitals64/test", "/home/francisco/mc-gan/results/GlyphNet_3_10000/test_600/images", 1)
    # print(value)

    # import capitals64 dataset from original repo as it is with an artificial class 0 for every sample
    # import_capitals64_as_is("/home/francisco/mc-gan/datasets/Capitals64","/home/francisco/dataset/capitals64Same")

    #value=compute_mse("/home/francisco/mc-gan/datasets/Capitals64/test", "/home/francisco/logs/fixed/Test",class_folders=False,fake_suffix=False)
    # print(value)

    #value=compute_fid("/home/francisco/dataset/capitals64Original/test", "/home/francisco/logs/denoiser_3/Test",batch_size=64)
    #print(value)

    # import capitals64 dataset from original repo as it is
    #import_capitals64_original("/home/francisco/mc-gan/datasets/Capitals64","/home/francisco/dataset/capitals64Original")

    #build_skeleton_masks('/home/francisco/dataset/capitals64Original', '/home/francisco/dataset/SkeletonMasks')

    #test_skeleton('/home/francisco/dataset/capitals64/test/0/3.png','/home/francisco/dataset/', apply_blur= False, lower_limit= 0)

    #test_original_mask('/home/francisco/dataset/capitals64Original/test/3.png','/home/francisco/mc-gan/datasets/Capitals64/BASE/Code New Roman.0.0.png','/home/francisco/dataset/',apply_blur= True,lower_limit= 0.5)

    """
    Path('/home/francisco/dataset/capitals64Original').mkdir(parents=True, exist_ok=True)

    splits = ["train", "test", "val"]
    for split_label in splits:
        split_path = os.path.join('/home/francisco/dataset/capitals64Original', split_label)
        split_list = [os.path.join(split_path, filename) for filename in os.listdir(split_path)]
        for i, filename in tqdm(enumerate(split_list)):
            skel = torch.Tensor()
            for j in range(26):
                image_path = os.path.join('/home/francisco/dataset/skeletons', split_label, str(j), f"{i}.png")
                skeleton=load_image_to_tensor(image_path,False,False).squeeze(0)
                mask=detect_poinsts(skeleton, True, 0.4, 0.5)
                skel=torch.cat((skel,mask),2)

            final_path=os.path.join('/home/francisco/dataset/masks',split_label,'0', os.path.basename(filename))
            Path(final_path).parent.mkdir(parents=True, exist_ok=True)
            save_image(skel, final_path)
    """
    #build_daniel_datset()
    
    """
    target_folder='/home/francisco/dataset/combined'
    source_folder='/home/francisco/mc-gan/datasets/Capitals64'
    Path('/home/francisco/dataset/combined/train').mkdir(parents=True, exist_ok=True)
    Path('/home/francisco/dataset/combined/test').mkdir(parents=True, exist_ok=True)

    splits = ["train", "test", "val"]
    for split_label in splits:
        split_path = os.path.join(source_folder, split_label)
        split_list = [os.path.join(split_path, filename) for filename in os.listdir(split_path)]
        for i, filename in tqdm(enumerate(split_list)):

            fontname=os.path.splitext(os.path.basename(filename))[0]
            if split_label=="train" or split_label=="val":
                shutil.copy2(filename, os.path.join('/home/francisco/dataset/combined/train',fontname+'.png'))
            else:
                shutil.copy2(filename, os.path.join('/home/francisco/dataset/combined/test',fontname+'.png'))

    
    target_folder='/home/francisco/dataset/combined'
    source_folder='/home/francisco/dataset/daniel26'
    Path('/home/francisco/dataset/combined/train').mkdir(parents=True, exist_ok=True)
    Path('/home/francisco/dataset/combined/test').mkdir(parents=True, exist_ok=True)

    splits = ["train", "test"]
    for split_label in splits:
        split_path = os.path.join(source_folder, split_label)
        split_list = [os.path.join(split_path, filename) for filename in os.listdir(split_path)]
        for i, filename in tqdm(enumerate(split_list)):

            fontname=os.path.splitext(os.path.basename(filename))[0]
            if split_label in ['train','val']:
                shutil.copy2(filename, os.path.join('/home/francisco/dataset/combined/train',fontname+'.png'))
            else:
                shutil.copy2(filename, os.path.join('/home/francisco/dataset/combined/test',fontname+'.png'))
   
   
    
    
    Path('/home/francisco/dataset/daniel26/all').mkdir(parents=True, exist_ok=True)
    target_folder='/home/francisco/dataset/daniel26/all'
    source_folder='/home/francisco/dataset/daniel26'
    splits = ["train", "test"]
    for split_label in splits:
        split_path = os.path.join(source_folder, split_label)
        split_list = [os.path.join(split_path, filename) for filename in os.listdir(split_path)]
        for i, filename in tqdm(enumerate(split_list)):

            fontname=os.path.splitext(os.path.basename(filename))[0]
            shutil.copy2(filename, os.path.join('/home/francisco/dataset/daniel26/all',fontname+'.png'))
   
    
    lista=os.listdir('/home/francisco/dataset/daniel26/all')
    print(len(lista))
    """