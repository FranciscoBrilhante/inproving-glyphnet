from tqdm import tqdm
import os
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw

from torchvision import transforms
from torchvision.utils import save_image
import torch 
from utils import  InvertTransform
import pandas
import string
import numpy as np
import random

def build_tmnist_datset(path_csv: str, target_folder:str):
    train_folder=os.path.join(target_folder,'train')
    test_folder=os.path.join(target_folder,'test')
    Path(train_folder).mkdir(parents=True, exist_ok=True)
    Path(test_folder).mkdir(parents=True, exist_ok=True)

    capital_letters = list(string.ascii_uppercase)
    df=pandas.read_csv(path_csv)

    while len(df.index)>0:
        fontname=df.iloc[0]['names']
        font_rows=df.loc[df['names']==fontname]
        font_rows_capitals=font_rows[font_rows['labels'].isin(capital_letters)]

        if len(font_rows_capitals.index)==26 and font_rows_capitals['labels'].duplicated().any()==False:
            image = torch.Tensor()
            for letter in capital_letters:
                values=font_rows_capitals.loc[font_rows_capitals['labels']==letter].iloc[0,2:].tolist()
                values=(np.array(values)/255).reshape((28,28))
                values=-torch.Tensor(values).unsqueeze(0)+1
                values=transforms.Resize(size=(64,64))(values)
                image=torch.cat((image,values),2)

            if random.random()>0.1:
                save_image(image, os.path.join(train_folder,f'{fontname}.png'))
            else:
                save_image(image, os.path.join(test_folder,f'{fontname}.png'))
            
        df = df.drop(font_rows.index)


if __name__=='__main__':
    path_csv='/home/francisco/dataset/tmnist/94_character_TMNIST.csv'
    target_folder='/home/francisco/dataset/tmnist/'
    build_tmnist_datset(path_csv,target_folder)