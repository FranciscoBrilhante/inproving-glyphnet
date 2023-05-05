from tqdm import tqdm
import os
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw
import random 
import string

from torchvision import transforms
from torchvision.utils import save_image
import torch 
from utils import  InvertTransform


def build_daniel_datset():
    class Found(Exception): pass
    root_path='/home/francisco/dataset/daniel_fonts'
    errors=0
    Path('/home/francisco/dataset/daniel26/train').mkdir(parents=True,exist_ok=True)
    Path('/home/francisco/dataset/daniel26/test').mkdir(parents=True,exist_ok=True)
    
    fonts=os.listdir(root_path)
    for elem in tqdm(fonts):
        extension=elem[-3:]
        
        if extension in ['ttf','TTF','otf']:
            font_path=os.path.join(root_path,elem)

            glyphs={}
            try:
                for i,letter in enumerate(list(string.ascii_uppercase)):
                   
                    glyph=Image.new('RGB',size=(128,128))
                    draw = ImageDraw.Draw(glyph)
                    # use a truetype font
                    font = ImageFont.truetype(font_path, 110)

                    draw.text((0, 0), letter, font=font, anchor='lt')

                    trans=transforms.Compose([
                        transforms.Grayscale(1),
                        transforms.ToTensor(),
                        ])
                    
                    glyph=trans(glyph).squeeze(0)
                    start_height_idx,start_width_idx=0,0
                    end_height_idx,end_width_idx=127,127
                    
                    try:
                        for row in range(glyph.size(0)):
                            if torch.sum(glyph[row,:])!=0:
                                start_height_idx=row
                                raise Found            
                    except Found:
                        pass
                    try:
                        for row in range(glyph.size(0)-1,-1,-1):
                            if torch.sum(glyph[row,:])!=0:
                                end_height_idx=row
                                raise Found    
                    except Found:
                        pass
                    try:
                        for col in range(glyph.size(1)):
                            if torch.sum(glyph[:,col])!=0:
                                start_width_idx=col
                                raise Found            
                    except Found:
                        pass
                    try:
                        for col in range(glyph.size(1)-1,-1,-1):
                            if torch.sum(glyph[:,col])!=0:
                                end_width_idx=col
                                raise Found    
                    except Found:
                        pass

                    glyph=glyph[start_height_idx:end_height_idx,start_width_idx:end_width_idx]
                    glyph=InvertTransform()(glyph).unsqueeze(0)
                    glyph=transforms.Resize(size=(60,60),interpolation=transforms.InterpolationMode.BILINEAR)(glyph)
                    glyph=transforms.Pad(padding=(2,2),fill=1)(glyph)

                    if torch.sum(glyph)>0.85*64*64:
                        raise Exception
                    glyphs[letter]=(glyph,elem[:-4])
                
                final_image=torch.ones(size=(1,64,64*26))
                for i,letter in enumerate(list(string.ascii_uppercase)):
                    final_image[0,:,i*64:i*64+64]=glyphs[letter][0]
                
                if random.random()>0.9:
                    save_image(final_image,os.path.join('/home/francisco/dataset/daniel26/test',glyphs[letter][1]+'.png'))
                else:
                    save_image(final_image,os.path.join('/home/francisco/dataset/daniel26/train',glyphs[letter][1]+'.png'))
                
            except Exception as e:
                errors+=1
    print(errors)

if __name__=='__main__':
    build_daniel_datset()