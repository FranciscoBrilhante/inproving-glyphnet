"""
===========================
@Author  : Linbo<linbo.me>
@Version: 1.0    25/10/2014
This is the implementation of the 
Zhang-Suen Thinning Algorithm for skeletonization.
===========================
"""
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

from torch import Tensor
from torchvision import transforms
from torchvision.utils import save_image

from utils import load_image_to_tensor, ThresholdTransform,InvertTransform

def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9

def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

def zhangSuen(image: Tensor):
    "image shape needs to be (h,w)"
    "the Zhang-Suen Thinning Algorithm"
    Image_Thinned = image.clone()  # deepcopy to protect the original image
    changing1 = changing2 = 1        #  the points to be removed (set as 0)
    while changing1 or changing2:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.size()               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions 
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2: S(P1)=1  
                    P2 * P4 * P6 == 0  and    # Condition 3   
                    P4 * P6 * P8 == 0):         # Condition 4
                    changing1.append((x,y))
        for x, y in changing1: 
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and        # Condition 0
                    2 <= sum(n) <= 6  and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    P2 * P4 * P8 == 0 and       # Condition 3
                    P2 * P6 * P8 == 0):            # Condition 4
                    changing2.append((x,y))    
        for x, y in changing2: 
            Image_Thinned[x][y] = 0
    return Image_Thinned

def get_skeleton(glyph: Tensor, apply_blur: bool, lower_limit: float):
    trans = transforms.Compose([
        ThresholdTransform(thr_255=254),
        InvertTransform()
    ])
    glyph = trans(glyph).squeeze(0)
    skeleton = zhangSuen(glyph)
    if apply_blur:
        skeleton = transforms.GaussianBlur(9, 2)(skeleton.unsqueeze(0))
    skeleton=skeleton*(1-lower_limit)+lower_limit
    return skeleton

def main():

    class ThresholdTransform(object):
        def __init__(self, thr_255):
            self.thr = thr_255 / 255.  # input threshold for [0..255] gray level, convert to [0..1]

        def __call__(self, x):
            return (x > self.thr).to(x.dtype)  # do not change the data type

    class InvertTransform(object):
        def __call__(self,x):
            return (x<1).to(x.dtype)

    path_original ='/home/francisco/dataset/capitals64/train/1/0.png'
    path_target='/home/francisco/dataset/test.png'
    original = load_image_to_tensor(path_original,False,False)
    trans=transforms.Compose([
        ThresholdTransform(thr_255=254),
        InvertTransform()
    ])
    original=trans(original).squeeze(0)

    skeleton = zhangSuen(original)
    skeleton=InvertTransform()(skeleton)
    save_image(skeleton,path_target)

if __name__=="__main__":
    main()