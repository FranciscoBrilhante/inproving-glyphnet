from PIL import Image

# pytorch modules
import torch
from torch import Tensor
from torchvision import transforms

def load_image_to_tensor(source_path: str, unsqueeze: bool, device: str):
    """
    :return: tensor width shape (channels,height, width)
    """
    image = Image.open(source_path)
    trans = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])
    image = trans(image)

    if unsqueeze:
        image = torch.unsqueeze(image, 0)
    image = image.to(device)

    return image


def attention_mask_from_font(font: Tensor, default_font: Tensor, apply_blur: bool, lower_limit: float):
    mask = torch.nn.functional.relu(torch.sub(default_font, font))
    if apply_blur:
        mask = transforms.GaussianBlur(9, 2)(mask)

    mask = mask*(1-lower_limit)+lower_limit
    return mask


def unfold_image(tensor: Tensor):
    unfolded = torch.ones((1, 64, 1664)).to(tensor.device)
    for i in range(26):
        unfolded[0, :, i*64:(i+1)*64] = tensor[i, :, :]
    return unfolded


def fold_image(tensor: Tensor) -> Tensor:
    sample_channeled = torch.zeros((26, 64, 64))
    for i in range(26):
        sample_channeled[i, :, :] = tensor[0, :, 64*i:64*(i+1)]
    return sample_channeled
