
import os
import random

# pytorch modules
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

#local modules
from models.glyphgan import GlyphGan
from .common import fold_image,load_image_to_tensor,attention_mask_from_font, unfold_image

class CapitalsDataset(Dataset):
    def __init__(self, root_dir, n_glyphs_visible: int, glyphs_visible: list[int], use_mask: bool, default_font_path: str, use_skeleton_mask: bool, masks_path: str) -> None:
        super().__init__()
        self.n_glyphs_visible = n_glyphs_visible
        self.glyphs_visible = glyphs_visible
        self.use_mask = use_mask
        self.default_font_path = default_font_path
        self.use_skeleton_mask = use_skeleton_mask
        self.masks_path = masks_path
        self.root_dir=root_dir
    
    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path_image=os.path.join(self.root_dir,os.listdir(self.root_dir)[idx])
        target=load_image_to_tensor(path_image,unsqueeze=False,device='cpu')
        target=fold_image(target)
        mask = torch.empty((1, 1))

        model_input = torch.clone(target)
        if self.glyphs_visible == None:
            idx_to_hide = random.sample(list(range(26)),26-self.n_glyphs_visible)
        else:
            idx_to_hide = [i for i in range(26) if i not in self.glyphs_visible]
        for i in idx_to_hide:
            model_input[i, :, :] = 1

        if self.use_mask:
            if self.use_skeleton_mask:
                path_mask = os.path.join(self.masks_path,os.listdir(self.masks_path)[idx])
                mask = load_image_to_tensor(path_mask, unsqueeze=False, device='cpu')
                mask = fold_image(mask)
            else:
                default_font = load_image_to_tensor(self.default_font_path, unsqueeze=False, device='cpu')
                default_font = fold_image(default_font)
                mask = attention_mask_from_font(target, default_font, True, 0.5)

        return model_input.detach(), target.detach(), mask.detach()

def data_loader(path: str, batch_size: int, n_glyphs_visible: int, glyphs_visible: list[int], use_mask: bool, default_font: str, seed: int, use_skeleton_mask: bool, masks_path: str):
    g = torch.Generator()
    g.manual_seed(seed)
    data = CapitalsDataset(root_dir=path, n_glyphs_visible=n_glyphs_visible,glyphs_visible=glyphs_visible, use_mask=use_mask,default_font_path=default_font,use_skeleton_mask=use_skeleton_mask,masks_path=masks_path)
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=4, generator=g)
    return data_loader


def save_random_batch_elem(model: GlyphGan, batch: tuple, path: str):
    idx = 0
    model_input, targets, mask = batch
    with torch.no_grad():
        inp = torch.clone(model_input[idx, :, :, :])
        out = model.forward(inp.unsqueeze(0))[0]
        targ = torch.clone(targets[idx, :, :, :])
        try:
            mas = torch.clone(mask[idx, :, :, :])
        except:
            mas = torch.ones(size=(26,64,64))

        save_image(unfold_image(inp), os.path.join(path, "input.png"))
        save_image(unfold_image(out), os.path.join(path, "output.png"))
        save_image(unfold_image(targ), os.path.join(path, "target.png"))
        save_image(unfold_image(mas), os.path.join(path, "mask.png"))