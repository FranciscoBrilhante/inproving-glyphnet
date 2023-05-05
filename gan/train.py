from torch.utils.tensorboard import SummaryWriter

import os
from tqdm import trange

#pytorch modules
from torch import save

# local modules
from data.glyphgan import data_loader, save_random_batch_elem
from configs import *
from models.glyphgan import GlyphGan
from models.glyphgan_new import GlyphGanNew
from utils import create_dirs_logs, update_telemetry, set_all_seeds

def train_default(config: dict):
    create_dirs_logs(config)
    path_log = os.path.join(config['path_log'], config['logname'])
    path_data_train = os.path.join(config['data_path'], 'train')
    path_checkpoint = os.path.join(path_log, "Checkpoints", f"last.pt")

    writer = SummaryWriter(log_dir=path_log)
    set_all_seeds(config['seed'])

    model = GlyphGan(is_train=True, beta=config['beta'], epochs_decay_lr=config['epochs_decay_lr'], lr=config['lr'], device=config['device'], l1_lambda=config['l1_lambda'], use_mask=config['use_mask'])
    model.to(config['device'])
    # model.print_network()
    #summary(model, (26, 64, 64))

    step = 0
    with trange(config['epochs']) as t:
        for epoch in t:
            data_train = data_loader(path=path_data_train, batch_size=config['batch_size'], n_glyphs_visible=config['n_visible'], glyphs_visible=config.get('glyphs_visible', None), use_mask=config['use_mask'],
                                     default_font=config['default_font'], seed=config['seed'], use_skeleton_mask=config['use_skeleton_mask'], masks_path=config['masks_path'])
            for batch in data_train:
                batch_x, targets, masks = batch
                batch_x, targets, masks = batch_x.to(config['device']), targets.to(config['device']), masks.to(config['device'])
                #save_random_batch_elem(model, (batch_x, targets, masks),path_log)
                loss = model.optimize_parameters(batch_x, targets, targets, masks)
        
                update_telemetry(writer, t, step, loss, lr=model.old_lr)
                step += 1

            if config['epochs']-epoch <= config['epochs_decay_lr']:
                model.update_learning_rate()
            save(model, path_checkpoint)
    writer.close()

if __name__ == "__main__":
    pass

        
    