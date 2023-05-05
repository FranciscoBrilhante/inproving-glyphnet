import torch
from torch.optim import Adam
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import random
from image_pool import ImagePool


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, stride_size: int, padding_size: int):
        super(ResnetBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, stride=stride_size, padding=padding_size),
            nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, stride=stride_size, padding=padding_size),
            nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )

    def forward(self, x: Tensor):
        out = x + self.block(x)
        return out


class GlyphGan(nn.Module):
    def __init__(self, lr: float, epochs_decay_lr: int, is_train: bool, beta: float, device: str, l1_lambda: float, use_mask:bool):
        super(GlyphGan, self).__init__()
        self.lr = lr
        self.old_lr=lr
        self.epochs_decay_lr = epochs_decay_lr
        self.is_train = is_train
        self.beta = beta
        self.device=device
        self.discriminator_input_pool = ImagePool(64)
        self.l1_lambda=l1_lambda
        self.use_mask=use_mask
        self.to(device)
        self.discriminator_loss=nn.MSELoss()
        self.generator_loss=nn.L1Loss()

        self.generator_3d = nn.Sequential(
            nn.Conv3d(26, 26, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=26),
            nn.BatchNorm3d(26, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.generator = nn.Sequential(
            nn.Conv2d(26, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 576, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            ResnetBlock(dim=576, kernel_size=3, stride_size=1, padding_size=1),
            ResnetBlock(dim=576, kernel_size=3, stride_size=1, padding_size=1),
            ResnetBlock(dim=576, kernel_size=3, stride_size=1, padding_size=1),
            ResnetBlock(dim=576, kernel_size=3, stride_size=1, padding_size=1),
            ResnetBlock(dim=576, kernel_size=3, stride_size=1, padding_size=1),
            ResnetBlock(dim=576, kernel_size=3, stride_size=1, padding_size=1),
            nn.ConvTranspose2d(576, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 26, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
            nn.Tanh(),
        )

        self.discriminator_local = nn.Sequential(
            nn.Conv2d(26, 26, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(26, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(26, 26, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(26, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.discriminator_global = nn.Sequential(
            nn.Conv2d(26, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
        )

        if self.is_train:
            self.optimizer_generator_3d = Adam(self.generator_3d.parameters(), lr=self.lr, betas=(self.beta, 0.999))
            self.optimizer_generator = Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta, 0.999))
            self.optimizer_discriminator_local = Adam(self.discriminator_local.parameters(), lr=self.lr, betas=(self.beta, 0.999))
            self.optimizer_discriminator_global = Adam(self.discriminator_global.parameters(), lr=self.lr, betas=(self.beta, 0.999))

    def print_network(self):
        for net in [self.generator_3d, self.generator, self.discriminator_local, self.discriminator_global]:
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            print(net)
            print('Total number of parameters: %d' % num_params)

    def update_learning_rate(self):
        lrd = self.lr / self.epochs_decay_lr
        lr = self.old_lr - lrd
        for param_group in self.optimizer_discriminator_global.param_groups:
            param_group['lr'] = lr

        for param_group in self.optimizer_discriminator_local.param_groups:
            param_group['lr'] = lr

        for param_group in self.optimizer_generator.param_groups:
            param_group['lr'] = lr

        for param_group in self.optimizer_generator_3d.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def add_label_noise(self, labels: list) -> list:
        labels = [1-label if random.random() > 0.8 else label for label in labels]
        return labels

    def labels_to_tensor(self, labels: list[int], size:tuple[int]) -> Tensor:
        tensor_labels=torch.ones(size)
        for i in range(len(labels)):
            tensor_labels[i,:,:,:]=labels[i]
        
        tensor_labels=tensor_labels.to(self.device)
        return tensor_labels

    def forward(self, input: Tensor) -> Tensor:
        input_indep = self.generator_3d.forward(input.unsqueeze(2))
        output = self.generator.forward(input_indep.squeeze(2))

        return output

    def backward_D(self, input_real: Tensor, input_fake: Tensor):
        #fake
        #get global discriminator output
        pool_fake=self.discriminator_input_pool.query(input_fake)
        output_fake_local=self.discriminator_global(pool_fake.detach())
        #create target labels with same size
        labels_fake_list=self.add_label_noise([0 for i in range(input_fake.size(0))])
        labels_fake=self.labels_to_tensor(labels_fake_list,output_fake_local.size())
        #calculate local loss
        loss_fake=self.discriminator_loss(output_fake_local,labels_fake)
        #get local then global discriminator output
        output_fake_aux=self.discriminator_local(pool_fake.detach())
        output_fake_global=self.discriminator_global(output_fake_aux)
        #create target labels with same size
        labels_fake=self.labels_to_tensor(labels_fake_list,output_fake_global.size())
        #calculate global loss
        loss_fake+=self.discriminator_loss(output_fake_global,labels_fake)

        #real
        #get global discriminator output
        output_real_local=self.discriminator_global(input_real.detach())
        #create target labels with same size
        labels_real_list=self.add_label_noise([1 for i in range(input_fake.size(0))])
        labels_real=self.labels_to_tensor(labels_real_list,output_real_local.size())
        #calculate local loss
        loss_real=self.discriminator_loss(output_real_local,labels_real)
        #get local then global discriminator output
        output_real_aux=self.discriminator_local(input_real.detach())
        output_real_global=self.discriminator_global(output_real_aux)
        #create target labels with same size
        labels_real=self.labels_to_tensor(labels_real_list,output_real_global.size())
        #calculate global loss
        loss_real+=self.discriminator_loss(output_real_global,labels_real)

        loss=(loss_fake+loss_real)*0.5
        loss.backward()
        return loss

    def backward_G(self, output: Tensor, target: Tensor, mask: Tensor):
        #get global discriminator output
        output_fake_local=self.discriminator_global(output)
        #create target labels with same size
        labels_real_list=[1 for i in range(output.size(0))]
        labels_real=self.labels_to_tensor(labels_real_list,output_fake_local.size())
        #calculate local loss
        loss_discriminator=self.discriminator_loss(output_fake_local,labels_real)

        #get local then global discriminator output
        output_fake_aux=self.discriminator_local(output)
        output_fake_global=self.discriminator_global(output_fake_aux)
        #create target labels with same size
        labels_real=self.labels_to_tensor(labels_real_list,output_fake_global.size())
        #calculate global loss
        loss_discriminator+=self.discriminator_loss(output_fake_global,labels_real)
        #calculate generator mse loss
        if self.use_mask:
            loss_generator=self.generator_loss(torch.mul(output, mask),torch.mul(target, mask))*self.l1_lambda
        else:
            loss_generator=self.generator_loss(output,target)*self.l1_lambda
        #sum gen and disc losses
        loss_total=loss_discriminator+loss_generator

        loss_total.backward()
        return loss_total

    def optimize_parameters(self,input_generator: Tensor, target_generator: Tensor, input_discriminator_real: Tensor, generator_loss_masks:Tensor):
        output_generator=self.forward(input_generator)

        self.optimizer_discriminator_global.zero_grad()
        self.optimizer_discriminator_local.zero_grad()

        loss_discriminator=self.backward_D(input_discriminator_real,output_generator)

        self.optimizer_discriminator_global.step()
        self.optimizer_discriminator_local.step()

        self.optimizer_generator.zero_grad()
        self.optimizer_generator_3d.zero_grad()

        loss_generator=self.backward_G(output_generator,target_generator, generator_loss_masks)

        self.optimizer_generator.step()
        self.optimizer_generator_3d.step()

        return {'loss_disc':loss_discriminator, 'loss_gen':loss_generator}