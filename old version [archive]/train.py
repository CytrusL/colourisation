import os
import yaml
import pprint

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import IllustDataset
from models import Generator, MultiscaleDiscriminator
from models.criterion import VGGLoss, GANLoss, GradientPenalty
from utils import save_examples


class Trainer(object):
    def __init__(self, config):
        self.train_config = config['train']
        self.data_config = config['dataset']
        self.model_config = config['model']
        self.loss_config = config['loss']

        if self.train_config['logger']:
            self.writer = SummaryWriter(log_dir=self.train_config['dir']['log'])

        self.train_dataset = IllustDataset(
            os.path.join(self.data_config['dir']['root'], self.data_config['dir']['train']),
            self.data_config['dir']['color'],
            self.data_config['dir']['sketch'],
            self.data_config['train_size'])
        print(self.train_dataset)

        self.val_dataset = IllustDataset(os.path.join(self.data_config['dir']['root'], self.data_config['dir']['val']),
                                         self.data_config['dir']['color'],
                                         self.data_config['dir']['sketch'],
                                         self.data_config['valid_size'])
        print(self.val_dataset)

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.train_config['batch_size'],
                                       num_workers=self.train_config['num_workers'],
                                       shuffle=True,
                                       pin_memory=True)

        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=self.train_config['valid_size'],
                                     shuffle=True)

        self.netG = Generator(
            self.model_config['gen']['in_ch'],
            self.model_config['gen']['base'],
            self.model_config['gen']['num_layers'],
            self.model_config['gen']['up_layers'],
        )
        self.netD = MultiscaleDiscriminator(
            self.model_config['disc']['multi'],
            self.model_config['disc']['in_ch'],
            self.model_config['disc']['base'],
        )  # Returns 3 tensors

        self.netG = nn.DataParallel(self.netG.to(self.train_config['device']))
        self.netD = nn.DataParallel(self.netD.to(self.train_config['device']))

        self.optG = optim.Adam(self.netG.parameters(),
                               lr=self.model_config['gen']['lr'],
                               betas=(self.model_config['gen']['b1'],
                                      self.model_config['gen']['b2']))
        self.optD = optim.Adam(self.netD.parameters(),
                               lr=self.model_config['disc']['lr'],
                               betas=(self.model_config['disc']['b1'],
                                      self.model_config['disc']['b2']))

        self.criterionGan = GANLoss(self.loss_config['advtype'])
        self.criterionVGG = VGGLoss(self.train_config['device'])
        self.criterionGP = GradientPenalty(self.train_config['device'])
        self.criterionL1 = nn.L1Loss()

        self.start_epoch = 0
        self.cur_iter = 0
        if self.train_config['load_model']:
            self.load(self.model_config['dir']['gen'],
                      self.netG,
                      self.optG)
            self.load(self.model_config['dir']['disc'],
                      self.netD,
                      self.optD)

        # train
        for epoch in range(self.start_epoch, self.train_config['epoch']):
            if epoch % self.train_config['interval']['save_examples'] == 0:
                self.save_examples(epoch)
            self.train(epoch)

            if epoch % config.SAVE_FREQ == 0:
                self.save(self.train_config['dir']['gen'], self.netG, self.optG, epoch + 1)
                self.save(self.train_config['dir']['disc'], self.netD, self.optD, epoch + 1)

            self.writer.close()
            self.writer = SummaryWriter(log_dir=self.train_config['dir']['log'])

        self.writer.close()

    def weight_decay(self):
        if self.cur_iter > 50000:
            self.loss_config['l1'] = 1
            self.loss_config['vgg'] = 100

    def save_examples(self, epoch):
        print('\n===============Saving Examples===============')
        x, h, y, style = next(iter(self.val_loader))
        x, h, y, style = x.to(self.train_config['device']), \
                         h.to(self.train_config['device']), \
                         y.to(self.train_config['device']), \
                         style.to(self.train_config['device'])
        self.netG.eval()
        with torch.no_grad():
            y_fake = self.netG(x, h, style)
        save_examples(x, h, y, y_fake, epoch, self.train_config['dir']['save_examples'])
        print('====================Done====================\n')
        self.netG.train()

    def train(self, epoch):
        for n_iter, (x, h, y, style) in enumerate(tqdm(self.train_loader, leave=True)):
            self.weight_decay()
            x, h, y, style = x.to(self.train_config['device']), \
                             h.to(self.train_config['device']), \
                             y.to(self.train_config['device']), \
                             style.to(self.train_config['device'])

            ############################
            # (1) Update D network
            ###########################

            y_fake = self.netG(x, h, style)
            D_real = self.netD(x, y)
            loss_D_real = self.criterionGan(D_real, True)

            D_fake = self.netD(x, y_fake.detach())
            loss_D_fake = self.criterionGan(D_fake, False)
            loss_D_ADV = (loss_D_real + loss_D_fake) * self.loss_config['adv']
            loss_D_GP = self.criterionGP(self.netD, x, y, y_fake.detach()) * self.loss_config['gp']

            D_loss = loss_D_ADV + loss_D_GP

            self.optD.zero_grad()
            D_loss.backward()
            self.optD.step()

            ############################
            # (2) Update G network
            ############################

            loss_G_ADV = self.criterionGan(D_fake, True, detach=True) * self.loss_config['adv']

            loss_G_FM = 0
            for i in range(len(D_fake)):
                loss_G_FM += self.criterionL1(D_fake[i].detach(), D_real[i].detach())
            loss_G_FM *= self.loss_config['fm']

            loss_G_L1 = self.criterionL1(y, y_fake.detach()) * self.loss_config['l1']

            loss_G_VGG = self.criterionVGG(y_fake, y) * self.loss_config['vgg']

            G_loss = loss_G_ADV + loss_G_FM + loss_G_L1 + loss_G_VGG

            self.optG.zero_grad()
            G_loss.backward()
            self.optG.step()

            ############################
            # (3) Report
            ############################

            if self.cur_iter % self.train_config['interval']['logiter'] == 0:
                print("\n==============================================")
                print(f"Epoch: {epoch + 1} \n"
                      f"Iter: {n_iter} / {self.cur_iter} \n"
                      f"G loss: {G_loss.item()} \n"
                      f"D loss: {D_loss.item()}")
                print("==============================================")

                # Record Losses
                if self.train_config['logger']:
                    self.writer.add_scalar('loss D', D_loss.item(), self.cur_iter)
                    self.writer.add_scalar('loss D adv', loss_D_ADV.item(), self.cur_iter)
                    self.writer.add_scalar('loss D real', loss_D_real.item(), self.cur_iter)
                    self.writer.add_scalar('loss D fake', loss_D_fake.item(), self.cur_iter)
                    self.writer.add_scalar('loss D gp', loss_D_GP.item(), self.cur_iter)
                    self.writer.add_scalar('loss G', G_loss.item(), self.cur_iter)
                    self.writer.add_scalar('loss G adv', loss_G_ADV.item(), self.cur_iter)
                    self.writer.add_scalar('loss G L1', loss_G_L1.item(), self.cur_iter)
                    self.writer.add_scalar('loss G perc', loss_G_VGG.item(), self.cur_iter)
                    self.writer.add_scalar('loss G fm', loss_G_FM.item(), self.cur_iter)
            self.cur_iter += 1

    def save(self, fp, model, optimizer, epoch):
        print("=> Saving checkpoint")
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "cur_iter": self.cur_iter
        }
        torch.save(checkpoint, fp)

    def load(self, fp, model, optimizer):
        print("=> Loading checkpoint")
        checkpoint = torch.load(fp, map_location=self.train_config['device'])
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        self.start_epoch = checkpoint["epoch"]
        self.cur_iter = checkpoint["cur_iter"]

        for param_group in optimizer.param_groups:
            param_group["lr"] = self.model_config['gen']['lr']


if __name__ == '__main__':
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
        pprint.pprint(config)

    Trainer(config)
