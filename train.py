import os
import yaml
import pprint

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from data.dataset import IllustDataset
from models import Generator, Discriminator, Vgg19
from utils.io import LossCalc
from torchvision.utils import save_image


class Trainer(object):
    def __init__(self, config):
        self.train_config = config['train']
        self.data_config = config['dataset']
        self.model_config = config['model']
        self.loss_config = config['loss']

        self.train_dataset = IllustDataset(
            os.path.join(self.data_config['dir']['root'], self.data_config['dir']['train']),
            self.data_config['image_size'],
            self.data_config['hint']['radius_range'],
            self.data_config['hint']['proportion_range'],
            self.train_config['device'],
        )
        print(self.train_dataset)

        self.val_dataset = IllustDataset(
            os.path.join(self.data_config['dir']['root'], self.data_config['dir']['val']),
            self.data_config['image_size'],
            self.data_config['hint']['radius_range'],
            self.data_config['hint']['proportion_range'],
            self.train_config['device'],
        )
        print(self.val_dataset)

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.train_config['batch_size'],
                                       num_workers=self.train_config['num_workers'],
                                       shuffle=True,
                                       pin_memory=False,
                                       drop_last=True,
                                       )

        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=self.train_config['valid_size'],
                                     shuffle=False)

        self.netG = Generator(
            self.model_config['gen']['in_ch'],
            self.model_config['gen']['depth'],
            self.model_config['gen']['dims'],
            self.model_config['gen']['attn'],
            self.model_config['gen']['drop_path_rate'],
            self.model_config['gen']['layer_scale_init_value'],
            self.model_config['gen']['scale'],
            training=False,
        )
        self.netD = Discriminator(
            self.model_config['disc']['in_ch'],
            self.model_config['disc']['depth'],
            self.model_config['disc']['dims'],
            self.model_config['disc']['drop_path_rate'],
            self.model_config['disc']['layer_scale_init_value'],
            self.model_config['disc']['scale'],
            self.model_config['disc']['num_D'],
            self.model_config['disc']['patch'],
            self.data_config['image_size'],
            training=False,
        )

        self.netG = self.netG.to(self.train_config['device'])
        self.netD = self.netD.to(self.train_config['device'])
        self.vgg = Vgg19().to(self.train_config['device']).eval()

        self.optG = optim.AdamW(self.netG.parameters(),
                                lr=self.model_config['gen']['lr'],
                                betas=(self.model_config['gen']['b1'],
                                       self.model_config['gen']['b2']),
                                eps=1e-6)
        self.optD = optim.AdamW(self.netD.parameters(),
                                lr=self.model_config['disc']['lr'],
                                betas=(self.model_config['disc']['b1'],
                                       self.model_config['disc']['b2']),
                                eps=1e-6)
        self.scalerG = GradScaler()
        self.scalerD = GradScaler()

        self.loss = LossCalc()

        self.start_epoch = 0
        self.cur_iter = 0
        if self.train_config['load_model']:
            self.load(os.path.join(self.model_config['dir']['root'],
                                   self.model_config['dir']['disc'],
                                   self.model_config['dir']['ext']),
                      self.netD,
                      self.optD)
            self.load(os.path.join(self.model_config['dir']['root'],
                                   self.model_config['dir']['gen'],
                                   self.model_config['dir']['ext']),
                      self.netG,
                      self.optG)

        self.val_x, self.val_h, self.val_y = next(iter(self.val_loader))
        self.val_x, self.val_h, self.val_y = self.val_x.to(self.train_config['device']), \
                                             self.val_h.to(self.train_config['device']), \
                                             self.val_y.to(self.train_config['device'])

        # train
        for epoch in range(self.start_epoch, self.train_config['epoch']):
            self.train(epoch)

    def _loss_weight_scheduler(self):
        pass

    def train(self, epoch):
        pbar = tqdm(self.train_loader, leave=True)
        for n_iter, (x, h, y) in enumerate(pbar):
            self._loss_weight_scheduler()

            # Save Examples
            if self.cur_iter % self.train_config['interval']['save_examples'] == 0:
                self.save_examples()

            # Save model
            if self.cur_iter % self.train_config['interval']['save_model'] == 0 and self.train_config['save_model']:
                self.save(os.path.join(self.model_config['dir']['root'],
                                       self.model_config['dir']['gen'],
                                       self.model_config['dir']['ext']),
                          self.netG,
                          self.optG,
                          epoch + 1)
                self.save(os.path.join(self.model_config['dir']['root'],
                                       self.model_config['dir']['disc'],
                                       self.model_config['dir']['ext']),
                          self.netD,
                          self.optD,
                          epoch + 1)

            # Initialize
            x, h, y = x.to(self.train_config['device']), \
                      h.to(self.train_config['device']), \
                      y.to(self.train_config['device'])

            self.optD.zero_grad()
            self.optG.zero_grad()

            ############################
            # (1) Update D network
            ###########################

            with autocast():
                y_f = self.netG(x, h)
                disc_loss = self.loss.adversarial_disc_loss(self.netD,
                                                            x, y, y_f.detach())

            self.scalerD.scale(disc_loss).backward()
            self.scalerD.step(self.optD)
            self.scalerD.update()

            ############################
            # (2) Update G network
            ############################

            with autocast():
                adv_gen_loss, fm_loss = self.loss.adversarial_gen_loss(self.netD,
                                                                       x, y, y_f)
                vgg_loss = self.loss.perceptual_loss(self.vgg, y, y_f)
                l1_loss = self.loss.content_loss(y, y_f)
                tv_loss = self.loss.total_variation_loss(y_f)

                adv_gen_loss *= self.loss_config['adv']
                fm_loss *= self.loss_config['fm']
                vgg_loss *= self.loss_config['vgg']
                l1_loss *= self.loss_config['l1']
                tv_loss *= self.loss_config['tv']

                gen_loss = adv_gen_loss + vgg_loss + fm_loss + l1_loss + tv_loss

            self.scalerG.scale(gen_loss).backward()
            self.scalerG.step(self.optG)
            self.scalerG.update()

            ############################
            # (3) Report
            ############################

            pbar.set_description(f'gen_loss: {gen_loss.item()}, disc_loss: {disc_loss.item()}')
            if self.cur_iter % self.train_config['interval']['log'] == 0:
                losses = [str(gen_loss.item()),
                          str(disc_loss.item()),
                          str(adv_gen_loss.item()),
                          str(fm_loss.item()),
                          str(vgg_loss.item()),
                          str(l1_loss.item()),
                          str(tv_loss.item()),
                          ]
                print("\n==============================================")
                print(f"Epoch: {epoch + 1} \n"
                      f"Iter: {n_iter} / {self.cur_iter} \n"
                      f"G loss: {losses[0]} \n"
                      f"D loss: {losses[1]} \n"
                      f"Other: {losses[2:]}")
                print("==============================================")

                if self.train_config['record_loss'] and self.cur_iter % 500 == 0:
                    with open(self.train_config['dir']['log'], 'a') as log:
                        log.write(f'iter {self.cur_iter}: ')
                        log.write(','.join(losses))
                        log.write('\n')

            self.cur_iter += 1

    def save_examples(self):
        print('\n=> Saving examples')
        with torch.no_grad():
            y_fake = self.netG(self.val_x, self.val_h)
        save_image(y_fake * 0.5 + 0.5, self.train_config['dir']['save_examples'] + f"/gen_{self.cur_iter}.png")
        save_image(self.val_x * 0.5 + 0.5, self.train_config['dir']['save_examples'] + f"/input_{self.cur_iter}.png")
        save_image(self.val_y * 0.5 + 0.5, self.train_config['dir']['save_examples'] + f"/label_{self.cur_iter}.png")
        save_image(self.val_h[:, :3] * 0.5 + 0.5,
                   self.train_config['dir']['save_examples'] + f"/hint_{self.cur_iter}.png")

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
