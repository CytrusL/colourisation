import cv2
import torch
from torch import nn


def tensor_to_numpy(x):
    return x.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()


def im_resize(x):
    y = cv2.resize(x, (850, 1215), cv2.INTER_CUBIC)
    cv2.imwrite("new.png", y)


class LossCalc:
    def __init__(self, device='cuda'):
        self.maeloss = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()
        self.device = device

    def adversarial_disc_loss(self, disc, x, y, y_f):
        sum_loss = 0
        fake_list, _ = disc(x, y_f)
        real_list, _ = disc(x, y)

        for fake, real in zip(fake_list, real_list):
            sum_loss += self.bce(real, torch.ones_like(real)) + self.bce(fake, torch.zeros_like(fake))

        return sum_loss

    def adversarial_gen_loss(self, disc, x, y, y_f):
        sum_adv_loss = 0
        sum_fm_loss = 0
        fake_list, fake_points = disc(x, y_f)
        real_list, real_points = disc(x, y)

        for fake in fake_list:
            sum_adv_loss += self.bce(fake, torch.ones_like(fake))

        for f_feat, r_feat in zip(fake_points, real_points):
            sum_fm_loss += self.maeloss(f_feat, r_feat.detach())

        return sum_adv_loss, sum_fm_loss

    def adversarial_hingedis(self, disc, x, y, y_f):
        sum_adv_loss = 0
        fake_list, _ = disc(x, y_f)
        real_list, _ = disc(x, y)

        for fake, real in zip(fake_list, real_list):
            sum_adv_loss += nn.ReLU()(1.0 + fake).mean()
            sum_adv_loss += nn.ReLU()(1.0 - real).mean()

        return sum_adv_loss

    def adversarial_hingegen(self, disc, x, y, y_f):
        sum_adv_loss = 0
        sum_fm_loss = 0
        fake_list, fake_points = disc(x, y_f)
        _, real_points = disc(x, y)

        for fake in fake_list:
            sum_adv_loss += -fake.mean()

        d_weight = float(1.0 / 3.0)
        feat_weight = float(4.0 / 7.0)

        for f_feat, r_feat in zip(fake_points, real_points):
            sum_fm_loss += d_weight * feat_weight * self.maeloss(f_feat, r_feat.detach())

        return sum_adv_loss, sum_fm_loss

    def positive_enforcing_loss(self, y: torch.Tensor) -> torch.Tensor:
        sum_loss = 0

        for color in range(3):
            perch = y[:, color, :, :]
            mean = torch.mean(perch)
            mean = mean * torch.ones_like(mean)
            loss = torch.mean((perch-mean)**2)
            sum_loss += loss

        return -sum_loss

    def perceptual_loss(self, vgg, y, y_f):
        y_vgg, yf_vgg = vgg(y), vgg(y_f)
        sum_loss = 0
        for i in range(len(y_vgg)):
            _, c, h, w = y_vgg[i].shape
            sum_loss += self.maeloss(y_vgg[i].detach(), yf_vgg[i]) / (c * h * w)

        return sum_loss

    def content_loss(self, y, y_f):
        return self.maeloss(y, y_f)

    def total_variation_loss(self, y: torch.Tensor) -> torch.Tensor:
        _, c, h, w = y.size()

        vertical_loss = torch.mean((torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:]))**2)
        horizon_loss = torch.mean((torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))**2)

        return (vertical_loss + horizon_loss) / (c * h * w)

    def gradient_penalty(self, disc, x, y, y_f):
        alpha = torch.rand(x.size(0), 1, 1, 1, device=self.device)

        interpolates = alpha * y + ((1 - alpha) * y_f)

        interpolates.requires_grad = True

        disc_interpolates = disc(x, interpolates)[0][0]

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size(), device=self.device),
                                        create_graph=True,
                                        retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty


