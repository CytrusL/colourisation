import torch
from torch import nn
from torch.autograd import Variable

from models import Vgg19


class GANLoss:
    def __init__(self, loss='bce'):
        self.loss_type = loss
        if loss == 'bce':
            self.loss = nn.BCEWithLogitsLoss()
        elif loss == 'mse':
            self.loss = nn.MSELoss()
        elif loss == 'w':
            self.loss = self.WLoss

    @staticmethod
    def WLoss(x, is_real):
        return -torch.mean(x.reshape(-1)) if is_real else torch.mean(x.reshape(-1))

    @staticmethod
    def get_target_tensor(inputs, is_real):
        if is_real:
            return torch.ones_like(inputs)
        else:
            return torch.zeros_like(inputs)

    def __call__(self, inputs, is_real: bool, detach=False):
        loss = 0

        for input_i in inputs:
            if not self.loss_type == 'w':
                target_tensor = self.get_target_tensor(input_i, is_real)
            else:
                target_tensor = is_real

            if not detach:
                loss += self.loss(input_i, target_tensor)
            else:
                loss += self.loss(input_i.detach(), target_tensor)

        return loss


class VGGLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class GradientPenalty:
    def __init__(self, device='cuda'):
        self.device = device

    def __call__(self, disc, x, y, y_fake):
        BATCH_SIZE, C, H, W = y.shape
        eps = torch.rand((BATCH_SIZE, 1, 1, 1))
        eps = eps.expand_as(y).to(self.device)
        interpolates = eps * y.data + (1 - eps) * y_fake.data
        interpolates = Variable(interpolates, requires_grad=True)

        scores = disc(x, interpolates)
        prob_interpolated = 0
        for score in scores:
            prob_interpolated += torch.mean(score)

        (gradients,) = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolates,
            grad_outputs=torch.ones_like(prob_interpolated),
            create_graph=True,
            retain_graph=True
        )
        gradients = gradients.view(BATCH_SIZE, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1))
        gradient_penalty = torch.mean((gradients_norm - 1) ** 2)

        return gradient_penalty