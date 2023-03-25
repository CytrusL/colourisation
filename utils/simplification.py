import copy

import numpy as np
import torch
from PIL import Image
from torch import nn
import cv2
from torchvision.transforms.functional import to_tensor
from utils.io import tensor_to_numpy
from data.models import DCSCN

"""
Sketch Simplification

"Mastering Sketching: Adversarial Augmentation for Structured Prediction"
   Edgar Simo-Serra*, Satoshi Iizuka*, Hiroshi Ishikawa (* equal contribution)
   ACM Transactions on Graphics (TOG), 2018

Modified from:
https://github.com/bobbens/sketch_simplification
https://github.com/bobbens/sketch_simplification/pull/12
"""


class SketchSimp(nn.Module):
    def __init__(self, device='cuda'):
        super(SketchSimp, self).__init__()
        self.device = device

        self.model = nn.Sequential(
            nn.Conv2d(1, 48, (5, 5), (2, 2), (2, 2)),
            nn.ReLU(),
            nn.Conv2d(48, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, (4, 4), (2, 2), (1, 1), (0, 0)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, (4, 4), (2, 2), (1, 1), (0, 0)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 48, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 48, (4, 4), (2, 2), (1, 1), (0, 0)),
            nn.ReLU(),
            nn.Conv2d(48, 24, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(24, 1, (3, 3), (1, 1), (1, 1)),
            nn.Sigmoid(),
        )

        self.immean = 0.9664114577640158
        self.imstd = 0.0858381272736797

    def forward(self, x):
        return self.model(x)

    def load_weights(self, pth):
        state_dict = torch.load(pth)
        state_dict_v2 = copy.deepcopy(state_dict)
        for key in state_dict:
            state_dict_v2['model.' + key] = state_dict_v2.pop(key)
        self.load_state_dict(state_dict_v2)

    @staticmethod
    def preprocess(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, se)

        data = cv2.divide(gray, bg, scale=255) - 1

        return data

    def generate(self, img):
        x = self.preprocess(img)
        w, h = x.shape

        pw = 8 - (w % 8) if w % 8 != 0 else 0
        ph = 8 - (h % 8) if w % 8 != 0 else 0

        x = ((to_tensor(x) - self.immean) / self.imstd).unsqueeze(0)

        if pw != 0 or ph != 0:
            x = torch.nn.ReplicationPad2d((0, pw, 0, ph))(x).data

        with torch.no_grad():
            pred = self(x.to(self.device))

        pred = tensor_to_numpy(pred)
        output = pred[0:w-pw, 0:h-ph, :]

        # p = np.random.choice(['erode', 'dilate', 'none'], p=[0.05, 0.25, 0.7])
        # if p == 'dilate':
        #     output = cv2.dilate(output, np.ones((2, 2), np.uint8), cv2.BORDER_REFLECT)
        # elif p == 'erode':
        #     output = cv2.erode(output, np.ones((2, 2), np.uint8), cv2.BORDER_REFLECT)

        output = cv2.resize(output, (int(h/2), int(w/2)))

        return output


if __name__ == '__main__':
    model = SketchSimp().to('cuda')
    DCSCN = DCSCN().to('cuda')
    DCSCN.load_state_dict(torch.load('../weights/DCSCN.pt'))
    model.load_weights('../weights/sketchSimp.pth')

    image = np.array(Image.open('og.jpg').resize((512, 520)))
    image = DCSCN.upsample(image)
    out = model.generate(image)
    print(out.shape)
    Image.fromarray(out).save('ss.png')
