import cv2
import numpy as np
import torch
from torch import nn

from PIL import Image


class SketchKeras(nn.Module):
    def __init__(self, device='cuda'):
        super(SketchKeras, self).__init__()
        self.device = device

        self.downblock_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32, eps=1e-3, momentum=0),
            nn.ReLU(),
        )
        self.downblock_2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0),
            nn.ReLU(),
        )
        self.downblock_3 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0),
            nn.ReLU(),
        )
        self.downblock_4 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0),
            nn.ReLU(),
        )
        self.downblock_5 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, kernel_size=4, stride=2),
            nn.BatchNorm2d(512, eps=1e-3, momentum=0),
            nn.ReLU(),
        )
        self.downblock_6 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.BatchNorm2d(512, eps=1e-3, momentum=0),
            nn.ReLU(),
        )

        self.upblock_1 = nn.Sequential(
            nn.Upsample((64, 64)),
            nn.ReflectionPad2d((1, 2, 1, 2)),
            nn.Conv2d(1024, 512, kernel_size=4, stride=1),
            nn.BatchNorm2d(512, eps=1e-3, momentum=0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0),
            nn.ReLU(),
        )

        self.upblock_2 = nn.Sequential(
            nn.Upsample((128, 128)),
            nn.ReflectionPad2d((1, 2, 1, 2)),
            nn.Conv2d(512, 256, kernel_size=4, stride=1),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0),
            nn.ReLU(),
        )

        self.upblock_3 = nn.Sequential(
            nn.Upsample((256, 256)),
            nn.ReflectionPad2d((1, 2, 1, 2)),
            nn.Conv2d(256, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0),
            nn.ReLU(),
        )

        self.upblock_4 = nn.Sequential(
            nn.Upsample((512, 512)),
            nn.ReflectionPad2d((1, 2, 1, 2)),
            nn.Conv2d(128, 64, kernel_size=4, stride=1),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32, eps=1e-3, momentum=0),
            nn.ReLU(),
        )

        self.last_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.last_conv = nn.Conv2d(64, 1, kernel_size=3, stride=1)

    def forward(self, x):
        d1 = self.downblock_1(x)
        d2 = self.downblock_2(d1)
        d3 = self.downblock_3(d2)
        d4 = self.downblock_4(d3)
        d5 = self.downblock_5(d4)
        d6 = self.downblock_6(d5)

        u1 = torch.cat((d5, d6), dim=1)
        u1 = self.upblock_1(u1)
        u2 = torch.cat((d4, u1), dim=1)
        u2 = self.upblock_2(u2)
        u3 = torch.cat((d3, u2), dim=1)
        u3 = self.upblock_3(u3)
        u4 = torch.cat((d2, u3), dim=1)
        u4 = self.upblock_4(u4)
        u5 = torch.cat((d1, u4), dim=1)

        out = self.last_conv(self.last_pad(u5))

        return out

    @staticmethod
    def preprocess(img):
        h, w, c = img.shape
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        highpass = img.astype(int) - blurred.astype(int)
        highpass = highpass.astype(np.float) / 128.0
        highpass /= np.max(highpass)

        ret = np.zeros((512, 512, 3), dtype=np.float)
        ret[0:h, 0:w, 0:c] = highpass
        return ret

    @staticmethod
    def postprocess(pred, thresh=0.18, smooth=False):
        assert thresh <= 1.0 and thresh >= 0.0
        thresh = np.random.uniform(0.10, 0.30)

        pred = np.amax(pred, 0)
        pred[pred < thresh] = 0
        pred = 1 - pred
        pred *= 255
        pred = np.clip(pred, 0, 255).astype(np.uint8)
        if smooth:
            pred = cv2.medianBlur(pred, 3)
        return pred

    def generate(self, img: np.array):
        h, w, _ = img.shape

        # preprocess
        img = self.preprocess(img)
        x = img.reshape(1, *img.shape).transpose(3, 0, 1, 2)
        x = torch.tensor(x).float()

        # feed into the network
        with torch.no_grad():
            pred = self(x.to(self.device))
        pred = pred.squeeze()

        # postprocess
        output = pred.cpu().detach().numpy()
        output = self.postprocess(output, thresh=0.25, #np.random.uniform(0.15, 0.35)
                                  smooth=False)
        output = output[:h, :w]

        return output


if __name__ == '__main__':
    model = SketchKeras().to('cuda')
    image = np.array(Image.open('og.jpg').resize((504, 512)))
    model.load_state_dict(torch.load('../weights/sketchKeras.pth'))
    out = model.generate(image)
    Image.fromarray(out).save('./sk.png')
