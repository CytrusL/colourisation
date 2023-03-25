import os

import numpy as np
import torch
from torch.utils import data
from PIL import Image
from skimage.draw import disk
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.pytorch.functional import img_to_tensor

from data.models import *
from data.hint import generate_hint

IMG_EXTENSIONS = [
    'jpg', 'jpeg', 'png',
]


def scale_resize(x, min_size, save_resize=False, fp=None):
    w, h = x.size
    assert w >= min_size and h >= min_size

    if w > min_size and h > min_size:
        scale = w / min_size if w < h else h / min_size
        x = x.resize((max(int(w / scale), min_size), max(int(h / scale), min_size)))
        if save_resize:
            assert fp is not None
            x.save(fp)

    return x


class IllustDataset(data.Dataset):
    def __init__(self,
                 root_dir,
                 image_size=(512, 512),
                 radius_range=(2, 16),
                 proportion_range=(0, 2e-4),
                 device='cuda',
                 ):
        assert isinstance(image_size, (tuple, list, int))

        self.root_dir = root_dir

        self.image_size = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        self.list_files = [img for img in os.listdir(self.root_dir) if img.split('.')[-1].lower() in IMG_EXTENSIONS]
        self.device = device

        self.sketchkeras = SketchKeras(device=device).to(device).eval()
        self.sketchsimp = SketchSimp(device=device).to(device).eval()
        self.DCSCN = DCSCN(device=device).to(device).eval()
        self.xDoG = xDoG()

        self.sketchkeras.load_state_dict(torch.load('data/weights/sketchKeras.pth'))
        self.sketchsimp.load_weights('data/weights/sketchSimp.pth')
        self.DCSCN.load_state_dict(torch.load('data/weights/DCSCN.pt'))

        self.jitter = A.ColorJitter(brightness=0.15,
                                    saturation=0.15,
                                    contrast=0.15,
                                    hue=0.15,
                                    p=0.2)

        self.radius_range = radius_range
        self.proportion_range = proportion_range

        self.preprocess = A.Compose(
            (
                A.RandomCrop(*self.image_size),
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=0),
            )
        )
        self.postprocess = A.Compose(
            (
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2(),
            )
        )

    def __len__(self):
        return len(self.list_files)

    def __repr__(self):
        return f'dataset length: {len(self.list_files)}'

    def __getitem__(self, index):
        color_file = os.path.join(self.root_dir, self.list_files[index])
        color = Image.open(color_file).convert('RGB')
        color = scale_resize(color, max(self.image_size), save_resize=True, fp=color_file)
        color = np.array(color)
        color = self.preprocess(image=color)['image']

        sketch_method = np.random.choice(['xdog', 'sk', 'ss'], p=[0.3, 0.3, 0.4])

        if sketch_method == 'xdog':
            sketch = self.xDoG(color / 255, sigma=np.random.uniform(0.3, 0.5)) * 255
        elif sketch_method == 'sk':
            sketch = self.sketchkeras.generate(color)
        else:
            c_up = self.DCSCN.upsample(color)
            sketch = self.sketchsimp.generate(c_up)

        sketch = np.expand_dims(sketch.astype(np.uint8), axis=2).repeat(3, axis=2)

        hint = generate_hint(color, self.radius_range)
        hint = img_to_tensor(hint)

        sketch = self.jitter(image=sketch)['image']

        color = self.postprocess(image=color)['image']
        sketch = self.postprocess(image=sketch)['image']

        return sketch, hint, color


if __name__ == '__main__':
    image = Image.open('dataset/val/86665132_p0.png')
    # im = scale_resize(image, 512)
    # print(im.size)
    # for i in range(10):
    # hint = generate_hint(np.array(image))
    # Image.fromarray(((hint[:, :, :3] + 1) * 127.5).astype(np.uint8)).show()
    for i in range(1):
        s, h, c = IllustDataset('dataset/train/')[i]
        print(c)