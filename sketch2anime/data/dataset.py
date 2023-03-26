import os

import numpy as np
from PIL import Image
from torch.utils import data
from torchvision.transforms import functional as TF
from torchvision import transforms


class IllustDataset(data.Dataset):
    def __init__(self, root_dir, c_folder, s_folder, image_size):
        assert isinstance(image_size, (tuple, int))
        self.root_dir = root_dir
        self.color_dir = os.path.join(root_dir, c_folder)
        self.sketch_dir = os.path.join(root_dir, s_folder)

        self.image_size = image_size
        self.list_files = os.listdir(self.color_dir)

        self.transform = transforms.Compose(
            (
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5),
            )
        )

    def __len__(self):
        return len(self.list_files)

    def __repr__(self):
        return f'dataset length: {len(self.list_files)}'

    def load_image(self, index):
        color_file = os.path.join(self.color_dir, self.list_files[index])
        sketch_file = os.path.join(self.sketch_dir, self.list_files[index])
        color = Image.open(color_file).convert('RGB')
        sketch = Image.open(sketch_file).convert('RGB')

        return color, sketch

    @staticmethod
    def _crop(colored,
              sketch,
              size=(512, 512),
              resize=True):

        size = (size, size) if isinstance(size, int) else size
        w, h = colored.width, colored.height

        if resize:
            scale = w / size[0] if h > w else h / size[1]

            w, h = int(w / scale), int(h / scale)
            colored = colored.resize((w, h), Image.BICUBIC)
            sketch = sketch.resize((w, h), Image.BICUBIC)

        x = np.random.randint(0, w - size[0]) if w != size[0] else 0
        y = np.random.randint(0, h - size[1]) if h != size[1] else 0

        colored = colored.crop((x, y, x + size[0], y + size[1]))
        sketch = sketch.crop((x, y, x + size[0], y + size[1]))

        return colored, sketch

    @staticmethod
    def _get_hint(image,
                  transform=None,
                  im_size=512,
                  max_strokes=120,
                  max_offs=8):
        Hinter = Hint(im_size, max_strokes, max_offs, transform)
        return Hinter(image)

    def __getitem__(self, index):
        colored, sketch = self.load_image(index)

        if np.random.random() > 0.5:
            colored = TF.hflip(colored)
            sketch = TF.hflip(sketch)

        colored, sketch = self._crop(colored, sketch, self.image_size)  # PIL

        sketch_l = sketch.convert('L')

        hint = self._get_hint(colored, self.transform, self.image_size)
        colored = self.transform(colored)

        style = self.transform(sketch)
        sketch = self.transform(sketch_l)

        return sketch, hint, colored, style
