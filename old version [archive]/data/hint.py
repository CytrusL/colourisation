import numpy as np
from skimage.draw import disk


def generate_hints(
        colors,
        n_color,
        radius_range=(2, 32 + 1),
        proportion_range=(1e-4, 1e-2),
):
    if n_color <= 0:
        return np.zeros((*colors.shape[:2], 4))

    elif n_color >= 1:
        return np.concatenate([
            colors, np.ones((*colors.shape[:2], 1)),
        ], axis=-1)

    proportion = np.random.uniform(*proportion_range)

    activations = np.random.rand(*colors.shape[:2])
    samples = np.random.random_sample(size=colors.shape[:2])
    interest = (proportion * activations >= samples) * np.

    hints = np.zeros((*colors.shape[:2], 4), dtype=np.float32)
    for position in zip(*np.nonzero(interest)):
        radius = np.random.randint(*radius_range)
        rr, cc = disk(position, radius=radius, shape=colors.shape[:2])
        hints[rr, cc, -1] = 1.0
        hints[rr, cc, :3] = colors[position]

    return hints


if __name__ == '__main__':


# NEW

# import numpy as np
# import torch
# from PIL import Image, ImageDraw
# from torchvision import transforms


# class Hint(object):
#     def __init__(self,
#                  im_size,
#                  stroke_prob=0.1,
#                  thickness_range=(2, 8),
#                  move_range=((1, 20), (1, 20)),
#                  nodes_range=(1, 5),
#                  mode='random',
#                  transform=None
#                  ):
#         assert mode in ('random', 'dot', 'stroke')
#
#         self.im_size = (im_size, im_size) if isinstance(im_size, int) else im_size
#         self.w, self.h = self.im_size
#         self.mean = np.array([self.w // 2, self.h // 2])
#         self.cov = np.diag([(self.w // 4) ** 2, (self.h // 4) ** 2])
#
#         self.thickness_range = thickness_range
#         self.stroke_prob = stroke_prob
#         self.move_range = move_range
#         self.nodes_range = nodes_range
#         self.mode = mode
#
#         self.hint_transform = transform if transform else transforms.ToTensor()
#         self.transform = transforms.ToTensor()
#
#         self.hint = Image.new('RGB', self.im_size, (0, 0, 0))
#         self.mask = Image.new('L', self.im_size, 0)
#
#         self.B = (0, 0, 0)
#         self.W = (255, 255, 255)
#         self.G = (127, 127, 127)
#
#     def __call__(self, image):
#         image = np.array(image)
#         n_strokes = np.random.geometric(self.stroke_prob)
#         draw_hint = ImageDraw.Draw(self.hint)
#         draw_mask = ImageDraw.Draw(self.mask)
#
#         for _ in range(n_strokes):
#             rand = np.random.rand()
#             t = np.random.randint(*self.thickness_range)
#             x, y = -1, -1
#             while not (0 < x < self.w+t) or not (0 < y < self.h+t):
#                 x, y = np.random.multivariate_normal(self.mean, self.cov)
#                 x, y = int(x), int(y)
#
#             color = tuple(image[y-t:y+t, x-t:x+t].mean((0, 1)).astype(np.int32))
#
#             if self.mode == 'dot' or (self.mode == 'random' and rand < 0.5):
#                 if np.random.rand() < 0.5:
#                     draw_hint.ellipse((x-t, y-t, x+t, y+t), fill=color)
#                     draw_mask.ellipse((x-t, y-t, x+t, y+t), fill=255)
#                 else:
#                     draw_hint.rectangle((x-t, y-t, x+t, y+t), fill=color)
#                     draw_mask.rectangle((x-t, y-t, x+t, y+t), fill=255)
#
#             if self.mode == 'stroke' or (self.mode == 'random' and rand > 0.5):
#                 n_nodes = np.random.randint(*self.nodes_range)
#                 x_move = np.random.randint(*self.move_range[0])
#                 y_move = np.random.randint(*self.move_range[1])
#
#                 X = np.sort(np.random.randint(x-x_move, x+x_move, n_nodes))
#                 Y = np.random.randint(y-y_move, y+y_move, n_nodes)
#
#                 pts = [(X[i], Y[i]) for i in range(X.shape[-1])]
#                 offs = t * 0.5 - 1
#
#                 draw_mask.line(pts, width=t*2, fill=255, joint='curve')
#                 draw_hint.line(pts, width=t*2, fill=color, joint='curve')
#                 for i in range(X.shape[-1]):
#                     a = X[i] - offs, Y[i] - offs
#                     b = X[i] + offs, Y[i] + offs
#                     draw_mask.ellipse((*a, *b), fill=255)
#                     draw_hint.ellipse((*a, *b), fill=color)
#
#         return self.hint, self.mask
#
#
# if __name__ == '__main__':
#     image = Image.open('xxxxxx.jpg')
#     image.show()
#     hint = Hint(image.size)
#     x, y = hint(image)
#     x.show()

# class Hint(object):
#     def __init__(self,
#                  im_size: int,
#                  max_strokes: int,
#                  max_offs: int,
#                  transform=None):
#
#         im_size = (im_size, im_size) if isinstance(im_size, int) else im_size
#         h, w = im_size
#         self.mean = np.array([h // 2, w // 2])
#         self.cov = np.diag([(h // 4) ** 2, (w // 4) ** 2])
#
#         self.max_strokes = max_strokes
#         self.max_offs = max_offs
#
#         self.s_transform = transform if transform else transforms.ToTensor()
#         self.m_transform = transforms.ToTensor()
#
#     @staticmethod
#     def _color_change(strokes, mask, color, x, y, offs):
#         strokes[x:x+offs, y:y+offs] = color
#         mask[x:x+offs, y:y+offs] = 1
#         return strokes, mask
#
#     @staticmethod
#     def _mask_gen(image):
#         mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
#         return mask
#
#     @staticmethod
#     def _strokes_gen(image):
#         strokes = np.ones_like(image, dtype=np.uint8) * 150
#         return strokes
#
#     def _stroke(self, image, strokes, mask):
#         offs = np.random.randint(1, self.max_offs)
#         x, y = np.random.multivariate_normal(self.mean, self.cov).astype(np.int16)
#         while not (0 < x < 512-offs and 0 < y < 512-offs):
#             x, y = np.random.multivariate_normal(self.mean, self.cov).astype(np.int16) - offs//2
#
#         color = image[x:x+offs, y:y+offs].mean((0, 1)).reshape(1, 1, -1).astype(np.uint8)
#         # color = image[(x+x+offs)//2, (y+y+offs)//2]
#
#         strokes, mask = self._color_change(strokes, mask, color, x, y, offs)
#
#         return strokes, mask
#
#     def __call__(self, image):
#         image = np.asarray(image, dtype=np.float32)
#         strokes = self._strokes_gen(image)
#         mask = self._mask_gen(image)
#         num_strokes = np.random.geometric(0.125)
#
#         for _ in range(num_strokes):
#             strokes, mask = self._stroke(image, strokes, mask)
#
#         strokes = self.s_transform(strokes)
#         mask = self.m_transform(mask)
#
#         hint = torch.cat([strokes, mask], dim=0)
#
#         return hint

# import numpy as np
# import torch
# from torchvision import transforms
# from torchvision.utils import save_image
# from PIL import Image, ImageDraw
#
#
# class Hint(object):
#     def __init__(self,
#                  im_size: int,
#                  max_strokes: int,
#                  max_offs: int,
#                  transform=None):
#
#         self.im_size = (im_size, im_size) if isinstance(im_size, int) else im_size
#         h, w = self.im_size
#         self.mean = np.array([h // 2, w // 2])
#         self.cov = np.diag([(h // 4) ** 2, (w // 4) ** 2])
#
#         self.max_strokes = max_strokes
#         self.max_offs = max_offs
#
#         self.s_transform = transform if transform else transforms.ToTensor()
#         self.m_transform = transforms.ToTensor()
#
#     def _stroke(self, image, strokes, mask):
#         STROKES, MASK = None, None
#
#         while not (STROKES and MASK):
#             draw_strokes = ImageDraw.Draw(strokes)
#             draw_mask = ImageDraw.Draw(mask)
#
#             offs = np.random.randint(1, self.max_offs)
#             x, y = np.random.multivariate_normal(self.mean, self.cov).astype(np.int16) - offs//2
#             while not (0 < x < self.im_size[0]) or not (0 < y < self.im_size[1]):
#                 x, y = np.random.multivariate_normal(self.mean, self.cov).astype(np.int16) - offs // 2
#
#             color = image[x:x+offs, y:y+offs].mean((0, 1)).astype(np.uint8)
#
#             draw_strokes.rectangle((x, y, x+offs, y+offs), fill=tuple(color))
#             draw_mask.rectangle((x, y, x+offs, y+offs), fill=255)
#
#             STROKES = strokes
#             MASK = mask
#
#         return STROKES, MASK
#
#     def __call__(self, image):
#         image = np.asarray(image, dtype=np.float32)
#         strokes = Image.new('RGB', self.im_size, (150, 150, 150))
#         mask = Image.new('L', self.im_size, 0)
#         num_strokes = np.random.geometric(0.125)
#
#         for _ in range(num_strokes):
#             strokes, mask = self._stroke(image, strokes, mask)
#
#         strokes = self.s_transform(strokes)
#         mask = self.m_transform(mask)
#
#         hint = torch.cat([strokes, mask], dim=0)
#
#         return hint
#
#
# if __name__ == '__main__':
#     hinter = Hint(512, 100, 8)
#     image = Image.open('../97149746_p0.jpg')
#     hint = hinter(image)
#
#     save_image(hint[:3], './test.png')
#     # hint.show()
