import cv2
import numpy as np
from PIL import Image


class xDoG:
    def __init__(self, gamma=0.95, epsilon=-1e1, psai=1e9, k=4.5):
        self.gamma = gamma
        self.epsilon = epsilon
        self.psai = psai
        self.k = k

    def __call__(self, img, sigma):
        x = (img[..., 0] + img[..., 1] + img[..., 2]) / 3

        gaussian_a = cv2.GaussianBlur(x, (0, 0), sigma)
        gaussian_b = cv2.GaussianBlur(x, (0, 0), sigma * self.k)
        aux = gaussian_a - self.gamma * gaussian_b

        for i in range(0, aux.shape[0]):
            for j in range(0, aux.shape[1]):
                if aux[i, j] < self.epsilon:
                    aux[i, j] = 1
                else:
                    aux[i, j] = 1 * np.tanh(self.psai * aux[i, j])
        return aux

if __name__ == '__main__':
    xdog = xDoG()
    im = np.array(Image.open('og.jpg'))
    im = xdog(im/255, 0.4)
    print(im.shape)
    Image.fromarray(im.astype(np.uint8) * 255).save('xdog.png')

