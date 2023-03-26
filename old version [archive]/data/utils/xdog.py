import cv2
from PIL import Image
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import threshold_otsu


def dog(img, size=(0, 0), k=1.6, sigma=0.5, gamma=1.):
    img1 = cv2.GaussianBlur(img, size, sigma)
    img2 = cv2.GaussianBlur(img, size, sigma * k)
    return img1 - gamma * img2


def xdog(img, sigma=0.4, k=2.5, gamma=0.95, epsilon=-0.5, phi=1e-9):
    aux = dog(img, sigma=sigma, k=k, gamma=gamma) / 255
    for i in range(0, aux.shape[0]):
        for j in range(0, aux.shape[1]):
            if aux[i, j] < epsilon:
                aux[i, j] = 1*255
            else:
                aux[i, j] = 255*(1 + np.tanh(phi * (aux[i, j])))
    return aux


def get_xdog(img, sigma=0.4, k=2.5, gamma=0.95, epsilon=-0.5, phi=1e9):
    xdog_image = xdog(img/255, sigma=sigma, k=k, gamma=gamma, epsilon=epsilon, phi=phi)
    return xdog_image * 255


if __name__ == '__main__':
    im = np.array(Image.open("../images/train/sketch/11995923_p0.jpg").convert("L"))
    sk = get_xdog(im)
    ret = Image.fromarray(sk).convert("RGB")
    ret.show()


class xDoG:
    def __init__(
            self,
            gamma: float = 0.95,
            psai: float = 1e9,
            eps: float = -1e1,
            k: float = 4.5,
            delta: float = 0.3,
    ) -> None:
        self.γ = gamma
        self.ϕ = psai
        self.ϵ = eps
        self.k = k
        self.σ = delta

    def __call__(self, img: np.ndarray) -> np.ndarray:
        x = (img[..., 0] + img[..., 1] + img[..., 2]) / 3

        gaussian_a = gaussian_filter(x, self.σ)
        gaussian_b = gaussian_filter(x, self.σ * self.k)

        dog = gaussian_a - self.γ * gaussian_b

        inf = dog < self.ε
        xdog = inf * 1 + ~inf * (1 - np.tanh(self.φ * dog))

        xdog -= xdog.min()
        xdog /= xdog.max()
        xdog = xdog >= threshold_otsu(xdog)
        xdog = 1 - xdog

        return xdog
