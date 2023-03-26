from tkinter import *
from tkinter import filedialog
from tkinter.colorchooser import askcolor

import os
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageTk
import webbrowser

from test import demo_test


def resize(image, min_size):
    w, h = image.size
    if h > min_size and w > min_size:
        if h < w:
            factor = min_size / h
        else:
            factor = min_size / w
        image = image.resize((int(w * factor), int(h * factor)))
    return image


class Paint(object):
    DEFAULT_PEN_SIZE = 6.0
    DEFAULT_COLOR = ((0, 0, 0), '#000000')
    MAX_SIZE = (512, 512)
    SAVE_PATH = './examples/'

    def __init__(self, device='cuda'):
        self.device = device

        self.root = Tk()
        self.root.geometry('1400x700')
        self.root.title('Demo')

        self.warn_label = Label(self.root, text='This is a free open-source project by Lin')
        self.warn_label.place(y=650, x=1310, anchor=NE)

        self.github = Label(self.root, text='My Github Page', fg='blue', cursor='hand2')
        self.github.place(y=670, x=1310, anchor=NE)
        self.github.bind('<Button-1>', lambda x: self.open_url('https://github.com/CytrusL'))

        self.sketch_frame = LabelFrame(self.root, text='Sketch Image', width=540, height=550)
        self.sketch_frame.place(y=40, x=200)

        self.pred_frame = LabelFrame(self.root, text='Generated Image', width=540, height=550)
        self.pred_frame.place(y=40, x=770)

        self.select = Button(self.root, text="Select Image", command=self.openImage)
        self.select.place(y=600, x=200, width=200, height=40)

        self.pen_button = Button(self.root, text='pen')
        self.pen_button.place(y=50, x=40, width=130, height=40)

        self.color_button = Button(self.root, text='color', command=self.choose_color)
        self.color_button.place(y=150, x=40, width=130, height=40)

        self.clear_button = Button(self.root, text='clear', command=self.clear)
        self.clear_button.place(y=250, x=40, width=130, height=40)

        self.choose_size_button = Scale(self.root, from_=1, to=12, orient=HORIZONTAL)
        self.choose_size_button.place(y=330, x=40, width=130, height=40)

        self.crop_var = IntVar()
        self.crop_button = Checkbutton(self.root, text='crop', variable=self.crop_var)
        self.crop_button.place(y=400, x=40, width=50, height=40)

        self.resize_var = IntVar()
        self.resize_button = Checkbutton(self.root, text='resize', variable=self.resize_var)
        self.resize_button.place(y=400, x=110, width=50, height=40)

        self.gen_button = Button(self.root, text='Generate', command=self.generate)
        self.gen_button.place(y=480, x=40, width=130, height=40)

        self.save_s_button = Button(self.root, text='Save Sketch', command=lambda: self.save_img('sketch'))
        self.save_s_button.place(y=600, x=590, width=150, height=40)

        self.save_g_button = Button(self.root, text='Save Generated', command=lambda: self.save_img('pred'))
        self.save_g_button.place(y=600, x=1160, width=150, height=40)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.canvas = None
        self.color_rec = []
        self.choose_size_button.set(self.DEFAULT_PEN_SIZE)
        self.im_size_label = None
        self.width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR

    def clear(self):
        if self.color_rec:
            self.canvas.delete(*self.color_rec)
            self.strokes = np.ones((self.label_h, self.label_w, 3)) * 150
            self.mask = np.zeros((self.label_h, self.label_w))

    def choose_color(self):
        self.color = askcolor(color=self.color[1])

    def openImage(self):
        img_path = filedialog.askopenfilenames(initialdir='./')
        self.file_name = img_path[-1].split('/')[-1]
        print(self.file_name)

        if img_path:
            img_open = Image.open(img_path[-1]).convert('RGB')

            if self.resize_var.get():
                img_open = resize(img_open, 512)
            if self.crop_var.get():
                img_open = img_open.crop((0, 0, 512, 512))

            self.sketch = img_open

            w, h = self.sketch.size
            self.w, self.h = w, h

            self.sketch_label = self._resize(img_open, w, h, self.MAX_SIZE)
            self.label_w, self.label_h = self.sketch_label.size
            self.img_label = ImageTk.PhotoImage(self.sketch_label)

            self.clear()

            if self.canvas:
                self.canvas.destroy()
            if self.im_size_label:
                self.im_size_label.destroy()

            self.strokes = np.ones((self.label_h, self.label_w, 3)) * 150
            self.mask = np.zeros((self.label_h, self.label_w))

            self.canvas = Canvas(self.sketch_frame, width=self.label_w, height=self.label_h)
            self.canvas.place(y=260, x=270, anchor=CENTER)
            self.canvas.create_image(0, 0, image=self.img_label, anchor=NW)

            self.im_size_label = Label(self.root, text=f'{w}x{h}')
            self.im_size_label.place(y=610, x=500)

            self.canvas.bind('<Button-1>', self.paint)

    @staticmethod
    def _resize(img, w, h, size):
        scale = w / size[0] if h < w else h / size[1]

        w, h = int(w / scale), int(h / scale)
        img = img.resize((w, h), Image.BICUBIC)

        return img

    def paint(self, event):
        self.width = self.choose_size_button.get()
        offs = self.width // 2
        paint_color = self.color

        if self.width % 2 == 0:
            x1, x2 = event.x - offs, event.x + offs
            y1, y2 = event.y - offs, event.y + offs
        else:
            x1, x2 = event.x - offs, event.x + offs + 1
            y1, y2 = event.y - offs, event.y + offs + 1
        self.color_rec.append(self.canvas.create_rectangle(x1, y1,
                                                           x2, y2,
                                                           outline=paint_color[1],
                                                           fill=paint_color[1]))
        self.strokes[y1:y2, x1:x2] = torch.tensor(paint_color[0], dtype=torch.uint8)
        self.mask[y1:y2, x1:x2] = 1

    def generate(self):
        strokes = Image.fromarray(self.strokes.astype(np.uint8)).resize((self.w, self.h), Image.NEAREST)
        mask = Image.fromarray(self.mask.astype(np.uint8)).resize((self.w, self.h), Image.NEAREST)
        strokes.save('./test.png')

        pred, pw, ph = demo_test(self.sketch, strokes, mask, device=self.device)
        self.pred = to_pil_image(pred[0])
        w, h = self.pred.size
        self.pred = self.pred.crop((0, 0, w-pw, h-ph))

        self.pred_label = self._resize(self.pred, w, h, self.MAX_SIZE)
        self.pred_img = ImageTk.PhotoImage(self.pred_label)

        self.pred_label = Label(self.pred_frame, image=self.pred_img)
        self.pred_label.place(y=260, x=270, anchor=CENTER)

    def save_img(self, type):
        if type == 'sketch':
            self.canvas.postscript(file=os.path.join(self.SAVE_PATH, 'sketch.eps'))
            img = Image.open(os.path.join(self.SAVE_PATH, 'sketch.eps'))
            img.save(os.path.join(self.SAVE_PATH, 'sketch.png'))
        elif type == 'pred':
            self.pred.save(os.path.join(self.SAVE_PATH, 'pred_'+self.file_name))

    def open_url(self, url):
        webbrowser.open_new(url)


if __name__ == '__main__':
    Paint()
