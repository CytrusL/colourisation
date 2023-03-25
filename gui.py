from tkinter import *
from tkinter import filedialog, ttk
from tkinter.colorchooser import askcolor

import os

import cv2
import numpy as np
import albumentations as A
import torch
import yaml
from PIL import Image, ImageTk
from skimage.draw import disk
from albumentations.pytorch import ToTensorV2
from albumentations.pytorch.functional import img_to_tensor
import webbrowser
import sv_ttk

from models import Generator
from utils import tensor_to_numpy
from data.dataset import scale_resize


def max_scale_resize(image, max_size):
    w, h = image.size
    if h > max_size or w > max_size:
        if h < w:
            factor = max_size / w
        else:
            factor = max_size / h
        image = image.resize((int(w * factor), int(h * factor)))
    return image


class Paint(object):
    DEFAULT_PEN_SIZE = 5
    DEFAULT_COLOR = ((0, 0, 0), '#000000')
    MAX_SIZE = 512
    SAVE_PATH = './saves/'
    # Preprocess
    transform = A.Compose(
        (
            A.Normalize(0.5, 0.5),
            ToTensorV2(),
        )
    )
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    device = config['train']['device']

    model = Generator(config['model']['gen']['in_ch'],
                      config['model']['gen']['depth'],
                      config['model']['gen']['dims'],
                      config['model']['gen']['attn'],
                      config['model']['gen']['drop_path_rate'],
                      config['model']['gen']['layer_scale_init_value'],
                      config['model']['gen']['scale'],
                      training=False).to(device).eval()

    model.load_state_dict(torch.load(config['model']['dir']['root'] +
                                     config['model']['dir']['gen'] +
                                     config['model']['dir']['ext'],
                                     map_location=device))

    def __init__(self):
        self.root = Tk()
        self.root.geometry('1350x610')
        self.root.title('Demo')

        self.sketch_frame = ttk.LabelFrame(self.root, text='Sketch Image', width=540, height=550)
        self.sketch_frame.place(y=30, x=30)

        self.pred_frame = ttk.LabelFrame(self.root, text='Generated Image', width=540, height=550)
        self.pred_frame.place(y=30, x=780)

        self.tool_frame = ttk.LabelFrame(self.root, text='Tools', width=150, height=210)
        self.tool_frame.place(y=40, x=600)

        self.func_frame = ttk.Frame(self.root, width=150, height=220)
        self.func_frame.place(y=260, x=600)

        self.select = ttk.Button(self.func_frame, text="Select Image", command=self.openImage)
        self.select.place(y=10, x=10, width=130, height=40)

        self.color_button = ttk.Button(self.tool_frame, text='color', command=self.choose_color)
        self.color_button.place(y=20, x=10, width=130, height=40)

        self.clear_button = ttk.Button(self.tool_frame, text='clear', command=self.clear)
        self.clear_button.place(y=80, x=10, width=130, height=40)

        self.pen_size = ttk.Label(self.tool_frame, text='Pen Size')
        self.pen_size.place(y=140, x=10, width=130, height=20)
        self.size_button = ttk.Scale(self.tool_frame, from_=1, to=12, orient=HORIZONTAL)
        self.size_button.place(y=160, x=10, width=130, height=30)

        self.preprocess_var = IntVar()
        self.preprocess_button = ttk.Checkbutton(self.func_frame, text='Preprocess', variable=self.preprocess_var)
        self.preprocess_button.place(y=165, x=10, width=130, height=40)

        self.gen_button = ttk.Button(self.root, text='Generate', command=self.generate, style="Accent.TButton")
        self.gen_button.place(y=510, x=610, width=130, height=40)

        self.save_s_button = ttk.Button(self.func_frame, text='Save Sketch', command=lambda: self.save_img('sketch'))
        self.save_s_button.place(y=60, x=10, width=130, height=40)

        self.save_g_button = ttk.Button(self.func_frame, text='Save Illustration', command=lambda: self.save_img('pred'))
        self.save_g_button.place(y=110, x=10, width=130, height=40)

        self.canvas = None
        self.im_size_label = None
        self.color_rec = []
        self.size_button.set(self.DEFAULT_PEN_SIZE)
        self.eraser_on = False
        self.radius = self.size_button.get()
        self.color = self.DEFAULT_COLOR
        self.tag_id = 0

        sv_ttk.set_theme("light")
        self.root.mainloop()

    def clear(self):
        if self.color_rec:
            self.canvas.delete(*self.color_rec)
            self.hint = np.zeros((self.label_h, self.label_w, 4))

    def choose_color(self):
        self.color = askcolor(color=self.color[1])

    def openImage(self):
        img_path = filedialog.askopenfilenames(initialdir='./')
        self.file_name = img_path[-1].split('/')[-1]
        print(self.file_name)

        if img_path:
            img_open = Image.open(img_path[-1]).convert('L').convert('RGB')

            if self.preprocess_var.get():
                img_open = scale_resize(img_open, 512)

            self.sketch = np.array(img_open)
            self.w, self.h = self.sketch.shape[:2]

            self.sketch_label = max_scale_resize(img_open, self.MAX_SIZE)
            self.label_w, self.label_h = self.sketch_label.size
            self.img_label = ImageTk.PhotoImage(self.sketch_label)

            self.clear()

            if self.canvas:
                self.canvas.destroy()
            if self.im_size_label:
                self.im_size_label.destroy()

            self.hint = np.zeros((self.label_h, self.label_w, 4), dtype=np.float32)

            self.canvas = Canvas(self.sketch_frame, width=self.label_w, height=self.label_h)
            self.canvas.place(y=260, x=270, anchor=CENTER)
            self.canvas.create_image(0, 0, image=self.img_label, anchor=NW)

            self.im_size_label = Label(self.root, text=f'{self.w}x{self.h}')
            self.im_size_label.place(y=480, x=610, width=130, height=20)

            self.canvas.bind('<B1-Motion>', self.paint)
            self.canvas.bind('<Button-1>', self.paint)

    def paint(self, event):
        self.radius = self.size_button.get()
        paint_color = self.color

        x1, x2 = event.x - self.radius, event.x + self.radius
        y1, y2 = event.y - self.radius, event.y + self.radius
        self.canvas.create_oval(x1, y1,
                                x2, y2,
                                outline=paint_color[1],
                                fill=paint_color[1], tags='tag' + str(self.tag_id))
        self.color_rec.append('tag' + str(self.tag_id))
        self.tag_id += 1

        rr, cc = disk((event.y, event.x), self.radius, shape=(self.label_h, self.label_w))
        self.hint[rr, cc, -1] = 1.
        self.hint[rr, cc, :3] = np.array(paint_color[0]) / 127.5 - 1.

    def generate(self):
        hint = cv2.resize(self.hint, (self.h, self.w), cv2.INTER_CUBIC)

        x = self.transform(image=self.sketch)['image'].unsqueeze(0)
        h = img_to_tensor(hint).unsqueeze(0)

        ph = 32 - (x.shape[2] % 32) if x.shape[2] % 32 != 0 else 0
        pw = 32 - (x.shape[3] % 32) if x.shape[3] % 32 != 0 else 0

        if pw != 0 or ph != 0:
            x = torch.nn.ReplicationPad2d((0, pw, 0, ph))(x).data
            h = torch.nn.ReplicationPad2d((0, pw, 0, ph))(h).data

        x, h = x.to(self.device), h.to(self.device)

        with torch.no_grad():
            self.pred = self.model(x, h)

        self.pred = tensor_to_numpy(self.pred * 0.5 + 0.5)[0:self.sketch.shape[0], 0:self.sketch.shape[1], :]

        assert self.pred.shape == self.sketch.shape
        self.pred = Image.fromarray(self.pred)

        self.pred_label = max_scale_resize(self.pred, self.MAX_SIZE)
        self.pred_img = ImageTk.PhotoImage(self.pred_label)

        self.pred_label = Label(self.pred_frame, image=self.pred_img)
        self.pred_label.place(y=260, x=270, anchor=CENTER)

    def save_img(self, type):
        if type == 'sketch':
            self.canvas.postscript(file=os.path.join(self.SAVE_PATH, 'sketch.eps'))
            img = Image.open(os.path.join(self.SAVE_PATH, 'sketch.eps'))
            img.save(os.path.join(self.SAVE_PATH, 'sketch.png'))
        elif type == 'pred':
            self.pred.save(os.path.join(self.SAVE_PATH, 'pred_' + self.file_name))

    def open_url(self, url):
        webbrowser.open_new(url)


if __name__ == '__main__':
    Paint()
