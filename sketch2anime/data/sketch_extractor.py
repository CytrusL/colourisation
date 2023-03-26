import argparse
import os

import cv2
import torch
from torchvision import transforms
from torchvision.utils import save_image

from data.utils.xdog import get_xdog
from data.utils.ss_model import get_ss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', type=str, default='./input', help='Path to input directory')
    parser.add_argument('--output_dir', '-o', type=str, default='./output', help='Path to output directory')
    parser.add_argument('--model', '-m', type=str, default='model_gan', help='Name of the model, default: model_gan')
    parser.add_argument('--use_ss', '-s', action='store_true', default=True, help='Use Sketch Simplification')
    parser.add_argument('--use_xdog', '-x', action='store_true', default=True, help='Use XDoG')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    if args.use_ss:
        ss_model, immean, imstd = get_ss()
        # Download weights: https://mega.nz/folder/2lUn1YbY#JhTkB1vdaBMeTCSs37iTVA
        ss_model.load_state_dict(torch.load('./utils/' + args.model + '.pth'))
        ss_model.eval()

    file_names = os.listdir(args.input_dir)
    file_len = len(file_names)

    for iter, file in enumerate(file_names):
        img = cv2.imread(os.path.join(args.input_dir, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if args.use_ss:
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
            bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, se)

            data = cv2.divide(gray, bg, scale=255)
            data -= 1

            h, w = data.shape

            pw = 8 - (w % 8) if w % 8 != 0 else 0
            ph = 8 - (h % 8) if h % 8 != 0 else 0

            data = ((transforms.ToTensor()(data) - immean) / imstd).unsqueeze(0)
            if pw != 0 or ph != 0:
                data = torch.nn.ReplicationPad2d((0, pw, 0, ph))(data).data

            if use_cuda:
                pred = ss_model.cuda().forward(data.cuda()).float()
            else:
                pred = ss_model.forward(data)

            pred = pred[:, :, 0:-ph, 0:-pw]
            save_image(pred[0], os.path.join(args.output_dir, 'ss_'+file))

        if args.use_xdog:
            data = get_xdog(gray)
            cv2.imwrite(os.path.join(args.output_dir, 'xdog_'+file), data)

        print(f'Done: {iter+1} / {file_len}')
    print('Finish')
