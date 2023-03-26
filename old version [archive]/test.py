import yaml
import torch
from torch import nn
from torchvision import transforms

from models import Generator

# Preprocess
transform = transforms.Compose(
    (
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    )
)


# GUI demo test
def demo_test(sketch, strokes, mask, device='cuda'):
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    sketch_l = sketch.convert('L')
    w, h = sketch.size

    style = transform(sketch).unsqueeze(0)
    x = transform(sketch_l).unsqueeze(0)
    strokes = transform(strokes).unsqueeze(0)
    mask = transforms.ToTensor()(mask).unsqueeze(0)

    hint = torch.cat([strokes, mask], dim=1)

    pw = 32 - (w % 32) if w % 32 != 0 else 0
    ph = 32 - (h % 32) if h % 32 != 0 else 0

    if pw != 0 or ph != 0:
        x = torch.nn.ReplicationPad2d((0, pw, 0, ph))(x).data
        style = torch.nn.ReplicationPad2d((0, pw, 0, ph))(style).data
        hint = torch.nn.ReplicationPad2d((0, pw, 0, ph))(hint).data

    x, hint, style = x.to(device), hint.to(device), style.to(device)

    gen = nn.DataParallel(Generator().to(device))
    checkpoint = torch.load(config['model']['dir']['gen'],
                            map_location=config['train']['device'])
    gen.load_state_dict(checkpoint["state_dict"])

    gen.eval()
    with torch.no_grad():
        pred = gen(x, hint, style)

    return pred * 0.5 + 0.5, pw, ph
