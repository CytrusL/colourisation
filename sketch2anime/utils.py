import torch
from torchvision.utils import save_image


def save_examples(x, h, y, y_fake, epoch, folder):
    save_image(y_fake * 0.5 + 0.5, folder + f"/gen_{epoch}.png")
    save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
    save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    save_image(h[:, :3] * 0.5 + 0.5, folder + f"/hint_{epoch}.png")


def save_checkpoint(model, optimizer, epoch, cur_iter, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "cur_iter": cur_iter
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr, device='cuda'):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]
    cur_iter = checkpoint["cur_iter"]

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return epoch, cur_iter
