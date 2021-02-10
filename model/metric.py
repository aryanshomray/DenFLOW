import torch
from .metric_classes import PSNR, SSIM
import lpips

_psnr = PSNR()
_ssim = SSIM()
# _lpips = lpips.LPIPS(net="vgg")

def psnr(output, target):
    with torch.no_grad():
        return _psnr(output, target)


def ssim(output, target):
    with torch.no_grad():
        return _ssim(output, target)

# def lpips_(output, target):
#     with torch.no_grad():
#         return _lpips(2*output-1, 2*target-1)
