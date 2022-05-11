

# --- Imports --- #
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Freq_Mutil_Net import HF_Net
from math import log10
from skimage import measure
import torchvision.utils as utils

import math
import numpy as np
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import os

import cv2
def cv_Laplacian(img):
    dst = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    dst = cv2.convertScaleAbs(dst)
    return dst

def save_image(dehaze, image_name, category):
    dehaze_images = torch.split(dehaze, 1, dim=0)
    batch_num = len(dehaze_images)

    if not os.path.exists('./{}_results/'.format(category)):
        os.makedirs('./{}_results/'.format(category))
    for ind in range(batch_num):
        utils.save_image(dehaze_images[ind], '{}_results/{}'.format(category, image_name[0]))

def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in
                      range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [measure.compare_ssim(dehaze_list_np[ind], gt_list_np[ind], data_range=1, multichannel=True) for ind in
                 range(len(dehaze_list))]

    return ssim_list


def psnr_indoor(pred, gt):
    pred = pred.clamp(0, 1).cpu().detach().numpy()
    gt = gt.clamp(0, 1).cpu().numpy()
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(1.0 / rmse)

def validation(net, val_data_loader, device, category, save_tag=False):

    for batch_id, val_data in enumerate(val_data_loader):
        with torch.no_grad():
            haze, haze_hf, image_name, _ = val_data

            haze = haze.to(device)
            haze_hf = haze_hf.to(device)

            dehaze, mid, chf_ = net(haze, haze_hf)

        # --- Save image --- #
        if save_tag:
            save_image(dehaze, image_name, category)

class ValData_Single(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()
        self.haze_imgs_dir = val_data_dir
        haze_names = []

        for file_name in os.listdir(self.haze_imgs_dir):
            haze_names.append(file_name)

        self.haze_names = haze_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        haze_name = self.haze_names[index]
        haze_img = Image.open(self.val_data_dir  + haze_name)
        # --- Transform to tensor --- #

        haze_hf_img = cv_Laplacian(np.array(haze_img))
        haze0 = Compose([ToTensor()])(haze_img)

        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_haze_hf = Compose([ToTensor()])

        haze = transform_haze(haze_img)
        haze_hf = transform_haze_hf(haze_hf_img)

        return haze, haze_hf, haze_name, haze0

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)


if __name__ == '__main__':
    # --- Parse hyper-parameters  --- #

    # --- Set category-specific hyper-parameters  --- #
    pic_format = '.png'
    category = 'outdoor'

    val_data_dir = './test_imgs/'
    is_save = True

    # --- Gpu device --- #
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(0)
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Validation data loader --- #
    val_data_loader = DataLoader(ValData_Single(val_data_dir), batch_size=1,
                                 shuffle=False, num_workers=1)

    # --- Define the network --- #
    net = HF_Net()

    # --- Multi-GPU --- #
    net = net.to(device)
    net = nn.DataParallel(net, device_ids=device_ids)

    pretrained_model = 'pretrained_model/{}_best.pk'.format(category, 14)
    if os.path.exists(pretrained_model):

        net.load_state_dict(torch.load(pretrained_model))
        # --- Use the evaluation model in testing --- #
        net.eval()
        validation(net, val_data_loader, device, category, save_tag=is_save)


