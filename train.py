
# --- Imports --- #
import time
import torch
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_data import TrainData_Freq
from torchvision.models import vgg16
from models import LossNetwork
from utils import LambdaLR
from Freq_Mutil_Net import HF_Net
import sys
import os
import pytorch_ssim

plt.switch_backend('agg')

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
parser.add_argument('-learning_rate', help='Set the learning rate', default=5e-4, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[240, 240], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=16, type=int)
parser.add_argument('-network_height', help='Set the network height (row)', default=3, type=int)
parser.add_argument('-network_width', help='Set the network width (column)', default=6, type=int)
parser.add_argument('-num_dense_layer', help='Set the number of dense layer in RDB', default=4, type=int)
parser.add_argument('-growth_rate', help='Set the growth rate in RDB', default=16, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-ex', help='The index of the experiments', default=1, type=str)
parser.add_argument('-lambda_freq', help='lambda freq loss', default=0.5, type=float)

args = parser.parse_args()

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
network_height = args.network_height
network_width = args.network_width
num_dense_layer = args.num_dense_layer
growth_rate = args.growth_rate
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
lambda_freq = args.lambda_freq

print('--- Hyper-parameters for training ---')
print(
    'learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nnetwork_height: {}\nnetwork_width: {}\n'
    'num_dense_layer: {}\ngrowth_rate: {}\nlambda_loss: {}\nlambda_freq:{}'.format(learning_rate,
                                                                                                 crop_size,
                                                                                                 train_batch_size,
                                                                                                 val_batch_size,
                                                                                                 network_height,
                                                                                                 network_width,
                                                                                                 num_dense_layer,
                                                                                                 growth_rate,
                                                                                                 lambda_loss,
                                                                                                 lambda_freq))

num_epochs = 80
train_data_dir = './'

# --- Gpu device --- #
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
net = HF_Net()

# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                   lr_lambda=LambdaLR(num_epochs, 0, decay_start_epoch=30).step)

# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)

# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
for param in vgg_model.parameters():
    param.requires_grad = False

loss_network = LossNetwork(vgg_model)
loss_network.eval()
# --- SSIM loss --- #
hf_ssim_loss = pytorch_ssim.SSIM()
dehaze_ssim_loss = pytorch_ssim.SSIM()

# --- Calculate all trainable parameters in network --- #
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))

train_data_loader = DataLoader(TrainData_Freq(crop_size, train_data_dir), batch_size=train_batch_size, shuffle=True,
                               num_workers=4)

for epoch in range(num_epochs):
    psnr_list = []
    start_time = time.time()
    net.train()
    for batch_id, train_data in enumerate(train_data_loader):
        haze, gt, haze_hf, gt_hf = train_data
        haze = haze.to(device)
        gt = gt.to(device)
        haze_hf = haze_hf.to(device)
        gt_hf = gt_hf.to(device)

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        dehaze, dehaze_hf, dehaze_mid = net(haze, haze_hf)

        smooth_loss = nn.L1Loss()(dehaze, gt)

        perceptual_loss = loss_network(dehaze, gt)
        hf_smooth_loss = nn.L1Loss()(dehaze_hf, gt_hf)

        hf_ssim = hf_ssim_loss(dehaze_hf, gt_hf)
        dehaze_ssim = dehaze_ssim_loss(dehaze, gt)

        loss = 1.0 * smooth_loss + lambda_loss * perceptual_loss + hf_smooth_loss * lambda_freq + 0.1 * (
                1.5 - dehaze_ssim - 0.5 * hf_ssim)

        loss.backward()
        optimizer.step()

        sys.stdout.write(
            '\rEpoch %s/%s;Iteration %s/%s; loss_G: %0.6f;feature_loss:%f;perceptual_loss:%f;freq_loss:%f; dehaze_ssim:%f;hf_ssim:%f;lr: %f ' %
            (epoch, num_epochs, batch_id, len(train_data_loader), loss.item(), smooth_loss, dehaze_ssim, hf_ssim,
             perceptual_loss,
             hf_smooth_loss, optimizer.param_groups[0]['lr']))

    # --- Save the network parameters --- #
    torch.save(net.state_dict(), ('./pretrained_model/epoch_%d.pk' % (epoch)))
    lr_scheduler_G.step()
