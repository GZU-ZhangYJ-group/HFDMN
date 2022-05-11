
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
from utils import  adjust_learning_rate
from Freq_Mutil_Net import HF_Net
import os
import pytorch_ssim

plt.switch_backend('agg')


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
parser.add_argument('-learning_rate', help='Set the learning rate', default=2e-6, type=float)
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
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nnetwork_height: {}\nnetwork_width: {}\n'
      'num_dense_layer: {}\ngrowth_rate: {}\nlambda_loss: {}\nlambda_freq:{}'.format(learning_rate, crop_size,
      train_batch_size, val_batch_size, network_height, network_width, num_dense_layer, growth_rate, lambda_loss,lambda_freq))




# --- Gpu device --- #
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Define the network --- #
net = HF_Net()

# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9,0.999))

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

num_epochs = 10
train_data_dir = './'

# --- Calculate all trainable parameters in network --- #
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))


# --- Load training data and validation/test data --- #
train_data_loader = DataLoader(TrainData_Freq(crop_size, train_data_dir), batch_size=train_batch_size, shuffle=True,num_workers=16)


for epoch in range(0,num_epochs):
    psnr_list = []
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch, category='outdoor')
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
        dehaze, dehaze_hf, dehaze_mid = net(haze,haze_hf)

        #smooth_loss = F.smooth_l1_loss(dehaze, gt)
        smooth_loss = nn.L1Loss()(dehaze, gt)

        perceptual_loss = loss_network(dehaze, gt)
        hf_smooth_loss = nn.L1Loss()(dehaze_hf, gt_hf)
        #mid_smooth_loss = nn.L1Loss()(dehaze_mid,gt)
        
        hf_ssim = hf_ssim_loss(dehaze_hf,gt_hf)
        dehaze_ssim = dehaze_ssim_loss(dehaze,gt)
        
        loss = 1.0 * smooth_loss + lambda_loss*perceptual_loss + hf_smooth_loss * lambda_freq + 0.1 * (1.5 - dehaze_ssim - 0.5* hf_ssim)#+ mid_smooth_loss

        loss.backward()
        optimizer.step()

        # --- To calculate average PSNR --- #
        # psnr_list.extend(to_psnr(dehaze, gt))
        print(
            '\rEpoch %s/%s;Iteration %s/%s; loss_G: %0.6f;feature_loss:%f;perceptual_loss:%f;freq_loss:%f; dehaze_ssim:%f;hf_ssim:%f;lr: %f ' %
            (epoch, num_epochs, batch_id, len(train_data_loader), loss.item(), smooth_loss,dehaze_ssim, hf_ssim,perceptual_loss,
             hf_smooth_loss, optimizer.param_groups[0]['lr']))
        #if not (batch_id % 100):
            #print('Epoch: {0}, Iteration: {1}'.format(epoch, batch_id))
        

    # --- Calculate the average training PSNR in one epoch --- #
    #train_psnr = sum(psnr_list) / len(psnr_list)

    # --- Save the network parameters --- #
    torch.save(net.state_dict(), ('./output_%s/epoch_%d_%0.4f_%0.4f.pk' % (category,epoch, 0, 0)))
    #lr_scheduler_G.step()

    # --- Use the evaluation model in testing --- #
    #net.eval()

    # val_psnr, val_ssim = validation(net, val_data_loader, device, category)
    one_epoch_time = time.time() - start_time
    print("one_epoch_time:", one_epoch_time)
    # print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category)
    
    # --- update the network weight --- #
    # if val_psnr >= old_val_psnr:
    #     torch.save(net.state_dict(), '{}_haze_best_{}_{}'.format(category, network_height, network_width))
    #     old_val_psnr = val_psnr
    



