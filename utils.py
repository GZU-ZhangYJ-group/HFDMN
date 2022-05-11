import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
import numpy as np
import cv2
import torch.nn as nn
from torchvision import models


# function to convert tensor to image

# Logging function for visdom


# sets a buffer to input images


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (
                max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


# set decay


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) >
                0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


# initislize weights from a normal distribution


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# extracts feature from vgg19 network's 2nd and 5th pooling layer


class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.select = ['9', '36']
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features[0], features[1]


def gaussian_filter(img, K_size=3, sigma=1.3):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape

    ## Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
    ## prepare Kernel
    K = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))

    K /= (2 * np.pi * sigma * sigma)
    K /= K.sum()
    tmp = out.copy()
    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])

    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

    return out 


def laplace_sharpen(input_image, c=-1):
    '''

    '''

    input_image_cp = np.copy(input_image)  # 输入图像的副本

    # 拉普拉斯滤波器
    laplace_filter = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1],
    ])

    input_image_cp = np.pad(input_image_cp, (1, 1), mode='constant', constant_values=0)  # 填充输入图像

    print(input_image_cp.shape)
    (m, n, tt) = input_image_cp.shape  # 填充后的输入图像的尺寸

    output_image = np.copy(input_image_cp)  # 输出图像

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            R = np.sum(laplace_filter * input_image_cp[i - 1:i + 2, j - 1:j + 2])  # 拉普拉斯滤波器响应

            output_image[i, j] = input_image_cp[i, j] + c * R

    output_image = output_image[1:m - 1, 1:n - 1]  # 裁剪

    return output_image


def laplacian_sharp2(img):
    ori_gray = img.convert('L')
    # ori_gray = img
    ori = np.array(img)
    ori_gray = np.array(ori_gray)
    weight = ori.shape[0]
    height = ori.shape[1]

    ori_pad = np.pad(ori_gray, ((1, 1), (1, 1)), 'edge')

    t1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    img = np.zeros((weight, height))
    for i in range(weight - 2):
        for j in range(height - 2):
            img[i, j] = np.sum(ori_pad[i:i + 3, j:j + 3] * t1)
            if img[i, j] < 0:
                img[i, j] = 0

    img_sharp = np.zeros((weight, height))
    img_sharp = ori_gray - img
    return img_sharp

def laplacian_sharp3(img):
    ori_gray = img.convert('L')
    ori = torch.tensor(img)
    ori_gray = np.array(ori_gray)
    weight = ori.shape[0]
    height = ori.shape[1]

    ori_pad = np.pad(ori_gray, ((1, 1), (1, 1)), 'edge')

    t1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    img = np.zeros((weight, height))
    for i in range(weight - 2):
        for j in range(height - 2):
            img[i, j] = np.sum(ori_pad[i:i + 3, j:j + 3] * t1)
            if img[i, j] < 0:
                img[i, j] = 0

    img_sharp = np.zeros((weight, height))
    img_sharp = ori_gray - img
    return img_sharp

def cv_Laplacian(img):
    dst = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    dst = cv2.convertScaleAbs(dst)
    return dst


def frequency_img(img):
    lf_imgs = torch.zeros(img.shape, dtype=np.float)
    hf_imgs = torch.zeros(img.shape, dtype=np.float)

    for i in range(img.shape[0]):
        temp = img[i].transpose((2, 1, 0)).astype(np.uint8)

        lf_img = torch.from_numpy(gaussian_filter(temp).transpose((2, 0, 1)))
        lf_imgs[i] = lf_img

        hf_img = torch.from_numpy(cv_Laplacian(temp).transpose((2, 0, 1)))
        hf_imgs[i] = hf_img

    # hf_img = laplace_sharpen(img)
    # hf_img = cv_Laplacian(img)
    freq_imgs = torch.cat((lf_imgs, hf_imgs), dim=1)

    return freq_imgs

def frequency_img_torch(img):
    lf_imgs = torch.zeros(img.shape, dtype=torch.float)
    hf_imgs = torch.zeros(img.shape, dtype=torch.float)

    for i in range(img.shape[0]):
        temp = img.permute((2, 1, 0))

        lf_img = gaussian_filter(temp).transpose((2, 0, 1))
        lf_imgs[i] = lf_img

        hf_img = torch.from_numpy(cv_Laplacian(temp).transpose((2, 0, 1)))
        hf_imgs[i] = hf_img

    # hf_img = laplace_sharpen(img)
    # hf_img = cv_Laplacian(img)
    freq_imgs = torch.cat((lf_imgs, hf_imgs), dim=1)

    return freq_imgs

def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps


def adjust_learning_rate(optimizer, epoch, category, lr_decay=0.5):

    # --- Decay learning rate --- #
    step = 20 if category == 'indoor' else 2

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))