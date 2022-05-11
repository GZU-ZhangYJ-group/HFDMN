
# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import os


# --- Training dataset --- #
class TrainData_Freq(data.Dataset):
    def __init__(self, crop_size, train_data_dir):
        super().__init__()

        self.haze_imgs_dir = os.path.join(train_data_dir, 'B')
        haze_names = []
        gt_names = []

        for file_name in os.listdir(self.haze_imgs_dir):
            haze_names.append(file_name)
            gt_names.append(file_name.split('_')[0])

        self.haze_names = haze_names
        self.gt_names = gt_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir

        
    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]

        haze_img = Image.open(self.train_data_dir + 'B/' + haze_name)
        
        width, height = haze_img.size
        if width < crop_width or height < crop_height:
            raise Exception('Bad image size: {}'.format(haze_name))
            #print('Bad image size: {}'.format(haze_name))
            #index = random.randint(1, len(self.haze_names)-1)
           # haze_name = self.haze_names[index]
            #gt_name = self.gt_names[index]
           # haze_img = Image.open(self.train_data_dir + 'B/' + haze_name)
          #  width, height = haze_img.size

        
        haze_hf_img = Image.open(self.train_data_dir + 'haze_hf/' + haze_name)
        try:
            gt_img = Image.open(self.train_data_dir + 'clear/' + gt_name + '.jpg')
            gt_hf_img = Image.open(self.train_data_dir + 'clear_hf/' + gt_name + '.jpg')
        except:
            gt_img = Image.open(self.train_data_dir + 'clear/' + gt_name + '.png').convert('RGB')
            gt_hf_img = Image.open(self.train_data_dir + 'clear_hf/' + gt_name + '.png').convert('RGB')
       

        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        #freq
        haze_hf_crop_img = haze_hf_img.crop((x, y, x + crop_width, y + crop_height))
        gt_hf_crop_img = gt_hf_img.crop((x, y, x + crop_width, y + crop_height))

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])

        haze = transform_haze(haze_crop_img)
        gt = transform_gt(gt_crop_img)

        # freq
        transform_haze_hf = Compose([ToTensor()])
        transform_gt_hf = Compose([ToTensor()])

        haze_hf = transform_haze_hf(haze_hf_crop_img)
        gt_hf = transform_gt_hf(gt_hf_crop_img)

        # --- Check the channel is 3 or not --- #
        if list(haze.shape)[0] is not 3 or list(gt.shape)[0] is not 3 \
                or list(haze_hf.shape)[0] is not 3 \
                or list(gt_hf.shape)[0] is not 3:
            print(haze.shape,gt.shape)
            raise Exception('Bad image channel: {}'.format(haze_name))

        return haze, gt, haze_hf, gt_hf

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)
