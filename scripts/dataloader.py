import numpy as np
import torch
import cv2

from PIL import Image
from torch.utils.data import Dataset


class DIV2KDataset(Dataset):
    def __init__(self, root_dir, patch_size=192, rotation=False, transform=None, is_training=True, lr_scale=2,
                 full_scale=False):
        '''
        Args:
          root_dir: Directory with all the images
          patch_size: crop image size, if no crop, then input = -1
          transform: transform data based on torchvision transform function
          is_training: (True) - training dataset, (False) - validation dataset
          lr_scale: low resolution image scale
        '''
        self.root_dir = root_dir
        self.is_training = is_training
        self.rotation = rotation
        self.transform = transform
        self.patch_size = patch_size
        self.lr_scale = lr_scale
        self.full_scale = full_scale
        root_dir = str(root_dir.resolve().absolute())
        self.train_hr_image_path = root_dir + '\\DIV2K_train_HR'
        self.train_lr_image_path = root_dir + '\\DIV2K_train_LR_bicubic\\X' + str(lr_scale)
        self.valid_hr_image_path = root_dir + '\\DIV2K_valid_HR'
        self.valid_lr_image_path = root_dir + '\\DIV2K_valid_LR_bicubic\\X' + str(lr_scale)

    def __len__(self):
        return 800 if self.is_training else 100

    def __getitem__(self, idx):
        if self.is_training:
            img_path = self.train_lr_image_path + '\\' + \
                       str(idx + 1).zfill(4) + 'x' + str(self.lr_scale) + '.png'
            label_path = self.train_hr_image_path + \
                         '\\' + str(idx + 1).zfill(4) + '.png'
        else:
            img_path = self.valid_lr_image_path + '\\' + \
                       str(idx + 801).zfill(4) + 'x' + str(self.lr_scale) + '.png'
            label_path = self.valid_hr_image_path + \
                         '\\' + str(idx + 801).zfill(4) + '.png'

        img = Image.open(img_path)#.convert('YCbCr')
        width = img.size[0] * self.lr_scale
        height = img.size[1] * self.lr_scale
        dim = (width, height)
        img = np.array(img)#[:, :, 0]
        img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
        label = np.array(Image.open(label_path))#.convert('YCbCr'))[:, :, 0]

        if self.rotation:
            pass

        if self.transform:
            img = self.transform(img)
            label = self.transform(label)

        if self.full_scale:
            return img, label

        img_crop = []
        label_crop = []
        if self.patch_size != -1:
            for i in range(4):
                W = img.size()[1]
                H = img.size()[2]

                Ws = np.random.randint(0, W - self.patch_size + 1, 1)[0]
                Hs = np.random.randint(0, H - self.patch_size + 1, 1)[0]

                img_crop.append(img[:, Ws:Ws+self.patch_size, Hs:Hs + self.patch_size])
                label_crop.append(label[:, Ws:Ws+self.patch_size, Hs:Hs + self.patch_size])

        return torch.stack(img_crop), torch.stack(label_crop)


class Set5Dataset(Dataset):
    def __init__(self, root_dir='datasets\\Set5', transform=None, lr_scale=2):
        '''
        Args:
          root_dir: Directory with all the images
          patch_size: crop image size, if no crop, then input = -1
          transform: transform data based on torchvision transform function
          is_training: (True) - training dataset, (False) - validation dataset
          lr_scale: low resolution image scale
        '''
        self.root_dir = root_dir
        self.transform = transform
        self.lr_scale = lr_scale
        self.img_names = ['baby', 'bird', 'butterfly', 'head', 'woman']
        self.hr_image_path = root_dir + '\\Set5_HR'
        self.lr_image_path = root_dir + f'\\Set5_LR_x{lr_scale}'

    def __len__(self):
        return 5

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = self.lr_image_path + '\\' + img_name +'.png'
        label_path = self.hr_image_path + '\\' + img_name +'.png'
        img = Image.open(img_path)
        width = img.size[0] * self.lr_scale
        height = img.size[1] * self.lr_scale
        dim = (width, height)
        img = np.array(img)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
        label = np.array(Image.open(label_path))

        if self.transform:
            img = self.transform(img)
            label = self.transform(label)

        return img, label


class Set14Dataset(Dataset):
  def __init__(self, root_dir='datasets\\Set14', transform=None, lr_scale=4):
    self.root_dir = root_dir
    self.transform = transform
    self.lr_scale = lr_scale
    self.img_names = ['img_001', 'img_002', 'img_005', 'img_011', 'img_014']
    self.hr_image_path = root_dir + '\\Set14_HR'
    self.lr_image_path = root_dir + f'\\Set14_LR_x{lr_scale}'

  def __len__(self):
    return 5

  def __getitem__(self, idx):
    img_name = self.img_names[idx]
    img_path = self.lr_image_path + '\\' + img_name + '.png'
    label_path = self.hr_image_path + '\\' + img_name + '.png'
    img = Image.open(img_path)
    width = img.size[0] * self.lr_scale
    height = img.size[1] * self.lr_scale
    dim = (width, height)
    img = np.array(img)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    label = np.array(Image.open(label_path))

    if self.transform:
      img = self.transform(img)
      label = self.transform(label)

    return img, label