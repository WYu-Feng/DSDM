import os
import glob
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import cv2
import torch.nn.functional as F

class AlignedDataset_all(BaseDataset):
    def __init__(self, opt, augment_flip=True, crop_patch=True, mode = 'train'):
        BaseDataset.__init__(self, opt)
        self.augment_flip = augment_flip
        self.crop_patch = crop_patch
        self.opt = opt
        self.A_paths = list()
        self.B_paths = list()

        self.mode = mode
        if self.mode == 'train':
        # Tranin origin----------------------------------------------------------------------------------------------------------
            self.A_paths_desnow = glob.glob(os.path.join('./datasets', 'Image desnowing', 'Snow100K-L', 'train', 'synthetic', '*.jpg'))
            self.B_paths_desnow = glob.glob(os.path.join('./datasets', 'Image desnowing', 'Snow100K-L', 'train', 'gt', '*.jpg'))
            self.A_paths = self.A_paths + self.A_paths_desnow
            self.B_paths = self.B_paths + self.B_paths_desnow

            self.A_paths_derain = glob.glob(os.path.join('./datasets', 'Image deraining', 'Rain13K', 'input', '*.jpg'))
            self.B_paths_derain = glob.glob(os.path.join('./datasets', 'Image deraining', 'Rain13K', 'target', '*.jpg'))
            for _ in range(len(self.A_paths_desnow)//len(self.A_paths_derain)):
                self.A_paths = self.A_paths + self.A_paths_derain
                self.B_paths = self.B_paths + self.B_paths_derain

            self.A_paths_dehaz = glob.glob(os.path.join('./datasets', 'Image dehazing', 'SOTS', 'outdoor', 'hazy', '*.png'))[100:]
            self.B_paths_dehaz = glob.glob(os.path.join('./datasets', 'Image dehazing', 'SOTS', 'outdoor', 'gt', '*.png'))[100:]
            for _ in range(len(self.A_paths_desnow)//len(self.A_paths_dehaz)):
                self.A_paths = self.A_paths + self.A_paths_dehaz
                self.B_paths = self.B_paths + self.B_paths_dehaz

            self.A_paths_deblur = glob.glob(os.path.join('./datasets', 'Image deblurring', 'GoPro', 'train', 'input', '*.png'))
            self.B_paths_deblur = glob.glob(os.path.join('./datasets', 'Image deblurring', 'GoPro', 'train', 'target', '*.png'))
            for _ in range(len(self.A_paths_desnow)//len(self.A_paths_deblur)):
                self.A_paths = self.A_paths + self.A_paths_deblur
                self.B_paths = self.B_paths + self.B_paths_deblur

            self.A_paths_light = glob.glob(os.path.join('./datasets', 'Low-light enhancement', 'LOL', 'train', 'low', '*.png'))
            self.B_paths_light = glob.glob(os.path.join('./datasets', 'Low-light enhancement', 'LOL', 'train', 'high', '*.png'))
            for _ in range(len(self.A_paths_desnow)//len(self.A_paths_light)):
                self.A_paths = self.A_paths + self.A_paths_light
                self.B_paths = self.B_paths + self.B_paths_light
        else: 
            self.A_paths_desnow = glob.glob(os.path.join('./datasets', 'Image desnowing', 'Snow100K-L', 'test', 'synthetic', '*.jpg'))
            self.B_paths_desnow = glob.glob(os.path.join('./datasets', 'Image desnowing', 'Snow100K-L', 'test', 'gt', '*.jpg'))
            self.A_paths = self.A_paths + self.A_paths_desnow
            self.B_paths = self.B_paths + self.B_paths_desnow

            self.A_paths_derain = glob.glob(os.path.join('./datasets', 'Image deraining', 'rain1400', 'test', 'rainy_image', '*.png'))
            self.B_paths_derain = glob.glob(os.path.join('./datasets', 'Image deraining', 'rain1400', 'test', 'ground_truth', '*.png'))
            for _ in range(len(self.A_paths_desnow)//len(self.A_paths_derain)):
                self.A_paths = self.A_paths + self.A_paths_derain
                self.B_paths = self.B_paths + self.B_paths_derain

            self.A_paths_dehaz = glob.glob(os.path.join('./datasets', 'Image dehazing', 'SOTS', 'outdoor', 'hazy', '*.png'))[:100]
            self.B_paths_dehaz = glob.glob(os.path.join('./datasets', 'Image dehazing', 'SOTS', 'outdoor', 'gt', '*.png'))[:100]
            for _ in range(len(self.A_paths_desnow)//len(self.A_paths_dehaz)):
                self.A_paths = self.A_paths + self.A_paths_dehaz
                self.B_paths = self.B_paths + self.B_paths_dehaz

            self.A_paths_deblur = glob.glob(os.path.join('./datasets', 'Image deblurring', 'GoPro', 'test', 'input', '*.png'))
            self.B_paths_deblur = glob.glob(os.path.join('./datasets', 'Image deblurring', 'GoPro', 'test', 'target', '*.png'))
            self.A_paths = self.A_paths + self.A_paths_deblur
            self.B_paths = self.B_paths + self.B_paths_deblur

            self.A_paths_light = glob.glob(os.path.join('./datasets', 'Low-light enhancement', 'LOL', 'eval', 'low', '*.png'))
            self.B_paths_light = glob.glob(os.path.join('./datasets', 'Low-light enhancement', 'LOL', 'eval', 'high', '*.png'))
            self.A_paths = self.A_paths + self.A_paths_light
            self.B_paths = self.B_paths + self.B_paths_light

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        print('A size: {:}, B size: {:}'.format(self.A_size, self.B_size))
        assert (self.A_size == self.B_size)

    def __getitem__(self, index):
        A_path = self.A_paths[index]  # make sure index is within then range
        B_path = self.B_paths[index]
        condition = Image.open(A_path).convert('RGB')  # condition
        gt = Image.open(B_path).convert('RGB')  # gt
        w, h = condition.size

        if self.mode == 'train':
            pixel_transform = transforms.Compose([
                transforms.RandomCrop(size=(256, 256)),
                transforms.ToTensor()
            ])
            
            condition = pixel_transform(condition)
            gt = pixel_transform(gt)
        else:
            pixel_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            
            condition = pixel_transform(condition)
            gt = pixel_transform(gt)
            if self.is_not_resolution_valid(w, h):
                condition = self.pad_to_valid_resolution(condition)
                gt = self.pad_to_valid_resolution(gt)

        field = os.path.basename(A_path)
        return {'adap': condition, 'gt': gt, 'paths': field}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def is_not_resolution_valid(self, height, width, window_size=8):
        # 检测图像分辨率保证能合法输入swin-block
        return height % window_size != 0 or width % window_size != 0

    def pad_to_valid_resolution(self, image, window_size=8):
        # 不合法的要进行最小值padding
        _, height, width = image.shape
        pad_height = (window_size - height % window_size) % window_size
        pad_width = (window_size - width % window_size) % window_size
        padding = (0, pad_width, 0, pad_height)
        padded_image = F.pad(image, padding, mode="constant", value=0)
        return padded_image

    def cv2equalizeHist(self, img):
        (b, g, r) = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        img = cv2.merge((b, g, r))
        return img

    def to_tensor(self, img):
        img = Image.fromarray(img)  # returns an image object.
        img_t = TF.to_tensor(img).float()
        return img_t