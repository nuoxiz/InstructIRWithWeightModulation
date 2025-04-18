
from torch.utils.data import Dataset

import torchvision.transforms.functional as TF
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
from PIL import Image
import numpy as np
import os
import random

from utils import load_img, modcrop
from torchvision import transforms
from training_utils.NoiseDegradation import NoiseDegradation



DEG_MAP = {
    "noise_15" : 0,
    "blur"     : 1,
    "rain"     : 2,
    "haze"     : 3,
    "lol"      : 4,
    "sr"       : 5,
    "en"       : 6,
    "noise_25" : 7,
    "noise_50" : 8
}

DEG2TASK = {
    "noise": "denoising",
    "blur" : "deblurring",
    "rain" : "deraining",
    "haze" : "dehazing",
    "lol"  : "lol",
    "sr"   : "sr",
    "en"   : "enhancement"
}

def crop_img(image, base=16):
    """
    Mod crop the image to ensure the dimension is divisible by base. Also done by SwinIR, Restormer and others.
    """
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]

def data_augmentation(image, mode):
    if mode == 0:
        # original
        out = image.numpy()
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')
    return out

def random_augmentation(*args):
    out = []
    flag_aug = random.randint(1, 7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out




################# DATASETS


class InstructIRTrainDataset(Dataset):
    """
    Dataset for Image Restoration having low-quality image and the reference image.
    Tasks: synthetic denoising, deblurring, super-res, etc.
    """

    def __init__(self, args):

        super(InstructIRTrainDataset, self).__init__()

        self.toTensor  = ToTensor()
        self.noise_gradation_generator = NoiseDegradation(args)
        self.args = args
        self.de_type = args.de_type
        print(self.de_type)

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size)
        ])

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_original_rain_name(self, rainy_name):
        og_name = rainy_name.split("rainy")[0] + 'original/norain-' + rainy_name.split('rain-')[-1]
        return og_name

    def _init_clean_image_for_noise_degradation(self):
        ref_file = self.args.data_file_dir + "clean_image_for_denoise.txt"
        temp_ids = []
        temp_ids+= [id_.strip() for id_ in open(ref_file)]
        clean_ids = []
        name_list = os.listdir(self.args.denoise_dir)
        clean_ids += [self.args.denoise_dir + id_ for id_ in name_list if id_.strip() in temp_ids]

        self.s15_ids = []
        self.s25_ids = []
        self.s50_ids = []

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x,"de_type":0} for x in clean_ids]
            self.s15_ids = self.s15_ids
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x,"de_type":7} for x in clean_ids]
            self.s25_ids = self.s25_ids
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x,"de_type":8} for x in clean_ids]
            self.s50_ids = self.s50_ids
            random.shuffle(self.s50_ids)
            self.s50_counter = 0


        print(f"Noisy Sigma 15 images len: {len(self.s15_ids)}")
        print(f"Noisy Sigma 25 images len: {len(self.s25_ids)}\n")
        print(f"Noisy Sigma 50 images len: {len(self.s50_ids)}\n")


    def _init_rs_ids(self):
        temp_ids = []
        rs = self.args.data_file_dir + "/rainy.txt"
        temp_ids+= [self.args.derain_dir + id_.strip() for id_ in open(rs)]
        self.rs_ids = [{"clean_id":x, "de_type":2} for x in temp_ids]
        self.rs_ids = self.rs_ids * 40

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Rainy Images : {}".format(self.num_rl))

    def _init_hazy_ids(self):
        temp_ids = []
        hazy = self.args.data_file_dir + "hazy_outside.txt"
        temp_ids+= [self.args.dehaze_dir + id_.strip() for id_ in open(hazy)]
        self.hazy_ids = [{"clean_id" : x,"de_type":3} for x in temp_ids] * 2

        self.hazy_counter = 0

        self.num_hazy = len(self.hazy_ids)
        print("Total Hazy Images : {}".format(self.num_hazy))

    def _get_nonhazy_name(self, hazy_name):
        dir_name = hazy_name.split("synthetic")[0] + 'original/'
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = '.' + hazy_name.split('.')[-1]
        nonhazy_name = dir_name + name + suffix
        return nonhazy_name


    def __len__(self):
        return len(self.dataset_ids)

    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_image_for_noise_degradation()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()

        random.shuffle(self.de_type)

    def _merge_ids(self):
        self.dataset_ids = []
        if "denoise_15" in self.de_type:
            self.dataset_ids += self.s15_ids
            self.dataset_ids += self.s25_ids
            self.dataset_ids += self.s50_ids
        if "derain" in self.de_type:
            self.dataset_ids+= self.rs_ids
        if "dehaze" in self.de_type:
            self.dataset_ids += self.hazy_ids

        print(f"Dataset_ids length: {len(self.dataset_ids)}")

    def __getitem__(self, idx):
        dataset = self.dataset_ids[idx]
        hq_path = dataset["clean_id"]
        deg_id = dataset["de_type"]

        if deg_id == 0 or deg_id == 7 or deg_id == 8:
            # noisy image removal
            if deg_id == 0:
                hq_path = dataset["clean_id"]
            elif deg_id == 7:
                hq_path = dataset["clean_id"]
            elif deg_id == 8:
                hq_path = dataset["clean_id"]

            clean_img = crop_img(np.array(Image.open(hq_path).convert('RGB')), base=16)
            clean_patch = self.crop_transform(clean_img)
            clean_patch= np.array(clean_patch)

            clean_name = hq_path.split("/")[-1].split('.')[0]

            clean_patch = random_augmentation(clean_patch)[0]

            degrad_patch = self.noise_gradation_generator.add_noise_degradation(clean_patch, deg_id)
        else:
            if deg_id == 2:
                # Rain Streak Removal
                degrad_img = crop_img(np.array(Image.open(dataset["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_original_rain_name(dataset["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif deg_id == 3:
                # Dehazing with SOTS outdoor training set
                degrad_img = crop_img(np.array(Image.open(dataset["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(dataset["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)

        return [clean_name, deg_id], degrad_patch, clean_patch