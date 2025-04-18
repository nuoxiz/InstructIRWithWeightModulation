import torch
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor

import random
import numpy as np

class NoiseDegradation(object):
    def __init__(self, args):
        super(NoiseDegradation, self).__init__()
        self.args = args
        self.toTensor = ToTensor()
        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])
    def _add_gaussian_noise(self, clean_patch, sigma):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)

        return noisy_patch, clean_patch

    def _add_noise_degradation_by_level(self, clean_patch, degrade_type):
        degraded_patch = None
        if degrade_type == 0:
            # noise level (sigma) =15
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=15)
        elif degrade_type == 7:
            # noise level (sigma) =25
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=25)
        elif degrade_type == 8:
            # noise level (sigma) =50
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=50)

         # If degraded_patch is still None (meaning degrade_type wasn't 0, 7, or 8)
        # handle the case (raise an error, return the original, etc.)
        if degraded_patch is None:
            degraded_patch = clean_patch # or raise ValueError(f"Invalid degrade_type: {degrade_type}")

        return degraded_patch, clean_patch

    def add_noise_degradation(self,clean_patch,degrade_type = None):
        if degrade_type == 0 or degrade_type == 7 or degrade_type == 8:
            degrade_type= degrade_type
        else:
            degrade_type = random.choices([0,7,8])

        degraded_patch, _ = self._add_noise_degradation_by_level(clean_patch, degrade_type)
        return degraded_patch