import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn  import functional as F
import torch

class OffsetGenerator(nn.Module):
    def __init__(self, in_prompt_dim, out_conv_shapes):
        """
        in_prompt_dim: The number of channels in the incoming prompt (e.g. prompt_dim).
        out_conv_shapes: Some descriptor of how many weights or channels
                         you need to offset (e.g. #channels_out * #channels_in * kernel_dim^2).
        """

        super().__init__()
        out_channel = 32

        # refine local patterns in the prompt
        self.conv = nn.Conv2d(in_prompt_dim, out_channel, kernel_size=3, padding=1, stride=1)

        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.output_conv_size = out_conv_shapes

        self.flatten_layer = nn.Linear(in_features=out_channel, out_features=self.output_conv_size)


    def forward(self, prompt):
        """
        prompt: output from PGM with shape (B, prompt_dim, H, W)
        returns: shape (B, out_conv_size)
        """

        prompt = self.conv(prompt)

        prompt = self.global_average_pool(prompt)

        prompt = prompt.view(prompt.size(0), -1)

        offset_vector = self.flatten_layer(prompt)

        offset_vector = torch.tanh(offset_vector)

        return offset_vector

