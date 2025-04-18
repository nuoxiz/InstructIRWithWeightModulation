import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init
from models.OffsetGenerator import OffsetGenerator
from models.nafnet import NAFBlock


# ------------------------------------------------------------------------
# Copyright (c) 2023 va1shn9v. All Rights Reserved.
# ------------------------------------------------------------------------
# Source: https://github.com/va1shn9v/PromptIR

'''
@inproceedings{potlapalli2023promptir,
  title={PromptIR: Prompting for All-in-One Image Restoration},
  author={Potlapalli, Vaishnav and Zamir, Syed Waqas and Khan, Salman and Khan, Fahad},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
'''
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock,self).__init__()
        self.prompt_dim= prompt_dim
        # prompt_dim=128: Defines the number of channels in the prompt.
        # prompt_len=5: The number of different prompts available.
        # prompt_size=96: The spatial resolution of each prompt (assumed to be square: 96×96).
        # lin_dim=192: The input dimension for the linear layer.
        # prompt_param's size = 1 * N * C * H * W where N = number of prompt components
        # prompt_param = learnable parameters
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        # A linear layer takes an input of size "lin_dim" and produces "prompt_len" outputs.
        # This layer generates weights that determine the importance of each prompt.
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        # A 3×3 convolution with the same number of input and output channels (prompt_dim).
        # Stride =1 and padding =1 ensure the spatial size remains unchanged.
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)

    # During training, the PromptGenBlock learns to encode prompt_len different degradations into the prompt_param tensor.
    def forward(self,x):
        # x = image feature representation
        B,C,H,W = x.shape
        # x is averaged over the last two dimensions (H, W). The output tensor is of shape (B, C) => global descriptor of the input
        emb = x.mean(dim=(-2,-1))
        # the embedding is passed through the linear layer to produce a tensor of shape (B, prompt_len)
        # softmax() ensures values in the linear layer output sum to 1 across the "prompt_len" dimension
        # meaning each prompt gets an importance weight for each sample in the batch
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        # prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) expands the dimensions of prompt_weights to shape (B, prompt_len, 1, 1, 1).
        # self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1):
        # Expands prompt_param to (B, prompt_len, prompt_dim, prompt_size, prompt_size), so each batch element has its own copy.
        # repeat(B, 1, 1, 1, 1, 1) replicates the prompts across the batch.
        # The two tensors are multiplied element-wise, meaning each prompt is weighted by the computed prompt_weights.
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        # The weighted sum across prompt_len produces a single prompt tensor for each batch element.
        # The new shape is (B, prompt_dim, prompt_size, prompt_size).
        prompt = torch.sum(prompt,dim=1)
        # resize the spatial dimentions of prompt (prompt_size, prompt_size) to (H, W) while keeping B and prompt_dim unchanged.
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        # The learned prompt undergoes a 3×3 convolution for further processing without changing the shape of "prompt"
        prompt = self.conv3x3(prompt)
        # final shape of prompt = (B, prompt_dim, H, W)
        return prompt
        # If multiple degradations exist in the input image feature, the returned prompt will encode a mixture of the degradations it has learned during training

class ICB(nn.Module):
    """
    Instruction Condition Block (ICB)
    Paper Section 3.3
    """

    def __init__(self, feature_dim, text_dim=768):
        super(ICB, self).__init__()
        self.fc    = nn.Linear(text_dim, feature_dim)
        self.block = NAFBlock(feature_dim)
        self.beta  = nn.Parameter(torch.zeros((1, feature_dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, feature_dim, 1, 1)), requires_grad=True)

    def forward(self, x, text_embedding):
        gating_factors = torch.sigmoid(self.fc(text_embedding))
        # mc
        gating_factors = gating_factors.unsqueeze(-1).unsqueeze(-1)


        f = x * self.gamma + self.beta  # 1) learned feature scaling/modulation
        f = f * gating_factors          # 2) (soft) feature routing based on text
        f = self.block(f)               # 3) block feature enhancement
        return f + x # skip connection


class InstructIR(nn.Module):
    """
    InstructIR model using NAFNet (ECCV 2022) as backbone.
    The model takes as input an RGB image and a text embedding (encoded instruction).
    Described in Paper Section 3.3
    """

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], txtdim=768, include_offset=False):
        super().__init__()

        self.intro  = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders    = nn.ModuleList()
        self.decoders    = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups         = nn.ModuleList()
        self.downs       = nn.ModuleList()
        self.enc_cond    = nn.ModuleList()
        self.dec_cond    = nn.ModuleList()

        self.include_offset = include_offset

        chan = width

        # if include_offset is True:
        self.prompt_block_level1 = PromptGenBlock(prompt_dim=chan*2,prompt_len=3,prompt_size = 128,lin_dim = chan*2)
        self.prompt_block_level2 = PromptGenBlock(prompt_dim=chan*4,prompt_len=3,prompt_size = 64,lin_dim = chan*4)
        self.prompt_block_level3 = PromptGenBlock(prompt_dim=chan*8,prompt_len=3,prompt_size = 32,lin_dim = chan*8)

        # prompt_dim_level3 = chan*2**3

        self.promptBlocks = nn.ModuleList()
        self.promptBlocks.append(self.prompt_block_level3)
        self.promptBlocks.append(self.prompt_block_level2)
        self.promptBlocks.append(self.prompt_block_level1)

        # self.promptBlocks = [self.prompt_block_level3, self.prompt_block_level2, self.prompt_block_level1]

        for num in enc_blk_nums:
            #  Each encoder applies multiple NAFBlocks
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            # Each encoding layer is modulated using a corresponding ICB
            # which incorporates the provided text embeddings.
            self.enc_cond.append(ICB(chan, txtdim))
            # Downsampling layers that reduce spatial resolution while increasing channel dimensions.
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2
        # Middle blocks: a series of NAFBlocks applied to the deepest,
        # most abstract representation of the image features.
        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        naf_block = self.middle_blks[0]
        cIn = naf_block.conv5.in_channels
        cOut = naf_block.conv5.out_channels
        kernel_size = naf_block.conv5.kernel_size
        vector_size = cIn * cOut * kernel_size[0] * kernel_size[1]

        self.middleblock_offsetGen=OffsetGenerator(in_prompt_dim=chan, out_conv_shapes=vector_size)


        self.prompt_block_middle_blks = PromptGenBlock(prompt_dim=chan,prompt_len=3,prompt_size = 32,lin_dim = chan)

        # decoding path
        for num in dec_blk_nums:
            # Upsampling layers that increase the spatial resolution and
            # decrease the channel dimensions
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            # sequantially processes upsampled features using multiple NAFBlocks
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            # Add text embedding as modulation
            self.dec_cond.append(ICB(chan, txtdim))

        self.padder_size = 2 ** len(self.encoders)
        self.offset_generators = nn.ModuleList()

        # if include_offset is True:
        for i, decoder in enumerate(self.decoders):
            if(i <3):
                naf_block = decoder[0]

                cIn = naf_block.conv5.in_channels
                cOut = naf_block.conv5.out_channels
                kernel_size = naf_block.conv5.kernel_size

                vector_size = cIn * cOut * kernel_size[0] * kernel_size[1]

                self.offset_generators.append(OffsetGenerator(in_prompt_dim=self.promptBlocks[i].prompt_dim, out_conv_shapes=vector_size))


    def forward(self, inp, txtembd):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)
        encs = []

        for encoder, enc_mod, down in zip(self.encoders, self.enc_cond, self.downs):
            x = encoder(x)
            x = enc_mod(x, txtembd)
            encs.append(x)
            x = down(x)


        if self.include_offset:
            middle_block_prompt = self.prompt_block_middle_blks(x)
            offset_vector = self.middleblock_offsetGen(middle_block_prompt)
            for naf_block in self.middle_blks:
                x = naf_block(x, offset_vector)
        else:
            x = self.middle_blks(x)

        index = 0
        for decoder, up, enc_skip, dec_mod in zip(self.decoders, self.ups, encs[::-1], self.dec_cond):

            if self.include_offset is True:
                x = up(x)
                x = x + enc_skip
                offset_vector = None
                if (index < 3):
                    degradation_aware_prompt = self.promptBlocks[index](x)
                    offset_vector = self.offset_generators[index](degradation_aware_prompt)
                    index += 1

                if offset_vector is not None:
                    for naf_block in decoder:
                        x = naf_block(x, offset_vector)
                else:
                    x = decoder(x)

                x = dec_mod(x, txtembd)
            else:
                x = up(x)
                x = x + enc_skip
                x = decoder(x)
                x = dec_mod(x, txtembd)

        # ending = conv layer to postprocess the final decoded feature into the desired image format.
        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


def create_model(input_channels = 3, width = 32, enc_blks = [2, 2, 4, 8], middle_blk_num = 12, dec_blks = [2, 2, 2, 2], txtdim=768, include_offset=False):

    net = InstructIR(img_channel=input_channels, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks, txtdim=txtdim, include_offset=include_offset)

    return net