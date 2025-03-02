import copy
import glob
import math
import os
import random
from collections import namedtuple
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
import torchvision.transforms as transforms
# import Augmentor
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from ema_pytorch import EMA
from PIL import Image
import time
from torch import einsum, nn
from torch.optim import Adam, RAdam
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import utils
from tqdm.auto import tqdm
from thop import profile
import copy
import importlib
from src.autoencoder_model import Encoder, Decoder

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

from src.openaimodel import UNetModel_fea
from src.attention import TransformerBlock1, TransformerBlock2
from src.swin_att import SwinTransformerBlock
from loss import get_style_loss, get_preceptual_loss, VGG16FeatureExtractor
from src.networks import MultiscaleDiscriminator, GANLoss

ModelResPrediction = namedtuple(
    'ModelResPrediction', ['pred_res', 
                           'pred_noise', 
                           'pred_x_start', 
                           'pred_res_z', 
                           'pred_noise_z', 
                           'pred_z_start', 
                           'x_befor', 
                           'x_after', 
                           'z_befor', 
                           'z_after', 
                           'mask1', 
                           'mask2'])
# helpers functions
metric_module = importlib.import_module('metrics')

def set_seed(SEED):
    # initialize random seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))),
                normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. '
                            f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# normalization functions


def normalize_to_neg_one_to_one(img):
    if isinstance(img, list):
        return [img[k] * 2 - 1 for k in range(len(img))]
    else:
        return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    if isinstance(img, list):
        return [(img[k] + 1) * 0.5 for k in range(len(img))]
    else:
        return (img + 1) * 0.5

# small helper modules


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1',
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(
            half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class Unet_img(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            resnet_block_groups=8,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            condition=False,
    ):
        super().__init__()

        # determine dimensions
        self.output_size = 256
        self.channels = channels
        self.depth = len(dim_mults)
        input_channels = channels + channels * (1 if condition else 0)

        # input_channels = input_channels + 1  # mask channels
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]

        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # # time embeddings
        #
        time_dim = dim * 4

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                    dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]

        # self.mid_block1 = block_klass(mid_dim + mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                    dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def encode(self, x, t):
        # H, W = x.shape[2:]
        x = self.check_image_size(x, self.output_size, self.output_size)
        x = self.init_conv(x)
        r = x.clone()

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)
        return h, x, r

    def decode(self, x, t, h, r):
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        # x = x[..., :self.output_size, :self.output_size].contiguous()
        return x

    def forward(self, x, t):
        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(t)
        # t2 = self.time_mlp(time[1][0])
        # t = self.time_mlp_sum(torch.cat((t1, t2), dim = 1))

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        x = x[..., :H, :W].contiguous()
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Fusion2(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            channels = 3
    ):
        super().__init__()

        # determine dimensions
        self.output_size = 256
        # self.channels = channels
        # self.depth = len(dim_mults)

        input_channels = channels + channels + channels
        block_klass = partial(ResnetBlock, groups = 6)

        # input_channels = input_channels + 1  # mask channels
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)
        # self.begin_block = block_klass(init_dim, init_dim)

        dim_in = init_dim // 3
        
        # layers
        input_resolution = self.output_size
        self.downs1 = nn.Sequential(
            block_klass(dim_in, dim_in),
            Residual(PreNorm(dim_in,
                             SwinTransformerBlock(dim=dim_in, input_resolution=(input_resolution, input_resolution),
                                                  num_heads=6, window_size=8)))
        )

        self.downs2 = nn.Sequential(
            Downsample(dim_in, 2 * dim_in),
            block_klass(2 * dim_in, 2 * dim_in),
            Residual(PreNorm(2 * dim_in, LinearAttention(2 * dim_in))),
            Upsample(2 * dim_in, dim_in)
        )

        self.downs3 = nn.Sequential(
            Downsample(dim_in, 2 * dim_in),
            block_klass(2 * dim_in, 2 * dim_in),
            Downsample(2 * dim_in, 4 * dim_in),
            block_klass(4 * dim_in, 4 * dim_in),
            Residual(PreNorm(4 * dim_in, LinearAttention(4 * dim_in))),
            Upsample(4 * dim_in, 2 * dim_in),
            block_klass(2 * dim_in, 2 * dim_in),
            Upsample(2 * dim_in, dim_in)
        )

        # self.final_res_block = block_klass(3 * dim_in, dim)
        self.final_conv = nn.Conv2d(3 * dim_in, out_dim, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x0, x1, x2):
        H, W = x0.shape[2:]
        x = self.init_conv(torch.cat((x0, x1, x2), dim=1))
        r = x.clone()
        C = x.shape[1] // 3
        x = torch.split(x, C, dim=1)

        x1 = self.downs1(x[0])
        x2 = self.downs2(x[1])
        x3 = self.downs3(x[2])

        x = self.final_conv(torch.cat((x1, x2, x3), dim=1))
        return x

class Fusion(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            resnet_block_groups=8,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            condition=False,
    ):
        super().__init__()

        # determine dimensions
        self.output_size = 256
        self.channels = channels
        self.depth = len(dim_mults)

        input_channels = channels + channels + channels

        # input_channels = input_channels + 1  # mask channels
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]

        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        input_resolution = self.output_size
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in),
                block_klass(dim_in, dim_in),
                Residual(PreNorm(dim_in, SwinTransformerBlock(dim = dim_in, input_resolution = (input_resolution, input_resolution), num_heads=4, window_size=8))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                    dim_in, dim_out, 3, padding=1)
            ]))

            input_resolution = input_resolution // 2

        mid_dim = dims[-1]

        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, SwinTransformerBlock(dim = mid_dim, input_resolution = (input_resolution, input_resolution), num_heads=4, window_size=8)))
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out),
                block_klass(dim_out + dim_in, dim_out),
                Residual(PreNorm(dim_out, SwinTransformerBlock(dim = dim_out, input_resolution = (input_resolution, input_resolution), num_heads=4, window_size=8))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                    dim_out, dim_in, 3, padding=1)
            ]))

            input_resolution = input_resolution * 2

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(init_dim + dim_in, dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x0, x1, x2):
        H, W = x0.shape[2:]
        x = self.init_conv(torch.cat((x0, x1, x2), dim = 1))
        r = x.clone()

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x)
            h.append(x)

            x = block2(x)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x)
        out = self.final_conv(x)
        # x = x[..., :H, :W].contiguous()
        return out

class Unet_feature(nn.Module):
    def __init__(
            self,
            dim = 32,
            init_dim=32,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=16,
            resnet_block_groups=8,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            condition=False,
    ):
        super().__init__()

        # # determine dimensions
        self.output_size = 32

        self.channels = channels
        # channels = 4
        self.depth = len(dim_mults)
        input_channels = channels + channels * (1 if condition else 0)

        # input_channels = input_channels + 1  # mask channels
        # init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(8, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]

        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # # time embeddings
        time_dim = dim * 4
        
        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                    dim_in, dim_out, 3, padding=1)
            ]))

        # self.cross_conv = nn.Conv2d(img_channels, mid_dim, 7, padding=3)
        mid_dim = dims[-1]

        # self.mid_block1 = block_klass(mid_dim + mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        # Attention + LayerNorm + Cross-Attention + LayerNorm
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                    dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = block_klass(dim_in * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def encode(self, x, t):
        # H, W = x.shape[2:]
        x = self.check_image_size(x, self.output_size, self.output_size)
        x = self.init_conv(x)
        r = x.clone()

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        return h, x, r

    def decode(self, x, t, h, r):
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        x = x[..., :self.output_size, :self.output_size].contiguous()
        return x

    def forward(self, x, t):
        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(t)

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        x = x[..., :H, :W].contiguous()
        return x

class P2L(nn.Module):
    def __init__(
            self,
            latent_dim,
            pixel_dim,
            pixel_down_rate = 4,
            input_resolution = 64
    ):
        super().__init__()

        self.beta = nn.Parameter(torch.zeros((1, latent_dim, 1, 1)), requires_grad=True)

        self.pixel_down = nn.ModuleList([Downsample(pixel_dim) for _ in range(pixel_down_rate // 2)])
        self.pixel_conv = nn.Sequential(
            LayerNorm(pixel_dim),
            nn.Conv2d(pixel_dim, pixel_dim, 3, padding=1)
        )
        
        self.latent_conv = nn.Sequential(
            LayerNorm(latent_dim),
            nn.Conv2d(latent_dim, latent_dim, 5, padding=2)
        )

        self.get_mask = nn.Sequential(
            nn.Conv2d(latent_dim + pixel_dim, 1, 7, padding=3),
            nn.Sigmoid()
        )

        self.trans_conv = nn.Sequential(
            LinearAttention(latent_dim + pixel_dim),
            nn.Conv2d(latent_dim + pixel_dim, latent_dim, 3, padding=1)
        )

    def forward(self, pixel, latent):
        
        for down in self.pixel_down:
            pixel = down(pixel)

        latent_fea = self.latent_conv(latent)
        pixel_fea = self.pixel_conv(pixel)

        mask = self.get_mask(torch.cat((latent_fea, pixel_fea), dim=1))
        
        latent_fea = self.trans_conv(torch.cat((pixel_fea * mask, latent_fea), dim = 1))

        return latent + self.beta * latent_fea, mask

class L2P(nn.Module):
    def __init__(
            self,
            latent_dim,
            pixel_dim,
            latent_up_rate = 4,
            input_resolution = 64
    ):
        super().__init__()

        self.beta = nn.Parameter(torch.zeros((1, pixel_dim, 1, 1)), requires_grad=True)

        self.latent_up = nn.ModuleList([Upsample(latent_dim) for _ in range(latent_up_rate // 2)])
        self.latent_conv = nn.Sequential(
            LayerNorm(latent_dim),
            nn.Conv2d(latent_dim, latent_dim, 5, padding=2)
        )

        self.pixel_conv = nn.Sequential(
            LayerNorm(pixel_dim),
            nn.Conv2d(pixel_dim, pixel_dim, 3, padding=1)
        )

        self.get_mask = nn.Sequential(
            nn.Conv2d(latent_dim + pixel_dim, 1, 7, padding=3),
            nn.Sigmoid()
        )

        self.att_conv = nn.Sequential(
            LinearAttention(latent_dim + pixel_dim),
            nn.Conv2d(latent_dim + pixel_dim, pixel_dim, 3, padding=1)
        )

    def forward(self, pixel, latent):
        for up in self.latent_up:
            latent = up(latent)

        latent_fea = self.latent_conv(latent)
        pixel_fea = self.pixel_conv(pixel)

        mask = self.get_mask(torch.cat((latent_fea, pixel_fea), dim=1))
        
        pixel_fea = self.att_conv(torch.cat((pixel_fea, latent_fea * mask), dim = 1))

        return pixel + self.beta * pixel_fea, mask
    
class UnetResx2(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            resnet_block_groups=8,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            num_unet=1,
            condition=False,
            objective='pred_res_noise',
            test_res_or_noise="res_noise"
    ):
        super().__init__()

        self.condition = condition
        self.channels = channels
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        self.objective = objective
        self.test_res_or_noise = test_res_or_noise

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(64),
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, 256)
        )

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quant_conv = torch.nn.Conv2d(2 * 4, 2 * 4, 1)
        self.post_quant_conv = torch.nn.Conv2d(4, 4, 1)
        self.scale_factor = 1

        self.unet_img = Unet_img(dim,
                                 init_dim=init_dim,
                                 out_dim=out_dim,
                                 dim_mults=dim_mults,
                                 channels=channels,
                                 resnet_block_groups=resnet_block_groups,
                                 learned_variance=learned_variance,
                                 learned_sinusoidal_cond=learned_sinusoidal_cond,
                                 random_fourier_features=random_fourier_features,
                                 learned_sinusoidal_dim=learned_sinusoidal_dim,
                                 condition=condition)

        # self.unet_feature = UNetModel_fea()

        self.unet_feature = Unet_feature(dim = 64,
                                         init_dim = 16,
                                         out_dim = out_dim,
                                         dim_mults=(2, 4, 8),
                                         channels=4,
                                         resnet_block_groups=resnet_block_groups,
                                         learned_variance=learned_variance,
                                         learned_sinusoidal_cond=learned_sinusoidal_cond,
                                         random_fourier_features=random_fourier_features,
                                         learned_sinusoidal_dim=learned_sinusoidal_dim,
                                         condition=condition)

        self.up_img_atts = nn.ModuleList([
            L2P(latent_dim=512, pixel_dim=512, latent_up_rate = 4, input_resolution = (32, 32)),
            L2P(latent_dim=256, pixel_dim=256, latent_up_rate = 4, input_resolution = (64, 64)),
            L2P(latent_dim=128, pixel_dim=128, latent_up_rate = 4, input_resolution = (128, 128)),
        ])

        self.up_sem_atts = nn.ModuleList([
            P2L(latent_dim=512, pixel_down_rate = 4, pixel_dim=512),
            P2L(latent_dim=256, pixel_down_rate = 4, pixel_dim=256),
            P2L(latent_dim=128, pixel_down_rate = 4, pixel_dim=128)
        ])


    def forward(self, x, z, time):
        x_befor, x_after, z_befor, z_after = [], [], [], []
        mask1_list = []
        mask2_list = []

        time = time[0].cuda()

        t = self.time_mlp(time)
        h1, f1, r1 = self.unet_img.encode(x, t)
        h2, f2, r2 = self.unet_feature.encode(z, t)

        for (block_img, block_fea, block_img_att, block_sem_att) in zip(self.unet_img.ups, self.unet_feature.ups, self.up_img_atts, self.up_sem_atts):
            f1 = torch.cat((f1, h1.pop()), dim=1)
            f1 = block_img[0](f1, t)
            f1 = torch.cat((f1, h1.pop()), dim=1)
            f1 = block_img[1](f1, t)
            f1 = block_img[2](f1)

            f2 = torch.cat((f2, h2.pop()), dim=1)
            f2 = block_fea[0](f2, t)
            f2 = torch.cat((f2, h2.pop()), dim=1)
            f2 = block_fea[1](f2, t)
            f2 = block_fea[2](f2)

            # print(f1.shape, f2.shape)
            x_befor.append(f1.detach())
            z_befor.append(f2.detach())

            f1, mask1 = block_img_att(f1, f2)
            f2, mask2 = block_sem_att(f1, f2)

            mask1_list.append(mask1.detach())
            mask2_list.append(mask2.detach())
            x_after.append(f1.detach())
            z_after.append(f2.detach())

            f1 = block_img[3](f1)
            f2 = block_fea[3](f2)

        f1 = torch.cat((f1, r1), dim=1)
        f1 = self.unet_img.final_res_block(f1, t)
        x_after.append(f1.detach())
        f1 = self.unet_img.final_conv(f1)
        # f1 = f1[..., :self.unet_img.output_size, :self.unet_img.output_size].contiguous()

        f2 = torch.cat((f2, r2), dim=1)
        f2 = self.unet_feature.final_res_block(f2, t)
        z_after.append(f2.detach())
        f2 = self.unet_feature.final_conv(f2)
        # f2 = f2[..., :self.unet_feature.output_size, :self.unet_feature.output_size].contiguous()
        # return x_befor[-1], x_after[-1], z_befor[-1], z_after[-1]
        return f1, f2, x_befor[-1], x_after[-1], z_befor[-1], z_after[-1], mask1_list[-1], mask2_list[-1]

    def updata_scale_factor(self, z):
        self.scale_factor = 1. / torch.std(z, dim=(1, 2, 3), keepdim=True)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        # self.scale_factor = 1. / moments.flatten().std()
        mean, _ = torch.chunk(moments, 2, dim=1)
        # mean = mean * self.scale_factor
        return mean

    def decode(self, z):
        # z = 1. / self.scale_factor * z
        z = self.post_quant_conv(z)
        z = self.decoder(z)
        return z

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def gen_coefficients(timesteps, schedule="increased", sum_scale=1, ratio=1):
    if schedule == "increased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        alphas = y/y_sum
    elif schedule == "decreased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        y = torch.flip(y, dims=[0])
        alphas = y/y_sum
    elif schedule == "lamda":
        x = np.linspace(0.0001, 0.02, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        alphas = 1 - y
    elif schedule == "average":
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)
    elif schedule == "normal":
        sigma = 1.0
        mu = 0.0
        x = np.linspace(-3+mu, 3+mu, timesteps, dtype=np.float32)
        y = np.e**(-((x-mu)**2)/(2*(sigma**2)))/(np.sqrt(2*np.pi)*(sigma**2))
        y = torch.from_numpy(y)
        alphas = y/y.sum()
    else:
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)

    return alphas*sum_scale

# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)

class ResidualDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        delta_end = 1.5e-3,
        sampling_timesteps=None,
        loss_type='l1',
        objective='pred_res_noise',
        ddim_sampling_eta=0.,
        condition=False,
        sum_scale=None,
        test_res_or_noise="None",
    ):
        super().__init__()
        assert not (
            type(self) == ResidualDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels
        self.image_size = image_size
        self.objective = objective
        self.condition = condition
        self.test_res_or_noise = test_res_or_noise
        self.delta_end = delta_end

        if self.condition:
            self.sum_scale = sum_scale if sum_scale else 0.01
            # ddim_sampling_eta = 0.25
        else:
            self.sum_scale = sum_scale if sum_scale else 1.

        beta_schedule = "linear"
        beta_start = 0.0001
        beta_end = 0.02
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            betas = (torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=torch.float32) ** 2)
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            betas = betas_for_alpha_bar(timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")
            
        delta_start = 1e-6
        delta = torch.linspace(delta_start, self.delta_end, timesteps, dtype=torch.float32)
        delta_cumsum = delta.cumsum(dim=0).clip(0, 1)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumsum = 1-alphas_cumprod ** 0.5
        betas2_cumsum = 1-alphas_cumprod

        alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
        betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
        alphas = alphas_cumsum-alphas_cumsum_prev
        alphas[0] = 0
        betas2 = betas2_cumsum-betas2_cumsum_prev
        betas2[0] = 0
        # raise
        betas_cumsum = torch.sqrt(betas2_cumsum)

        posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters
        # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('alphas', alphas)
        register_buffer('alphas_cumsum', alphas_cumsum)
        register_buffer('delta', delta)
        register_buffer('delta_cumsum', delta_cumsum)
        register_buffer('one_minus_alphas_cumsum', 1-alphas_cumsum)
        register_buffer('betas2', betas2)
        register_buffer('betas', torch.sqrt(betas2))
        register_buffer('betas2_cumsum', betas2_cumsum)
        register_buffer('betas_cumsum', betas_cumsum)
        register_buffer('posterior_mean_coef1',
                        betas2_cumsum_prev/betas2_cumsum)
        register_buffer('posterior_mean_coef2', (betas2 *
                        alphas_cumsum_prev-betas2_cumsum_prev*alphas)/betas2_cumsum)
        register_buffer('posterior_mean_coef3', betas2/betas2_cumsum)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def init(self):
        timesteps = 5

        beta_schedule = "linear"
        beta_start = 0.0001
        beta_end = 0.02
        if beta_schedule == "linear":
            betas = torch.linspace(
                beta_start, beta_end, timesteps, dtype=torch.float32)

        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            betas = betas_for_alpha_bar(timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")
                
        delta_start = 1e-6
        delta = torch.linspace(delta_start, self.delta_end, timesteps, dtype=torch.float32)
        delta_cumsum = delta.cumsum(dim=0).clip(0, 1)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumsum = 1-alphas_cumprod ** 0.5
        betas2_cumsum = 1-alphas_cumprod

        alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
        betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
        alphas = alphas_cumsum-alphas_cumsum_prev
        alphas[0] = alphas[1]
        betas2 = betas2_cumsum-betas2_cumsum_prev
        betas2[0] = betas2[1]

        betas_cumsum = torch.sqrt(betas2_cumsum)
    
        posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)

        self.alphas = alphas
        self.alphas_cumsum = alphas_cumsum
        self.delta = delta
        self.delta_cumsum = delta_cumsum
        self.one_minus_alphas_cumsum = 1-alphas_cumsum
        self.betas2 = betas2
        self.betas = torch.sqrt(betas2)
        self.betas2_cumsum = betas2_cumsum
        self.betas_cumsum = betas_cumsum
        self.posterior_mean_coef1 = betas2_cumsum_prev/betas2_cumsum
        self.posterior_mean_coef2 = (
            betas2 * alphas_cumsum_prev-betas2_cumsum_prev*alphas)/betas2_cumsum
        self.posterior_mean_coef3 = betas2/betas2_cumsum
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(
            posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def predict_noise_from_res(self, x_t, t, x_input, pred_res):
        return (
            (x_t - (1-extract(self.delta_cumsum,t,x_t.shape)) * x_input - (extract(self.alphas_cumsum, t, x_t.shape)-1)
             * pred_res)/extract(self.betas_cumsum, t, x_t.shape)
        )

    def predict_start_from_xinput_noise(self, x_t, t, x_input, noise):
        return (
            (x_t-extract(self.alphas_cumsum, t, x_t.shape)*x_input -
             extract(self.betas_cumsum, t, x_t.shape) * noise)/extract(self.one_minus_alphas_cumsum, t, x_t.shape)
        )

    def predict_start_from_res_noise(self, x_t, t, x_res, noise):
        return (
            x_t-extract(self.alphas_cumsum, t, x_t.shape) * x_res -
            extract(self.betas_cumsum, t, x_t.shape) * noise
        )

    def q_posterior_from_res_noise(self, x_res, noise, x_t, t):
        return (x_t-extract(self.alphas, t, x_t.shape) * x_res -
                (extract(self.betas2, t, x_t.shape)/extract(self.betas_cumsum, t, x_t.shape)) * noise)

    def q_posterior(self, pred_res, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef2, t, x_t.shape) * pred_res +
            extract(self.posterior_mean_coef3, t, x_t.shape) * x_start
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_input, x, z_input, z, t, task=None, clip_denoised=True):
        if not self.condition:
            x_in = x
            z_in = z
        else:
            x_in = torch.cat((x, x_input), dim=1)
            z_in = torch.cat((z, z_input), dim=1)

        model_output = self.model(x_in, z_in,
                                  [self.alphas_cumsum[t]*self.num_timesteps,
                                      self.betas_cumsum[t]*self.num_timesteps])
        maybe_clip = partial(torch.clamp, min=-1.,
                             max=1.) if clip_denoised else identity

        if self.objective == 'pred_res_noise':
            if self.test_res_or_noise == "res_noise":
                pred_res = model_output[0]
                pred_noise = model_output[1]
                pred_res = maybe_clip(pred_res)
                x_start = self.predict_start_from_res_noise(
                    x, t, pred_res, pred_noise)
                x_start = maybe_clip(x_start)
            elif self.test_res_or_noise == "res":
                pred_res = model_output[0]
                pred_res = maybe_clip(pred_res)
                pred_noise = self.predict_noise_from_res(
                    x, t, x_input, pred_res)
                x_start = x_input - pred_res
                x_start = maybe_clip(x_start)
            elif self.test_res_or_noise == "noise":
                pred_noise = model_output[1]
                x_start = self.predict_start_from_xinput_noise(
                    x, t, x_input, pred_noise)
                x_start = maybe_clip(x_start)
                pred_res = x_input - x_start
                pred_res = maybe_clip(pred_res)
        elif self.objective == 'pred_x0_noise':
            pred_res = x_input-model_output[0]
            pred_noise = model_output[1]
            pred_res = maybe_clip(pred_res)
            x_start = maybe_clip(model_output[0])
        elif self.objective == "pred_noise":
            pred_noise = model_output[0]
            x_start = self.predict_start_from_xinput_noise(
                x, t, x_input, pred_noise)
            x_start = maybe_clip(x_start)
            pred_res = x_input - x_start
            pred_res = maybe_clip(pred_res)
        elif self.objective == "pred_res":
            pred_res = model_output[0]
            pred_res = maybe_clip(pred_res)
            pred_noise = self.predict_noise_from_res(x, t, x_input, pred_res)
            x_start = x_input - pred_res
            x_start = maybe_clip(x_start)

            pred_res_z = model_output[1]
            pred_noise_z = None
            z_start = None

            x_befor = model_output[2]
            x_after = model_output[3]
            z_befor = model_output[4]
            z_after = model_output[5]
            mask1, mask2 = model_output[6], model_output[7]
        
        return ModelResPrediction(pred_res, pred_noise, x_start, pred_res_z, pred_noise_z, z_start, x_befor, x_after, z_befor, z_after, mask1, mask2)

    def p_mean_variance(self, x_input, x, t):
        preds = self.model_predictions(x_input, x, t)
        pred_res = preds.pred_res
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            pred_res=pred_res, x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x_input, x, t: int):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full(
            (x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x_input, x=x, t=batched_times)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, x_input, shape, last=True):
        x_input = x_input[0]

        batch, device = shape[0], self.betas.device

        if self.condition:
            img = x_input+math.sqrt(self.sum_scale) * \
                torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None

        if not last:
            img_list = []

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img, x_start = self.p_sample(x_input, img, t)

            if not last:
                img_list.append(img)

        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)

    @torch.no_grad()
    def ddim_sample(self, x_input, z_input, z_gt, size_x, size_z, last=True, task=None):
        x_input = x_input[0]
        z_input = z_input

        batch, device, total_timesteps, sampling_timesteps, eta, objective = size_x[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)#[:num]

        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        noise_z = torch.randn(size_z, device=device)
        noise_x = torch.randn(size_x, device=device)
        img = (1 - self.delta_cumsum[-1]) * x_input + math.sqrt(self.sum_scale) * noise_x
        z = (1 - self.delta_cumsum[-1]) * z_input + math.sqrt(self.sum_scale) * noise_z
        
        x_start = None
        type = "use_pred_noise"
        last=False

        if not last:
            img_list_pixel = []
            img_list_latent = []
            latent_fea_list = []
            mask1_list = []
            mask2_list = []
            latent_fea_list
            z_out = 1. / self.model.scale_factor * z
            sem2img = self.model.decode(z_out)
            img_list_latent.append(sem2img)

            latent_fea_list.append(z_gt)
            latent_fea_list.append(z_input)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long)

            preds = self.model_predictions(x_input, img, z_input, z, time_cond, task)
            # break

            mask1 = preds.mask1
            mask2 = preds.mask2
            mask1_list.append(mask1)
            mask1_list.append(mask2)

            pred_res = preds.pred_res

            pred_res_z = preds.pred_res_z

            if time_next < 0:
                z = z_input - pred_res_z
                z_out = 1. / self.model.scale_factor * z
                sem2img = self.model.decode(z_out)
                img_list_latent.append(sem2img)

                img = x_input - pred_res
                img_list_pixel.append(img)

                latent_fea_list.append(z)
                break

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum-alpha_cumsum_next
            delta_cumsum = self.delta_cumsum[time]
            delta_cumsum_next = self.delta_cumsum[time_next]
            delta = delta_cumsum-delta_cumsum_next
            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum-betas2_cumsum_next
            sigma2 = eta * (betas2*betas2_cumsum_next/betas2_cumsum)

            if type == "use_pred_noise":
                z = z - alpha * pred_res_z + delta * z_input + sigma2.sqrt() * noise_z
                img = img - alpha * pred_res + delta * x_input + sigma2.sqrt() * noise_x

            # if not last:
            img_list_pixel.append(img)

            z_out = 1. / self.model.scale_factor * z
            sem2img = self.model.decode(z_out)
            img_list_latent.append(sem2img)

        return unnormalize_to_zero_to_one(img_list_pixel), unnormalize_to_zero_to_one(img_list_latent)

    @torch.no_grad()
    def sample(self, x_input=0, gt = 0, batch_size=16, last=True, task=None):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.ddim_sample
        # sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        # x_input = (x_input + gt) / 2
        if self.condition:
            x_input = 2 * x_input - 1
            gt = 2 * gt - 1
            z_input = self.model.encode(x_input).detach()
            z_gt = self.model.encode(gt).detach()
            # z_gt = self.model.encode(gt).detach()
            
            self.model.updata_scale_factor(z_input)
            z_input = z_input * self.model.scale_factor
            z_gt = z_gt * self.model.scale_factor

            x_input = x_input.unsqueeze(0)
            batch_size, channels, h, w = x_input[0].shape
            size_x = (batch_size, channels, h, w)
            batch_size, channels, h, w = z_input.shape
            size_z = (batch_size, channels, h, w)
        else:
            size = (batch_size, channels, image_size, image_size)
        return sample_fn(x_input, z_input, z_gt, size_x, size_z, last=last, task=task)

    def q_sample_z(self, x_start, x_res, condition, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            x_start+extract(self.alphas_cumsum, t, x_start.shape) * x_res +
            extract(self.betas_cumsum, t, x_start.shape) * noise -
            extract(self.delta_cumsum, t, x_start.shape) * condition
        )

    def q_sample_x(self, x_start, x_res, condition, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            x_start+extract(self.alphas_cumsum, t, x_start.shape) * x_res +
            extract(self.betas_cumsum, t, x_start.shape) * noise -
            extract(self.delta_cumsum, t, x_start.shape) * condition
        )

    @property
    def loss_fn(self, loss_type='l1'):
        if loss_type == 'l1':
            return F.l1_loss
        elif loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def p_losses(self, imgs, t, noise=None):
        if isinstance(imgs, list):  # Condition
            x_input = 2 * imgs[1] - 1
            x_start = 2 * imgs[0] - 1  #gt:imgs[0], cond:imgs[1]
            task = imgs[2][0]

            z_input = self.model.encode(x_input).detach()
            self.model.updata_scale_factor(z_input)
            z_start = self.model.encode(x_start).detach()
            z_input = z_input * self.model.scale_factor
            z_start = z_start * self.model.scale_factor

        x_noise = default(noise, lambda: torch.randn_like(x_start))
        z_noise = default(noise, lambda: torch.randn_like(z_start))

        x_res = x_input - x_start
        z_res = z_input - z_start
        
        b, c, h, w = x_start.shape

        # noise sample
        x = self.q_sample_x(x_start, x_res, x_input, t, noise = x_noise)
        z = self.q_sample_z(z_start, z_res, z_input, t, noise = z_noise)

        # predict and take gradient step
        if not self.condition:
            x_in = x
            z_in = z
        else:
            x_in = torch.cat((x, x_input), dim=1)
            z_in = torch.cat((z, z_input), dim=1)

        model_out = self.model(x_in, z_in,
                               [self.alphas_cumsum[t]*self.num_timesteps,
                                self.betas_cumsum[t]*self.num_timesteps])

        x_pred = model_out[0]
        z_pred = model_out[1]
        
        # x_loss = self.loss_fn(model_out[0], x_res, reduction='none')
        # z_loss = torch.abs(z_res) * self.loss_fn(z_pred, z_res, reduction='none') / extract(self.alphas_cumsum, t, z_res.shape)
        x_loss = (torch.abs(x_res) + 1) * self.loss_fn(x_pred, x_res, reduction='none')
        z_loss = (torch.abs(z_res) + 1) * self.loss_fn(z_pred, z_res, reduction='none')
        # z_decode_loss = self.loss_fn(z_pred_decode, x_res, reduction='none')

        x_loss = reduce(x_loss, 'b ... -> b (...)', 'mean').mean()
        z_loss = reduce(z_loss, 'b ... -> b (...)', 'mean').mean()
        # z_decode_loss = reduce(z_decode_loss, 'b ... -> b (...)', 'mean').mean()

        # print(z_loss, x_loss)
        loss = z_loss + 5 * x_loss
        # loss = z_loss
        return loss

    def forward(self, img, *args, **kwargs):
        if isinstance(img, list):
            b, c, h, w, device, img_size, = * \
                img[0].shape, img[0].device, self.image_size
        else:
            b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        max_num = int(self.num_timesteps * 0.01) + 1
        t = torch.randint(0, max_num, (b,)) * 100 - 1
        t = torch.clamp(t, min=0).cuda().long()
        # t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(img, t, *args, **kwargs)


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        opts,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder='./results/sample',
        amp=False,
        fp16=False,
        split_batches=True,
        convert_image_to=None,
        condition=False,
        sub_dir=False,
    ):
        super().__init__()

        self.device = 'cuda'

        self.accelerator_frn = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )

        self.accelerator_munet = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )

        self.sub_dir = sub_dir
        self.accelerator_frn.native_amp = amp
        self.accelerator_munet.native_amp = amp

        assert has_int_squareroot(
            num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.condition = condition

        self.FRN = Fusion(dim = 64,
                                 init_dim = 16,
                                 out_dim = 3,
                                 dim_mults = (1, 2, 4, 8),
                                 channels = 3,
                                 resnet_block_groups=8,
                                 learned_variance=False,
                                 learned_sinusoidal_cond=False,
                                 random_fourier_features=False,
                                 learned_sinusoidal_dim=16,
                                 condition=False).to(self.device)

        self.MUNet_diffusion = diffusion_model.to(self.device)

        if opts.phase == "train":
            self.train_data = cycle(self.accelerator_munet.prepare(
                DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True, num_workers=4)))
        else:
            self.test_data = cycle(self.accelerator_munet.prepare(
                DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)))
        self.sample_dataset = dataset

        # optimizer
        self.opt_frn = Adam(self.FRN.parameters(), lr=0.0001, betas=adam_betas)
        self.opt_munet = Adam(self.MUNet_diffusion.parameters(), lr=0.0001, betas=adam_betas)

        if self.accelerator_munet.is_main_process:
            self.ema_frn = EMA(self.FRN, beta=ema_decay,
                           update_every=ema_update_every).to(self.device)

            self.ema_munet = EMA(self.MUNet_diffusion, beta=ema_decay,
                           update_every=ema_update_every).to(self.device)

            self.set_results_folder(results_folder)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.FRN, self.opt_frn = self.accelerator_frn.prepare(self.FRN, self.opt_frn)
        self.MUNet_diffusion, self.opt_munet = self.accelerator_munet.prepare(self.MUNet_diffusion, self.opt_munet)

    def save(self, milestone):
        data = {
            'step': self.step,
            'MUNet_diffusion': self.accelerator_munet.get_state_dict(self.MUNet_diffusion),
            'FRN': self.accelerator_frn.get_state_dict(self.FRN),
            'opt_munet': self.opt_munet.state_dict(),
            'opt_frn': self.opt_frn.state_dict(),
            'ema_munet': self.ema_munet.state_dict(),
            'ema_frn': self.ema_frn.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, path):
        AE = torch.load('512-inpainting-ema.ckpt')['state_dict']

        state_dict_encoder = {}
        AE_encoder_state_dict = {key: value for key, value in AE.items() if
                              key.startswith('first_stage_model.encoder')}
        for key, value in AE_encoder_state_dict.items():
            new_key = key.replace('first_stage_model.encoder.', '', 1)
            state_dict_encoder[new_key] = value
        self.MUNet_diffusion.model.encoder.load_state_dict(state_dict_encoder)

        state_dict_decoder = {}
        AE_decoder_state_dict = {key: value for key, value in AE.items() if
                              key.startswith('first_stage_model.decoder')}
        for key, value in AE_decoder_state_dict.items():
            new_key = key.replace('first_stage_model.decoder.', '', 1)
            state_dict_decoder[new_key] = value
        self.MUNet_diffusion.model.decoder.load_state_dict(state_dict_decoder)

        state_dict_quant = {}
        AE_quant_state_dict = {key: value for key, value in AE.items() if
                              key.startswith('first_stage_model.quant_conv')}
        for key, value in AE_quant_state_dict.items():
            new_key = key.replace('first_stage_model.quant_conv.', '', 1)
            state_dict_quant[new_key] = value
        self.MUNet_diffusion.model.quant_conv.load_state_dict(state_dict_quant)

        state_dict_post = {}
        AE_post_state_dict = {key: value for key, value in AE.items() if
                              key.startswith('first_stage_model.post_quant_conv')}
        for key, value in AE_post_state_dict.items():
            new_key = key.replace('first_stage_model.post_quant_conv.', '', 1)
            state_dict_post[new_key] = value
        self.MUNet_diffusion.model.post_quant_conv.load_state_dict(state_dict_post)

        for param in self.MUNet_diffusion.model.encoder.parameters():
            param.requires_grad = False

        for param in self.MUNet_diffusion.model.decoder.parameters():
            param.requires_grad = False

        for param in self.MUNet_diffusion.model.quant_conv.parameters():
            param.requires_grad = False

        for param in self.MUNet_diffusion.model.post_quant_conv.parameters():
            param.requires_grad = False

        # path = Path('./ckpt_universal/diffuir/model-nolight.pt')
        # path = Path('./ckpt_universal/diffuir/model-light.pt')
        if path.exists():
            data = torch.load(str(path), map_location=self.device)
            self.MUNet_diffusion = self.accelerator_munet.unwrap_model(self.MUNet_diffusion)

            self.MUNet_diffusion.load_state_dict(data['MUNet_diffusion'], strict=False)
            self.FRN.load_state_dict(data['FRN'], strict=False)

            self.step = data['step']

            print("load model - "+str(path))

        self.ema_munet.to(self.device)
        self.ema_frn.to(self.device)

    def train_frn(self):
        self.step = 0
        loss_fn = F.l1_loss  # 计算L1 loss
        lossNet = VGG16FeatureExtractor().to(self.device)  # 计算感知loss

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                data = next(self.train_data)
                file_ = data["paths"]
                batches = self.num_samples
                x_input_sample = data["adap"].to(self.device)
                gt = data["gt"].to(self.device)

                with torch.no_grad():
                    pixel_list, latent_list = list(self.MUNet_diffusion.sample(
                        x_input_sample, gt, batch_size=batches, last=True, task=file_))
                
                out = self.FRN(x_input_sample, pixel_list[-1].detach(), latent_list[-1].detach())

                # L1 loss
                out_loss = (torch.abs(gt - x_input_sample) + 1) * loss_fn(out, gt, reduction='none')
                out_loss = reduce(out_loss, 'b ... -> b (...)', 'mean').mean()

                # 感知 loss
                real_feats = lossNet(gt)
                fake_feats = lossNet(out)
                preceptual_loss = get_preceptual_loss(real_feats, fake_feats)

                loss = out_loss + 0.1 * preceptual_loss
                self.accelerator_frn.backward(loss)
                self.opt_frn.step()
                self.opt_frn.zero_grad()

                self.step += 1

                if self.step % 200 == 0:
                    nrow = out.shape[0]
                    file_name = 'rec_{:}.png'.format(self.step)
                    full_path = os.path.join('./save', file_name)
                    utils.save_image(out, full_path, nrow=nrow)

                    file_name = 'noised_{:}.png'.format(self.step)
                    full_path = os.path.join('./save', file_name)
                    utils.save_image(x_input_sample, full_path, nrow=nrow)

                    file_name = 'gt_{:}.png'.format(self.step)
                    full_path = os.path.join('./save', file_name)
                    utils.save_image(gt, full_path, nrow=nrow)
                
                if self.step % 2000 == 0:
                    milestone = self.step // self.save_and_sample_every
                    self.save(milestone)

    def train_mu_net(self):
        self.step = 0
        loss_sum = 0
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                self.step += 1
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.train_data)
                    gt = data["gt"].to(self.device)
                    cond_input = data["adap"].to(self.device)
                    file_ = data["paths"]
                    data = [gt, cond_input, file_]

                    with self.accelerator_munet.autocast():
                        loss = self.MUNet_diffusion(data)
                        loss_sum = loss_sum + loss.detach()
                    self.accelerator_munet.backward(loss / self.gradient_accumulate_every)

                self.accelerator_munet.clip_grad_norm_(self.MUNet_diffusion.parameters(), 1.0)
                self.accelerator_munet.wait_for_everyone()

                self.opt_munet.step()
                self.opt_munet.zero_grad()

                self.step += 1

                if self.step % 5 == 0:
                    print(loss_sum / 5)
                    loss_sum = 0
                    self.ema_munet.update()

                if self.step % 2000 == 0:
                    milestone = self.step // self.save_and_sample_every
                    self.save(milestone)

                if self.step % 200 == 0:
                    with torch.no_grad():
                        data = next(self.train_data)
                        file_ = data["paths"][0]
                        batches = self.num_samples
                        x_input_sample = data["adap"].to(self.device)
                        gt = data["gt"].to(self.device)

                        pixel_list, latent_list = list(self.MUNet_diffusion.sample(
                            x_input_sample, gt, batch_size=batches, last=True, task=file_))

                    last_pixel_img = pixel_list[-1]
                    last_latent_img = latent_list[-1]
                    for idx in range(last_latent_img.shape[0]):
                        nrow = 1
                        i = 0
                        for image in last_pixel_img:
                            i = i + 1
                            file_name = 'rec_img_{:}_{:}_{:}.png'.format(self.step, idx, i)
                            full_path = os.path.join('./save', file_name)
                            utils.save_image(image[idx:idx+1], full_path, nrow=nrow)

                        for image in last_latent_img:
                            i = i + 1
                            file_name = 'rec_latent_{:}_{:}_{:}.png'.format(self.step, idx, i)
                            full_path = os.path.join('./save', file_name)
                            utils.save_image(image[idx:idx+1], full_path, nrow=nrow)

                        file_name = 'noised_{:}_{:}.png'.format(self.step, idx)
                        full_path = os.path.join('./save', file_name)
                        utils.save_image(x_input_sample[idx:idx+1], full_path, nrow=nrow)

                        file_name = 'real_{:}_{:}.png'.format(self.step, idx)
                        full_path = os.path.join('./save', file_name)
                        utils.save_image(gt[idx:idx+1], full_path, nrow=nrow)

    def test_mu_net(self):
        test_step = 0
        test_num_steps = 1000
        while test_step < test_num_steps:
            test_step = test_step + 1
            with torch.no_grad():
                data = next(self.test_data)
                file_ = data["paths"][0]
                batches = self.num_samples
                x_input_sample = data["adap"].to(self.device)
                gt = data["gt"].to(self.device)

                pixel_list, latent_list = list(self.MUNet_diffusion.sample(
                    x_input_sample, gt, batch_size=batches, last=True, task=file_))

            last_pixel_img = pixel_list[-1]
            last_latent_img = latent_list[-1]

            # test batch size = 1
            nrow = 1
            file_name = 'rec_pixel_img_{:}.png'.format(test_step)
            full_path = os.path.join('./save', file_name)
            utils.save_image(last_pixel_img, full_path, nrow=nrow)
            
            file_name = 'rec_latent_img_{:}.png'.format(test_step)
            full_path = os.path.join('./save', file_name)
            utils.save_image(last_latent_img, full_path, nrow=nrow)

            file_name = 'noised_{:}.png'.format(test_step)
            full_path = os.path.join('./save', file_name)
            utils.save_image(x_input_sample, full_path, nrow=nrow)

            file_name = 'real_{:}.png'.format(test_step)
            full_path = os.path.join('./save', file_name)
            utils.save_image(gt, full_path, nrow=nrow)

    def test_fusion(self):
        test_step = 0
        test_num_steps = 1000
        while test_step < test_num_steps:
            test_step = test_step + 1
            with torch.no_grad():
                data = next(self.test_data)
                file_ = data["paths"][0]
                batches = self.num_samples
                x_input_sample = data["adap"].to(self.device)
                gt = data["gt"].to(self.device)

                pixel_list, latent_list = list(self.MUNet_diffusion.sample(
                    x_input_sample, gt, batch_size=batches, last=True, task=file_))
                
                out = self.FRN(x_input_sample, pixel_list[-1].detach(), latent_list[-1].detach())

            # test batch size = 1
            nrow = 1
            file_name = 'rec_{:}.png'.format(test_step)
            full_path = os.path.join('./save', file_name)
            utils.save_image(out, full_path, nrow=nrow)

            file_name = 'noised_{:}.png'.format(test_step)
            full_path = os.path.join('./save', file_name)
            utils.save_image(x_input_sample, full_path, nrow=nrow)

            file_name = 'gt_{:}.png'.format(test_step)
            full_path = os.path.join('./save', file_name)
            utils.save_image(gt, full_path, nrow=nrow)

    def set_results_folder(self, path):
        self.results_folder = Path(path)
        if not self.results_folder.exists():
            os.makedirs(self.results_folder)