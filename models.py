import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from training_utils import choose_timesteps, aug_ode, compute_bits_per_dim


class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=None, hidden_dim=256):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim + 1, hidden_dim),
            torch.nn.SELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SELU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        if t.dim()==0:
            t = t.expand(x.size(0),)
        x = torch.cat((x, t[:,None]),dim=-1) 
        return self.net(x)
    

class NODE(nn.Module):
    def __init__(self, ode_func: nn.Module):
        super().__init__()
        self.odefunc = ode_func

    def forward(self, x, traj=False, t=torch.linspace(1, 0, 2).type(torch.float32) ,odeint_method='dopri5'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        flow = odeint(self.odefunc, x.to(device), t.to(device), 
                              method=odeint_method,        
                              atol=1e-10,
                              rtol=1e-10,)
        if traj:
            return flow
        else:
            return flow[-1]
        
    def log_likelihood(self, x, n_iter=5, odeint_method="dopri5"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t = choose_timesteps(odeint_method).to(device)
        l0 = torch.zeros(x.size(0), 1).to(device)
        initial_values = (x.to(device), l0)

        aug_dynamics = aug_ode(self.odefunc, n_iter)
        with torch.no_grad():
            x_t, log_det_t = odeint(aug_dynamics, initial_values, t,
                                    method = odeint_method,
                                    atol=1e-4,
                                    rtol=1e-4,)
            
        if x.dim() == 2:
            normal = torch.distributions.MultivariateNormal(torch.zeros(2).to(device), torch.eye(2).to(device))
            logp_x = normal.log_prob(x_t[-1]) + log_det_t[-1]
            return -logp_x.mean()
        else:
            logp_x = compute_bits_per_dim(x_t[-1], log_det_t[-1])
            return logp_x

        

#!/usr/bin/env python

# The file is copied and modified from 
# 
# C.-W. Huang, J. H. Lim, and A. Courville. 
# A variational perspective on diffusion-based generative models and score matching. 
# Advances in Neural Information Processing Systems, 2021
# 
# Code from https://github.com/CW-Huang/sdeflow-light

"""
Copied and modified from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
Copied and modified from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# noinspection PyProtectedMember
from torch.nn.init import _calculate_fan_in_and_fan_out


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        return torch.sigmoid(x) * x


def group_norm(out_ch):
    return nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-6, affine=True)


def upsample(in_ch, with_conv):
    up = nn.Sequential()
    up.add_module('up_nn', nn.Upsample(scale_factor=2, mode='nearest'))
    if with_conv:
        up.add_module('up_conv', conv2d(in_ch, in_ch, kernel_size=(3, 3), stride=1))
    return up


def downsample(in_ch, with_conv):
    if with_conv:
        down = conv2d(in_ch, in_ch, kernel_size=(3, 3), stride=2)
    else:
        down = nn.AvgPool2d(2, 2)
    return down


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None, conv_shortcut=False, dropout=0., normalize=group_norm, act=Swish()):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch if out_ch is not None else in_ch
        self.conv_shortcut = conv_shortcut
        self.dropout = dropout
        self.act = act

        self.norm1 = normalize(in_ch) if normalize is not None else nn.Identity()
        self.conv1 = conv2d(in_ch, out_ch)
        self.norm2 = normalize(out_ch) if normalize is not None else nn.Identity()
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0. else nn.Identity()
        self.conv2 = conv2d(out_ch, out_ch, init_scale=0.)
        if in_ch != out_ch:
            if conv_shortcut:
                self.shortcut = conv2d(in_ch, out_ch)
            else:
                self.shortcut = conv2d(in_ch, out_ch, kernel_size=(1, 1), padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # forward conv1
        h = x
        h = self.act(self.norm1(h))
        h = self.conv1(h)

        # forward conv2
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        # shortcut
        x = self.shortcut(x)

        # combine and return
        assert x.shape == h.shape
        return x + h


class SelfAttention(nn.Module):
    """
    copied modified from https://github.com/voletiv/self-attention-GAN-pytorch/blob/master/sagan_models.py#L29
    copied modified from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py#L66
    """

    def __init__(self, in_channels, normalize=group_norm):
        super().__init__()
        self.in_channels = in_channels
        self.attn_q = conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.attn_k = conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.attn_v = conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, init_scale=0.)
        self.softmax = nn.Softmax(dim=-1)
        if normalize is not None:
            self.norm = normalize(in_channels)
        else:
            self.norm = nn.Identity()

    # noinspection PyUnusedLocal
    def forward(self, x):
        """ t is not used """
        _, C, H, W = x.size()

        h = self.norm(x)
        q = self.attn_q(h).view(-1, C, H * W)
        k = self.attn_k(h).view(-1, C, H * W)
        v = self.attn_v(h).view(-1, C, H * W)

        attn = torch.bmm(q.permute(0, 2, 1), k) * (int(C) ** (-0.5))
        attn = self.softmax(attn)

        h = torch.bmm(v, attn.permute(0, 2, 1))
        h = h.view(-1, C, H, W)
        h = self.proj_out(h)

        assert h.shape == x.shape
        return x + h


def _calculate_correct_fan(tensor, mode):
    """
    copied and modified from https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py#L337
    """
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out', 'fan_avg']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_uniform_(tensor, gain=1., mode='fan_in'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: multiplier to the dispersion
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in')
    """
    fan = _calculate_correct_fan(tensor, mode)
    # gain = calculate_gain(nonlinearity, a)
    var = gain / max(1., fan)
    bound = math.sqrt(3.0 * var)  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def variance_scaling_init_(tensor, scale):
    return kaiming_uniform_(tensor, gain=1e-10 if scale == 0 else scale, mode='fan_avg')


def dense(in_channels, out_channels, init_scale=1.):
    lin = nn.Linear(in_channels, out_channels)
    variance_scaling_init_(lin.weight, scale=init_scale)
    nn.init.zeros_(lin.bias)
    return lin


def conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=1, dilation=1, padding=1, bias=True, padding_mode='zeros',
           init_scale=1.):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                     bias=bias, padding_mode=padding_mode)
    variance_scaling_init_(conv.weight, scale=init_scale)
    if bias:
        nn.init.zeros_(conv.bias)
    return conv


def get_sinusoidal_positional_embedding(timesteps: torch.LongTensor, embedding_dim: int):
    """
    Copied and modified from
        https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/nn.py#L90

    From Fairseq in
        https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py#L15
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.size()) == 1
    timesteps = timesteps.to(torch.get_default_dtype())
    device = timesteps.device

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # bsz x embd
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), "constant", 0)
    assert list(emb.size()) == [timesteps.size(0), embedding_dim]
    return emb


class UNet(nn.Module):
    def __init__(self,
                 input_channels=1,
                 input_height=28,
                 ch=32,
                 output_channels=None,
                 ch_mult=(1, 2, 4),
                 num_res_blocks=2,
                 attn_resolutions=(16,),
                 dropout=0.,
                 resamp_with_conv=True,
                 act=Swish(),
                 normalize=group_norm,
                 ):
        super().__init__()
        self.input_channels = input_channels
        self.input_height = input_height
        self.ch = ch
        self.output_channels = output_channels = input_channels if output_channels is None else output_channels
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.resamp_with_conv = resamp_with_conv
        self.act = act
        self.normalize = normalize

        # init
        self.num_resolutions = num_resolutions = len(ch_mult)
        in_ht = input_height
        in_ch = input_channels
        assert in_ht % 2 ** (num_resolutions - 1) == 0, "input_height doesn't satisfy the condition"


        # Downsampling
        self.begin_conv = conv2d(in_ch, ch)
        unet_chs = [ch]
        in_ht = in_ht
        in_ch = ch
        down_modules = []
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            block_modules = {}
            out_ch = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block_modules['{}a_{}a_block'.format(i_level, i_block)] = \
                    ResidualBlock(
                        in_ch=in_ch,
                        out_ch=out_ch,
                        dropout=dropout,
                        act=act,
                        normalize=normalize,
                    )
                if in_ht in attn_resolutions:
                    block_modules['{}a_{}b_attn'.format(i_level, i_block)] = SelfAttention(out_ch, normalize=normalize)
                unet_chs += [out_ch]
                in_ch = out_ch
            # Downsample
            if i_level != num_resolutions - 1:
                block_modules['{}b_downsample'.format(i_level)] = downsample(out_ch, with_conv=resamp_with_conv)
                in_ht //= 2
                unet_chs += [out_ch]
            # convert list of modules to a module list, and append to a list
            down_modules += [nn.ModuleDict(block_modules)]
        # conver to a module list
        self.down_modules = nn.ModuleList(down_modules)

        # Middle
        mid_modules = []
        mid_modules += [
            ResidualBlock(in_ch, out_ch=in_ch, dropout=dropout, act=act, normalize=normalize)]
        mid_modules += [SelfAttention(in_ch, normalize=normalize)]
        mid_modules += [
            ResidualBlock(in_ch, out_ch=in_ch, dropout=dropout, act=act, normalize=normalize)]
        self.mid_modules = nn.ModuleList(mid_modules)

        # Upsampling
        up_modules = []
        for i_level in reversed(range(num_resolutions)):
            # Residual blocks for this resolution
            block_modules = {}
            out_ch = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                block_modules['{}a_{}a_block'.format(i_level, i_block)] = \
                    ResidualBlock(
                        in_ch=in_ch + unet_chs.pop(),
                        out_ch=out_ch,
                        dropout=dropout,
                        act=act,
                        normalize=normalize)
                if in_ht in attn_resolutions:
                    block_modules['{}a_{}b_attn'.format(i_level, i_block)] = SelfAttention(out_ch, normalize=normalize)
                in_ch = out_ch
            # Upsample
            if i_level != 0:
                block_modules['{}b_upsample'.format(i_level)] = upsample(out_ch, with_conv=resamp_with_conv)
                in_ht *= 2
            # convert list of modules to a module list, and append to a list
            up_modules += [nn.ModuleDict(block_modules)]
        # conver to a module list
        self.up_modules = nn.ModuleList(up_modules)
        assert not unet_chs

        # End
        self.end_conv = nn.Sequential(
            normalize(in_ch),
            self.act,
            conv2d(in_ch, output_channels, init_scale=0.),
        )

    # noinspection PyMethodMayBeStatic
    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    # noinspection PyArgumentList,PyShadowingNames
    def forward(self, t, x):
        del t
        # Init
        B, C, H, W = x.size()

        # Downsampling
        hs = [self.begin_conv(x)]
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            block_modules = self.down_modules[i_level]
            for i_block in range(self.num_res_blocks):
                resnet_block = block_modules['{}a_{}a_block'.format(i_level, i_block)]
                h = resnet_block(hs[-1])
                if h.size(2) in self.attn_resolutions:
                    attn_block = block_modules['{}a_{}b_attn'.format(i_level, i_block)]
                    h = attn_block(h)
                hs.append(h)
            # Downsample
            if i_level != self.num_resolutions - 1:
                downsample = block_modules['{}b_downsample'.format(i_level)]
                hs.append(downsample(hs[-1]))

        # Middle
        h = hs[-1]
        h = self._compute_cond_module(self.mid_modules, h)

        # Upsampling
        for i_idx, i_level in enumerate(reversed(range(self.num_resolutions))):
            # Residual blocks for this resolution
            block_modules = self.up_modules[i_idx]
            for i_block in range(self.num_res_blocks + 1):
                resnet_block = block_modules['{}a_{}a_block'.format(i_level, i_block)]
                h = resnet_block(torch.cat([h, hs.pop()], axis=1))
                if h.size(2) in self.attn_resolutions:
                    attn_block = block_modules['{}a_{}b_attn'.format(i_level, i_block)]
                    h = attn_block(h)
            # Upsample
            if i_level != 0:
                upsample = block_modules['{}b_upsample'.format(i_level)]
                h = upsample(h)
        assert not hs

        # End
        h = self.end_conv(h)
        assert list(h.size()) == [x.size(0), self.output_channels, x.size(2), x.size(3)]
        return h
   

'''from https://github.com/cfinlay/ffjord-rnode'''
import lib.odenvp as odenvp
import lib.layers as layers
import six
import math
import lib.layers.wrappers.cnf_regularization as reg_lib


REGULARIZATION_FNS = {
    "kinetic_energy": reg_lib.quadratic_cost,
    "jacobian_norm2": reg_lib.jacobian_frobenius_regularization_fn,
    }

def create_regularization_fns():
    regularization_fns = []
    regularization_coeffs = []

    for arg_key, reg_fn in six.iteritems(REGULARIZATION_FNS):
        regularization_fns.append(reg_fn)
        regularization_coeffs.append(0.01)

    regularization_fns = tuple(regularization_fns)
    regularization_coeffs = tuple(regularization_coeffs)
    return regularization_fns, regularization_coeffs


def get_regularization(model, regularization_coeffs):
    if len(regularization_coeffs) == 0:
        return None

    acc_reg_states = tuple([0.] * len(regularization_coeffs))

    for module in model.modules():
        if isinstance(module, layers.CNF):
            reg = module.get_regularization_states()
            acc_reg_states = tuple(acc_reg_states[i] + reg[i] for i in range(len(reg)))

    return acc_reg_states

def set_step_size(step_size, model):

    def _set(module):
        if isinstance(module, layers.CNF):
            # Set training settings
            module.solver_options['step_size'] = step_size

    model.apply(_set)

def create_model(regularization_fns, solver='rk4'):
    hidden_dims = tuple(map(int, "64,64,64".split(",")))
    strides = tuple(map(int, "1,1,1,1".split(",")))

    model = odenvp.ODENVP(
        (200, *(1, 28, 28)),
        n_blocks=2,
        intermediate_dims=hidden_dims,
        div_samples=1,
        strides=strides,
        squeeze_first=False,
        nonlinearity="softplus",
        layer_type="concat",
        zero_last=True,
        alpha=1e-6,
        cnf_kwargs={"T": 1.0, "train_T": False, "regularization_fns": regularization_fns, "solver": solver},
    )

    set_step_size(0.25, model)

    return model
