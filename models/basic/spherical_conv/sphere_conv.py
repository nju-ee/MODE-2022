import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair, _single

import numpy as np

from . import sphere_conv_cuda
#import sphere_conv_cuda


class SphereConvFunction(Function):
  @staticmethod
  def forward(ctx, input, position, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if input is not None and input.dim() != 4:
      raise ValueError('Expected 4D tensor as input, got {}D tensor instead.'.format(input.dim()))
    ctx.stride = _pair(stride)
    ctx.padding = _pair(padding)
    ctx.dilation = _pair(dilation)
    ctx.groups = groups
    ctx.has_bias = bias is not None

    #output = input.new_empty(SphereConvFunction._output_size(input, weight, ctx.padding, ctx.dilation, ctx.stride))

    #ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]  # columns, ones

    if not ctx.has_bias:
      bias = input.new_empty(1)  # fake tensor
    if not input.is_cuda:
      raise NotImplementedError("Only support cuda tensor!")
    output = input.new_empty(SphereConvFunction._infer_shape(ctx, input, weight))
    ctx.save_for_backward(input, position, weight, bias)
    ctx._bufs = [input.new_empty(0), input.new_empty(0)]
    sphere_conv_cuda.sphere_conv_forward_cuda(input,
                                              weight,
                                              bias,
                                              ctx._bufs[0],
                                              position,
                                              output,
                                              ctx._bufs[1],
                                              weight.size(2),
                                              weight.size(3),
                                              ctx.stride[0],
                                              ctx.stride[1],
                                              ctx.padding[0],
                                              ctx.padding[1],
                                              ctx.dilation[0],
                                              ctx.dilation[1],
                                              ctx.groups,
                                              ctx.has_bias)
    return output

  @staticmethod
  @once_differentiable
  def backward(ctx, grad_output):
    input, position, weight, bias = ctx.saved_tensors

    grad_input = torch.zeros_like(input)
    grad_weight = torch.zeros_like(weight)
    grad_bias = torch.zeros_like(bias)

    if not grad_output.is_cuda:
      raise NotImplementedError
    sphere_conv_cuda.sphere_conv_backward_cuda(input,
                                               weight,
                                               bias,
                                               ctx._bufs[0],
                                               position,
                                               ctx._bufs[1],
                                               grad_input,
                                               grad_weight,
                                               grad_bias,
                                               grad_output,
                                               weight.size(2),
                                               weight.size(3),
                                               ctx.stride[0],
                                               ctx.stride[1],
                                               ctx.padding[0],
                                               ctx.padding[1],
                                               ctx.dilation[0],
                                               ctx.dilation[1],
                                               ctx.groups,
                                               ctx.has_bias)
    if not ctx.has_bias:
      grad_bias = None
    return (grad_input, None, grad_weight, grad_bias, None, None, None, None)  # same num as inputs of forward, only weight need grad

  @staticmethod
  def _output_size(input, weight, padding, dilation, stride):
    channels = weight.size(0)
    output_size = (input.size(0), channels)
    for d in range(input.dim() - 2):  # H, W
      in_size = input.size(d + 2)
      pad = padding[d]
      kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
      stride_ = stride[d]
      output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1,)
    if not all(map(lambda s: s > 0, output_size)):
      raise ValueError('convolution input is too small (output would be {})'.format('x'.join(map(str, output_size))))
    return output_size

  @staticmethod
  def _infer_shape(ctx, input, weight):
    n = input.size(0)
    channels_out = weight.size(0)
    height, width = input.shape[2:4]
    kernel_h, kernel_w = weight.shape[2:4]
    height_out = (height + 2 * ctx.padding[0] - (ctx.dilation[0] * (kernel_h - 1) + 1)) // ctx.stride[0] + 1
    width_out = (width + 2 * ctx.padding[1] - (ctx.dilation[1] * (kernel_w - 1) + 1)) // ctx.stride[1] + 1
    return n, channels_out, height_out, width_out


sphere_conv = SphereConvFunction.apply


class SphereConv(nn.Module):
  def __init__(self, in_height, in_width, sphereType, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
    super(SphereConv, self).__init__()
    assert (sphereType is not None) and (sphereType in ['Cassini', 'ERP'])
    assert (in_height is not None) and (in_height > 0)
    assert (in_width is not None) and (in_width > 0)
    #assert not bias
    assert in_channels % groups == 0, 'in_channels {} cannot be divisible by groups {}'.format(in_channels, groups)
    assert out_channels % groups == 0, 'out_channels {} cannot be divisible by groups {}'.format(out_channels, groups)

    #
    in_h = min(in_height, in_width)
    in_w = max(in_height, in_width)
    assert in_w == 2 * in_h

    self.in_height = in_h
    self.in_width = in_w
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = _pair(kernel_size)
    self.stride = _pair(stride)
    self.padding = _pair(padding)
    self.dilation = _pair(dilation)
    self.groups = groups
    self.sphereType = sphereType
    # enable compatibility with nn.Conv2d
    self.transposed = False
    self.output_padding = _single(0)
    self.input_size = (1, self.in_channels, self.in_height, self.in_width)
    self.output_size = self.cal_output_size()
    self.position = self.gen_sphere_position()
    self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
    if bias:
      self.bias = nn.parameter(torch.Tensor(out_channels))
    else:
      self.register_parameter('bias', None)
    self.position = self.position.cuda()
    self.position.requires_grad = False
    self.reset_parameters()

  def reset_parameters(self):
    n = self.in_channels
    for k in self.kernel_size:
      n *= k
    stdv = 1. / math.sqrt(n)
    self.weight.data.uniform_(-stdv, stdv)

  def cal_output_size(self):
    output_size = (1, self.out_channels)
    in_hw = (self.in_height, self.in_width)
    for d in range(2):  # H, W
      in_size = in_hw[d]
      pad = self.padding[d]
      kernel = self.dilation[d] * (self.kernel_size[d] - 1) + 1
      stride_ = self.stride[d]
      output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1,)
    if not all(map(lambda s: s > 0, output_size)):
      raise ValueError('convolution input is too small (output would be {})'.format('x'.join(map(str, output_size))))
    return output_size

  def gen_sphere_position(self):
    height, width = self.input_size[2:]
    Kh, Kw = self.kernel_size
    # stride_h, stride_w = self.stride
    stride_h, stride_w = 1, 1
    delta_lat = np.pi / height
    delta_lon = 2 * np.pi / width
    range_x = np.arange(-(Kw // 2), Kw // 2 + 1)
    if not Kw % 2:
      range_x = np.delete(range_x, Kw // 2)
    range_y = np.arange(-(Kh // 2), Kh // 2 + 1)
    if not Kh % 2:
      range_y = np.delete(range_y, Kh // 2)
    kerX = np.tan(range_x * delta_lon)
    kerY = np.tan(range_y * delta_lat) / np.cos(range_y * delta_lon)
    kerX, kerY = np.meshgrid(kerX, kerY)
    rho = np.sqrt(kerX**2 + kerY**2)
    # when the value of rho at center is zero, some lat values explode to `nan`.
    if Kh % 2 and Kw % 2:
      rho[Kh // 2][Kw // 2] = 1e-8

    nu = np.arctan(rho)
    cos_nu = np.cos(nu)
    sin_nu = np.sin(nu)
    h_range = np.arange(0, height, stride_h)
    w_range = np.arange(0, width, stride_w)

    lat_range = ((h_range / height) - 0.5) * np.pi
    lon_range = ((w_range / width) - 0.5) * (2 * np.pi)

    # generate latitude sampling pattern
    lat = np.array([np.arcsin(cos_nu * np.sin(_lat) + kerY * sin_nu * np.cos(_lat) / rho) for _lat in lat_range])  # (H, Kh, Kw)

    lat = np.array([lat for _ in lon_range])  # (W, H, Kh, Kw)
    lat = lat.transpose((1, 0, 2, 3))  # (H, W, Kh, Kw)

    # generate longitude sampling pattern
    # Note: use atan2 for 2pi value range
    lon = np.array([np.arctan2(kerX * sin_nu, (rho * np.cos(_lat) * cos_nu - kerY * np.sin(_lat) * sin_nu)) for _lat in lat_range])  # (H, Kh, Kw)

    lon = np.array([lon + _lon for _lon in lon_range])  # (W, H, Kh, Kw)
    lon = lon.transpose((1, 0, 2, 3))  # (H, W, Kh, Kw)

    # (radian) -> (index of pixel)
    lat = (lat / np.pi + 0.5) * height
    lon = ((lon / (2 * np.pi) + 0.5) * width) % width
    if self.sphereType == 'ERP':
      LatLon = np.stack((lat, lon)).astype(np.float32)  # (2, H, W, Kh, Kw) = ((lat, lon), H, W, Kh, Kw)
      LatLon = LatLon.transpose((3, 4, 0, 1, 2))  # (Kh, Kw,2, H, W) = (Kh, Kw,(lat, lon), H, W)
      Kh, Kw, d, H, W = LatLon.shape
      LatLon = LatLon.reshape((1, d * Kh * Kw, H, W))  # (1, 2*Kh*Kw, H, W)
    else:  #cassini
      # TODO: build for both ERP and Cassini
      LatLon = np.stack((lon, lat)).astype(np.float32)
      LatLon = LatLon.transpose((3, 4, 0, 2, 1))
      Kh, Kw, d, H, W = LatLon.shape
      LatLon = LatLon.reshape((1, d * Kh * Kw, H, W))
    return torch.from_numpy(LatLon)

  def forward(self, x):
    pos = self.position.cuda()  # only for cuda version
    # b, c, h, w = x.shape
    # pos = self.position.repeat([b, 1, 1, 1]).cuda()
    return sphere_conv(x, pos, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

  def getPosition(self):
    return self.position

  # def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
  #   version = local_metadata.get('version', None)

  #   if version is None or version < 2:
  #     # the key is different in early versions
  #     # In version < 2, DeformConvPack loads previous benchmark models.
  #     if (prefix + 'conv_offset.weight' not in state_dict and prefix[:-1] + '_offset.weight' in state_dict):
  #       state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(prefix[:-1] + '_offset.weight')
  #     if (prefix + 'conv_offset.bias' not in state_dict and prefix[:-1] + '_offset.bias' in state_dict):
  #       state_dict[prefix + 'conv_offset.bias'] = state_dict.pop(prefix[:-1] + '_offset.bias')

  #   super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


if __name__ == '__main__':
  spc = SphereConv(in_height=5, in_width=10, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, sphereType='ERP').cuda()
  x = torch.randn((1, 1, 5, 10)).float().cuda()
  t = torch.ones_like(x)
  opt = torch.optim.SGD(spc.parameters(), lr=0.1)
  spc.train()
  opt.zero_grad()
  print("x: \n", x)
  print("t: \n", t)
  print("spc:\n")
  for k, v in spc.state_dict().items():
    print(k, v)
  out = spc(x)
  loss = F.smooth_l1_loss(out, t)
  print(loss)
  loss.backward()
  opt.step()
  print("spc:\n")
  for k, v in spc.state_dict().items():
    print(k, v)
