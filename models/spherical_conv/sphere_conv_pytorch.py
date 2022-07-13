import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# TODO: build spherical CNN


class GridGenerator:
  def __init__(self, height: int, width: int, kernel_size, stride=1, sphereType='ERP'):
    self.height = height
    self.width = width
    self.kernel_size = kernel_size  # (Kh, Kw)
    self.stride = stride  # (H, W)
    self.sphereType = sphereType
    #print(self.stride)

  def createSamplingPattern(self):
    """
    :return: (1, H*Kh, W*Kw, (Lat, Lon)) sampling pattern
    """
    kerX, kerY = self.createKernel()  # (Kh, Kw)

    # create some values using in generating lat/lon sampling pattern
    rho = np.sqrt(kerX**2 + kerY**2)
    Kh, Kw = self.kernel_size
    # when the value of rho at center is zero, some lat values explode to `nan`.
    if Kh % 2 and Kw % 2:
      rho[Kh // 2][Kw // 2] = 1e-8

    nu = np.arctan(rho)
    cos_nu = np.cos(nu)
    sin_nu = np.sin(nu)

    stride_h, stride_w = self.stride
    h_range = np.arange(0, self.height, stride_h)
    w_range = np.arange(0, self.width, stride_w)

    lat_range = ((h_range / self.height) - 0.5) * np.pi
    lon_range = ((w_range / self.width) - 0.5) * (2 * np.pi)

    # generate latitude sampling pattern
    lat = np.array([np.arcsin(cos_nu * np.sin(_lat) + kerY * sin_nu * np.cos(_lat) / rho) for _lat in lat_range])  # (H, Kh, Kw)

    lat = np.array([lat for _ in lon_range])  # (W, H, Kh, Kw)
    lat = lat.transpose((1, 0, 2, 3))  # (H, W, Kh, Kw)

    # generate longitude sampling pattern
    lon = np.array([np.arctan2(kerX * sin_nu, (rho * np.cos(_lat) * cos_nu - kerY * np.sin(_lat) * sin_nu)) for _lat in lat_range])  # (H, Kh, Kw)

    lon = np.array([lon + _lon for _lon in lon_range])  # (W, H, Kh, Kw)
    lon = lon.transpose((1, 0, 2, 3))  # (H, W, Kh, Kw)

    # (radian) -> (index of pixel)
    lat = (lat / np.pi + 0.5) * self.height
    lon = ((lon / (2 * np.pi) + 0.5) * self.width) % self.width

    # if self.sphereType == 'ERP':
    #   LatLon = np.stack((lat, lon)).astype(np.float32)  # (2, H, W, Kh, Kw) = ((lat, lon), H, W, Kh, Kw)
    #   LatLon = LatLon.transpose((3, 4, 0, 1, 2))  # (Kh, Kw,2, H, W) = (Kh, Kw,(lat, lon), H, W)
    #   Kh, Kw, d, H, W = LatLon.shape
    #   LatLon = LatLon.reshape((1, d * Kh * Kw, H, W))  # (1, 2*Kh*Kw, H, W)
    # else:  #cassini
    #   LatLon = np.stack((lon, lat)).astype(np.float32)
    #   LatLon = LatLon.transpose((3, 4, 0, 2, 1))
    #   Kh, Kw, d, H, W = LatLon.shape
    #   LatLon = LatLon.reshape((1, d * Kh * Kw, H, W))
    # return torch.from_numpy(LatLon)
    if self.sphereType == 'ERP':
      LatLon = np.stack((lat, lon))  # (2, H, W, Kh, Kw) = ((lat, lon), H, W, Kh, Kw)
      LatLon = LatLon.transpose((1, 3, 2, 4, 0))  # (H, Kh, W, Kw, 2) = (H, Kh, W, Kw, (lat, lon))
      H, Kh, W, Kw, d = LatLon.shape
      LatLon = LatLon.reshape((1, H * Kh, W * Kw, d))  # (1, H*Kh, W*Kw, 2)
    else:
      LatLon = np.stack((lon, lat))  # (2, H, W, Kh, Kw) = ((lat, lon), H, W, Kh, Kw)
      LatLon = LatLon.transpose((2, 4, 1, 3, 0))  # (H, Kh, W, Kw, 2) = (H, Kh, W, Kw, (lat, lon))
      H, Kh, W, Kw, d = LatLon.shape
      LatLon = LatLon.reshape((1, H * Kh, W * Kw, d))  # (1, H*Kh, W*Kw, 2)
    return LatLon

  def createKernel(self):
    """
    :return: (Ky, Kx) kernel pattern
    """
    Kh, Kw = self.kernel_size

    delta_lat = np.pi / self.height
    delta_lon = 2 * np.pi / self.width

    range_x = np.arange(-(Kw // 2), Kw // 2 + 1)
    if not Kw % 2:
      range_x = np.delete(range_x, Kw // 2)

    range_y = np.arange(-(Kh // 2), Kh // 2 + 1)
    if not Kh % 2:
      range_y = np.delete(range_y, Kh // 2)

    kerX = np.tan(range_x * delta_lon)
    kerY = np.tan(range_y * delta_lat) / np.cos(range_y * delta_lon)

    return np.meshgrid(kerX, kerY)  # (Kh, Kw)
    # return (kerX, kerY)


class SphereConv2d(nn.Conv2d):
  """
  kernel_size: (H, W)
  """
  def __init__(self,
               in_height,
               in_width,
               sphereType,
               in_channels: int,
               out_channels: int,
               kernel_size=(3,
                            3),
               stride=1,
               padding=0,
               dilation=1,
               groups: int = 1,
               bias: bool = True,
               padding_mode: str = 'zeros'):
    super(SphereConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
    self.grid_shape = (in_height, in_width)
    self.sphereType = sphereType
    self.grid = None
    in_h = min(in_height, in_width)
    in_w = max(in_height, in_width)
    assert in_w == 2 * in_h
    self.genSamplingPattern(in_h, in_w)

  def genSamplingPattern(self, h, w):
    gridGenerator = GridGenerator(h, w, self.kernel_size, self.stride, self.sphereType)
    LonLatSamplingPattern = gridGenerator.createSamplingPattern()

    # generate grid to use `F.grid_sample`
    lat_grid = (LonLatSamplingPattern[:, :, :, 0] / h) * 2 - 1
    lon_grid = (LonLatSamplingPattern[:, :, :, 1] / w) * 2 - 1

    grid = np.stack((lon_grid, lat_grid), axis=-1)
    with torch.no_grad():
      self.grid = torch.FloatTensor(grid)
      self.grid.requires_grad = False
      self.grid = self.grid.cuda()

  def forward(self, x):
    # Generate Sampling Pattern
    B, C, H, W = x.shape

    # if (self.grid_shape is None) or (self.grid_shape != (H, W)):
    #   self.grid_shape = (H, W)
    #   self.genSamplingPattern(H, W)

    with torch.no_grad():
      grid = self.grid.repeat((B, 1, 1, 1))  # (B, H*Kh, W*Kw, 2)
      grid.requires_grad = False

    x = F.grid_sample(x, grid, align_corners=True, mode='nearest')  # (B, in_c, H*Kh, W*Kw)

    # self.weight -> (out_c, in_c, Kh, Kw)
    x = F.conv2d(x, self.weight, self.bias, stride=self.kernel_size)

    return x  # (B, out_c, H/stride_h, W/stride_w)


class GirdCassini:
  def __init__(self, height, width, kernelSize, stride=1):
    self.height = height
    self.width = width
    self.kernelSize = kernelSize

  def __genSamplingPattern(self):
    kernelH, kernelW = self.__genKernel()

  def __genKernel(self):
    dh = 2 * np.pi / self.height
    dw = np.pi / self.width
    kh = self.kernelSize
    kw = self.kernelSize
    rangeH = np.arange(-(kh // 2), kh // 2 + 1)
    rangeW = np.arange(-(kw // 2), kw // 2 + 1)
    if not kh % 2:
      rangeH = np.delete(rangeH, kh // 2)
    if not kw % 2:
      rangeW = np.delete(rangeW, kw // 2)
    kernelH = np.tan(rangeH * dh)
    kernelW = np.tan(rangeW * dw) / np.cos(rangeW * dh)
    print(kernelH.shape, kernelW.shape)
    print("kernelH:\n{}".format(kernelH))
    print("kernelW:\n{}".format(kernelW))
    #return np.meshgrid(kernelH, kernelW)
    return kernelH, kernelW


if __name__ == '__main__':
  cassini = True
  rootDir = 'Y:/projects/omniDepth/tmp'
  h, w = 64, 128
  gridg = GridGenerator(h, w, 3)
  g = gridg.createSamplingPattern()
  print("g: ", g.shape)
  #print("g ca: ", gca.shape)
  x0, y0 = 125, 60

  print('Cassini')
  imgName = 'e2ca_0_0b217f59904d4bdf85d35da2cab963471_color_0_Left_Down_0.0.png'
  img = cv2.imread(os.path.join(rootDir, imgName))
  print(img.shape)

  biasx, biasy = x0, y0
  for i in range(3):
    for j in range(3):
      x = gca[0, i + biasx * 3, j + biasy * 3, 0]
      y = gca[0, i + biasx * 3, j + biasy * 3, 1]
      xn = int(round(x))
      yn = int(round(y))
      xn = xn % w
      yn = yn % h
      print('[', x, ',', y, ']', ' [', xn, ',', yn, ']')
      img[xn, yn, 0:2] = 0
      img[xn, yn, 2] = 255
  cv2.imwrite(os.path.join(rootDir, "mark_ca.png"), img)
  #cv2.imshow("show", img)
  #cv2.waitKey(0)

  print('ERP')
  imgName = '0_0b217f59904d4bdf85d35da2cab963471_color_0_Left_Down_0.0.png'
  img = cv2.imread(os.path.join(rootDir, imgName))
  print(img.shape)

  biasx, biasy = y0, x0
  for i in range(3):
    for j in range(3):
      x = g[0, i + biasx * 3, j + biasy * 3, 0]
      y = g[0, i + biasx * 3, j + biasy * 3, 1]
      xn = int(round(x))
      yn = int(round(y))
      xn = xn % h
      yn = yn % w
      print('[', x, ',', y, ']', ' [', xn, ',', yn, ']')
      img[xn, yn, 0:2] = 0
      img[xn, yn, 2] = 255
  cv2.imwrite(os.path.join(rootDir, "mark_erp.png"), img)
  #cv2.imshow("show", img)
  #cv2.waitKey(0)
