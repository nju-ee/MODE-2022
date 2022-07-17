import torch


def mae(pred, gt):
  return torch.mean(torch.abs(pred - gt)).cpu().numpy()


def max_ae(pred, gt):
  return torch.max(torch.abs(pred - gt)).cpu().numpy()


def rmse(pred, gt):
  return torch.sqrt(torch.mean(torch.square(pred - gt))).cpu().numpy()


def absrel(pred, gt):
  mask = gt > 0
  return torch.mean(torch.abs(pred[mask] - gt[mask]) / gt[mask]).cpu().numpy()


def sqrel(pred, gt):
  mask = gt > 0
  return torch.mean(torch.square(pred[mask] - gt[mask]) / torch.square(gt[mask])).cpu().numpy()


def silog(pred, gt):
  # sqrt of the silog(following KITTI)
  mask1 = gt > 0
  mask2 = pred > 0
  mask = mask1 * mask2
  d = torch.log(pred[mask]) - torch.log(gt[mask])
  return torch.sqrt(torch.mean(torch.square(d)) - torch.square(torch.mean(d))).cpu().numpy()


def pixel_error_pct(th_pixel, pred, gt):
  error = torch.abs(pred - gt)
  return 100 * torch.numel(error[error >= th_pixel]) / torch.numel(error)


def D1(th_pixel, th_pct, pred, gt):
  error = torch.abs(pred - gt)
  return 100 * torch.numel(error[(error >= th_pixel) * (error >= th_pct * gt)]) / torch.numel(error)


def delta_acc(exp, pred, gt):
  error = torch.max(pred / gt, gt / pred)
  return 100 * torch.numel(error[error < 1.25**exp]) / torch.numel(error)


def threshold_acc(err_pct, pred, gt):
  error = torch.max(pred / gt, gt / pred)
  return 100 * torch.numel(error[error < (1 + err_pct)]) / torch.numel(error)
