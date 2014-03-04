from .pycaffe import *
import numpy as np
import scipy.ndimage.filters as filters

def compute_gradient(net, inputs, classes=None):
  """
  Compute pixelwise gradients w.r.t. a given class (or default to prediction).

  Take
    net: CaffeNet, initialized.
    inputs: list of images as ndarrays prepared for Caffe
    classes: list of classes to compute the gradient for. -1 sets all classes for
             a class-independent gradient.
  """
  # determine net io dimensions
  in_num = net.blobs()[0].num
  in_size = net.blobs()[0].width

  data_arr = np.zeros((in_num, 3, in_size, in_size))
  data_arr = np.ascontiguousarray(np.concatenate(inputs), dtype=np.float32)
  data_dims = data_arr.shape

  input_blobs = [data_arr]
  output_blobs = [np.empty((10, 1000, 1, 1), dtype=np.float32)]

def flatten_gradient(grad, mode='sumsqr'):
  """
  Flatten gradient across channels into a spatial map.

  Take
    grad: Caffe gradients in K x H x W ndarray
    mode: max = max absolute value
          sum = sum of absolute value
          sumsqr = sum of squares

    give: single channel gradient map in a H x W ndarray
  """
  MODES = ('max', 'sum', 'sumsqr')
  if mode == 'max':
    grad = np.abs(grad).max(0)
  elif mode == 'sum':
    grad = np.abs(grad).sum(0)
  elif mode == 'sumsqr':
    grad = (grad ** 2).sum(0)
  else:
    raise Exception('Unknown mode: not in {}'.format(MODES))
  return grad


def groom_gradient(grad_map, power=0.25, sigma=5):
  """
  Normalize, scale, smooth, and renormalize gradient map.
  Good for visualization.

  Take
    grad_map: gradient map in H x W ndarray
    power: scale to **power
    sigma: standard dev. of Gaussian filter for smoothing

  Give
    grad_map: gradient map ready for visualization.
  """
  grad_map /= grad_map.max()
  grad_map **= power
  grad_map = filters.gaussian_filter(grad_map, sigma=sigma)
  grad_map /= grad_map.max()
  return grad_map
