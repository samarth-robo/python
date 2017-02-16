"""
program to calculate the mean images of a large dataset of images
"""
import cv2
import os.path as osp
import numpy as np
from proto_utils import np_to_proto
import argparse

class DatasetMeaner:
  """
  calculate mean of large dataset and save it as .binaryproto and .npy
  """
  def __init__(self, data_dir, list_file, out_dir, height, width):
    self.height = height
    self.width = width
    self.out_dir = osp.expanduser(out_dir)
    print 'Reading list file ...'
    with open(osp.expanduser(list_file), 'r') as f:
      self.im_names = [osp.join(osp.expanduser(data_dir),
                                l.rstrip().split(' ')[0]) for l in f]
    print 'Found {:d} image names'.format(len(self.im_names))

  def save_mean(self, m):
    if (self.height > 0) and (self.width > 0):
      m = cv2.resize(m, (self.width, self.height))
    b = np_to_proto(m)

    bproto_fn = osp.join(self.out_dir, 'mean_{:d}_{:d}.binaryproto'.\
                         format(self.height, self.width))
    npy_fn    = osp.join(self.out_dir, 'mean_{:d}_{:d}.npy'.\
                         format(self.height, self.width))
    with open(bproto_fn, 'wb') as f:
      f.write(b.SerializeToString())
    print 'Saved ', bproto_fn
    np.save(npy_fn, m)
    print 'Saved ', npy_fn

  def collect_mean(self):
    acc = cv2.imread(self.im_names[0]).astype(float)
    if acc is None:
      print 'Could not read ', self.im_names[0]
      return
    count = 0
    for im_name in self.im_names[1:]:
      if (count % 100 == 0) and (count > 0):
        print 'Accumulated {:d} images / {:d}'.format(count, len(self.im_names))
        if count % 5000 == 0:
          mean_im = acc / count
          self.save_mean(mean_im)
      im = cv2.imread(im_name)
      if im is None:
        print 'Could not read ', im_name
        continue
      acc += im.astype(float)
      count += 1
    mean_im = acc / count
    self.save_mean(mean_im)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('data_dir', help='Base directory for image files')
  parser.add_argument('list_file', help='TXT file with list of image names')
  parser.add_argument('out_dir', help='Directory for writing output .binaryproto'\
                                           'and .npy files')
  parser.add_argument('--height', type=int, default=-1)
  parser.add_argument('--width', type=int, default=-1)

  args = parser.parse_args()
  dm = DatasetMeaner(args.data_dir, args.list_file, args.out_dir, args.height,
                     args.width)
  dm.collect_mean()