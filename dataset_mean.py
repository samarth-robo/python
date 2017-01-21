# program to calculate the mean images of a large dataset of images
import glob
import cv2
import sys
import numpy as np
from proto_utils import np_to_proto

class DatasetMeaner:
  """
  calculate mean of large dataset and save it as .binaryproto and .npy
  """
  def __init__(self, input_dir, out_filename, h, w):
    print 'Gathering image data...'
    self.im_names = glob.glob(input_dir + '/*.jpg')
    print 'Found {:d} images'.format(len(self.im_names))
    self.out_filename = out_filename
    self.h = h
    self.w = w

  def save_as_binaryproto(self, n, filename):
    if (self.h > 0) and (self.w > 0):
      n = cv2.resize(n, (self.w, self.h))
    b = np_to_proto.convert(n)
    with open(filename, 'wb') as f:
      f.write(b.SerializeToString())
    print 'Saved ', filename
    npy_filename = '{:s}.npy'.format(filename.split('.')[0])
    np.save(npy_filename, n)
    print 'Saved ', npy_filename

  def save_mean(self):
    acc = cv2.imread(self.im_names[0]).astype(float)
    if acc is None:
      print 'Could not read ', im_names[0]
      return
    count = 0
    for im_name in self.im_names[1:]:
      if (count % 100 == 0) and (count > 0):
        print 'Accumulated {:d} images / {:d}'.format(count, len(self.im_names))
        if count % 5000 == 0:
          mean_im = acc / count
          self.save_as_binaryproto(mean_im, self.out_filename)
      im = cv2.imread(im_name)
      if im is None:
        print 'Could not read ', im_name
        continue
      acc += im.astype(float)
      count += 1
    mean_im = acc / count
    self.save_as_binaryproto(mean_im, self.out_filename)

if __name__ == '__main__':
  if len(sys.argv) == 5:
    input_dir = sys.argv[1]
    out_filename = sys.argv[2]
    h = int(sys.argv[3])
    w = int(sys.argv[4])
  elif len(sys.argv) == 3:
    input_dir = sys.argv[1]
    out_filename = sys.argv[2]
    h = -1
    w = -1
  else:
    print 'Usage: python dataset_mean.py input_dir out.binaryproto [h w]'
    sys.exit(-1)

  m = DatasetMeaner(input_dir, out_filename, h, w)
  m.save_mean()
