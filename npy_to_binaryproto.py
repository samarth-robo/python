import caffe
import numpy as np

def np_to_proto(mm):
  """
  convert np array to caffe binaryproto
  """
  mm = np.expand_dims(mm, axis=3)
  mm = np.transpose(mm, (3, 2, 0, 1)) # reshape to n x c x h x w
  # print 'size %d x %d x %d' % (mm.shape[2], mm.shape[3], mm.shape[1])

  bb = caffe.io.array_to_blobproto(mm)
  return bb

def proto_to_np(proto_fn):
  """
  convert binaryproto filename to np array
  """
  blob = caffe.proto.caffe_pb2.BlobProto()
  with open(proto_fn, 'rb') as f:
    data = f.read()
  blob.ParseFromString(data)
  arr = np.asarray(caffe.io.blobproto_to_array(blob))
  return arr
