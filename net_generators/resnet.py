import caffe
from caffe import layers as L, params as P

def resnet_cbs_block(ns, name, bottom, nout, phase=caffe.TRAIN,
                     name_suffix='', inc_exc_dict={}, **kwargs):
  """
  adds a conv + bn + scale block to netspec ns
  """
  conv = 'res{:s}{:s}'.format(name, name_suffix)
  conv_args = dict(kwargs, param={'name': 'res{:s}_w'.format(name)},
                   **inc_exc_dict)
  if phase is caffe.TRAIN:
    conv_args = dict(conv_args, weight_filler=dict(type='msra'))
  ns[conv] = L.Convolution(ns[bottom], num_output=nout, **conv_args)

  bn = 'bn{:s}{:s}'.format(name, name_suffix)
  ns[bn] = L.BatchNorm(ns[conv], in_place=True,
                       param=[{'name': 'bn{:s}_m'.format(name)},
                              {'name': 'bn{:s}_v'.format(name)},
                              {'name': 'bn{:s}_b'.format(name)}],
                       **inc_exc_dict)

  scale = 'scale{:s}{:s}'.format(name, name_suffix)
  ns[scale] = L.Scale(ns[bn], bias_term=True, in_place=True,
                      param=[{'name': 'scale{:s}_s'.format(name)},
                             {'name': 'scale{:s}_b'.format(name)}],
                      **inc_exc_dict)

  return ns, scale

def resnet_cbsr_block(ns, name, bottom, nout, phase=caffe.TRAIN,
                      name_suffix='', inc_exc_dict={}, **kwargs):
  """
  adds a conv + bn + scale + relu block to netspec
  :param ns: netspec
  :param name: suffix name for layers
  :param bottom: input blob to the block
  :param nout: number of output channels of the block
  :param phase: net phase (train / test)
  :param name_suffix: suffix to be added after layer names
  :param inc_exc_dict: dictionary holding include / exclude information
  :param kwargs: args for conv layer
  :return: netspec, top
  """
  ns, scale = resnet_cbs_block(ns, name, bottom, nout, phase, name_suffix,
                               inc_exc_dict, **kwargs)

  relu = 'res{:s}_relu{:s}'.format(name, name_suffix)
  ns[relu] = L.ReLU(ns[scale], in_place=True, **inc_exc_dict)

  return ns, relu

def resnet_block(ns, idx, bottom, nout, n_sblocks, subsample=False,
    phase=caffe.TRAIN, name_suffix='', inc_exc_dict={}):
  """
  adds a resnet block to the netspec
  :param ns: netspec
  :param idx: index of this resnet block in the net (used to generate layer names)
  :param bottom: input blob to the block
  :param nout: number of output channels
  :param n_sblocks: number of block with identity shortcuts
  :param subsample: first conv layers in both branches will have stride=2 if True
  :param phase: net phase (train / test)
  :param name_suffix: suffix to be added after layer names
  :param inc_exc_dict: dictionary holding include / exclude information
  :return: netspec, top
  """
  assert (nout % 4 == 0)
  alphabet = 'abcdefghijklmnopqrstuvwxyz'
  bm = bottom
  for sblock in xrange(n_sblocks):
    name = '{:d}{:s}'.format(idx, alphabet[sblock])
    first_conv_stride = 2 if (subsample and (sblock==0)) else 1
    ns, bm_b = resnet_cbsr_block(ns, '{:s}_branch2a'.format(name), bm, nout/4,
                                 phase, name_suffix, inc_exc_dict,
                                 kernel_size=1, stride=first_conv_stride, pad=0,
                                 bias_term=False)
    ns, bm_b = resnet_cbsr_block(ns, '{:s}_branch2b'.format(name), bm_b, nout/4,
                                 phase, name_suffix, inc_exc_dict, kernel_size=3,
                                 stride=1, pad=1, bias_term=False)
    ns, bm_b = resnet_cbs_block(ns, '{:s}_branch2c'.format(name), bm_b, nout/1,
                                phase, name_suffix, inc_exc_dict, kernel_size=1,
                                stride=1, pad=0, bias_term=False)
    if sblock == 0:
      ns, bm = resnet_cbs_block(ns, '{:s}_branch1'.format(name), bm, nout/1,
                                phase, name_suffix, inc_exc_dict,
                                kernel_size=1, stride=first_conv_stride, pad=0,
                                bias_term=False)

    eltwise = 'res{:s}{:s}'.format(name, name_suffix)
    ns[eltwise] = L.Eltwise(ns[bm], ns[bm_b], **inc_exc_dict)

    relu = 'res{:s}_relu{:s}'.format(name, name_suffix)
    ns[relu] = L.ReLU(ns[eltwise], in_place=True, **inc_exc_dict)

    bm = relu
  return ns, bm

def resnet50(ns, bottom, sblocks=None, n_outs=None, phase=caffe.TRAIN,
             name_suffix='', exclude_stages=[]):
  """
  adds a ResNet50 to the netspec
  :param ns: netspec
  :param bottom: input blob to the net
  :param sblocks: sub-block structure (see Table 1 in paper)
  :param n_outs: output channel configuration (see Table 1 in paper)
  :param phase: net phase (train / test)
  :param name_suffix: suffix to be added after layer names
  :param exclude_stages: list of stages of training during which this block
  should be excluded
  :return: netspec, top
  """
  if (sblocks is None) or (n_outs is None):
    # default structure for ResNet50
    sblocks = [3, 4, 6, 3]
    n_outs = [256, 512, 1024, 2048]

  exclude_dict = {'exclude': [{'stage': s} for s in exclude_stages]} if \
    len(exclude_stages) > 0 else {}

  conv = 'conv1{:s}'.format(name_suffix)
  conv_args = {'kernel_size': 7, 'pad': 3, 'stride': 2,
               'param': [{'name': 'conv1_w'}, {'name': 'conv1_b'}]}
  conv_args = dict(conv_args, **exclude_dict)
  if phase is caffe.TRAIN:
    conv_args = dict(conv_args, weight_filler=dict(type='msra'),
                     bias_filler=dict(type='constant'))
  ns[conv] = L.Convolution(ns[bottom], num_output=64, **conv_args)

  bn = 'bn_conv1{:s}'.format(name_suffix)
  ns[bn] = L.BatchNorm(ns[conv], in_place=True,
                       param=[{'name': 'bn_conv1_m'}, {'name': 'bn_conv1_v'},
                              {'name': 'bn_conv1_b'}], **exclude_dict)

  scale = 'scale_conv1{:s}'.format(name_suffix)
  ns[scale] = L.Scale(ns[bn], bias_term=True, in_place=True,
                      param=[{'name': 'scale_conv1_s'}, {'name': 'scale_conv1_b'}],
                      **exclude_dict)

  relu = 'conv1_relu{:s}'.format(name_suffix)
  ns[relu] = L.ReLU(ns[scale], in_place=True, **exclude_dict)

  pool = 'pool1{:s}'.format(name_suffix)
  ns[pool] = L.Pooling(ns[relu], kernel_size=3, stride=2, pool=P.Pooling.MAX,
                       **exclude_dict)

  bm = pool

  for i, (n_sblocks, n_out) in enumerate(zip(sblocks, n_outs)):
    subsample = True if i > 0 else False
    ns, bm = resnet_block(ns, i+2, bm, n_out, n_sblocks, subsample, phase,
                          name_suffix, inc_exc_dict=exclude_dict)

  # final global average pooling layer
  ga_pool = 'pool{:d}{:s}'.format(len(sblocks)+1, name_suffix)
  ns[ga_pool] = L.Pooling(ns[bm], stride=1, pool=P.Pooling.AVE,
      global_pooling=True, **exclude_dict)
  bm = ga_pool

  return ns, bm
