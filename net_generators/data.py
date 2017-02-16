"""
commonly used data layers
"""
import caffe
from caffe import layers as L, params as P
import json

def dummy_data(ns, **kwargs):
  """
  adds a DummyData layer to the netspec
  :param ns: netspec
  :param kwargs:
  :return: netspec, top
  """
  data = 'data'
  ns[data] = L.DummyData(**kwargs)
  return ns, data

def multi_input_data(ns, batch_size, shuffle, source, root_folder, new_height,
                     new_width, crop_size, mean_fn, phase, stage):
  """
  adds a multi_input_data layer to the netspec
  :param ns: netspec
  :param batch_size
  :param shuffle
  :param source: sources for data (TXT or H5 files)
  :param root_folder base directory for image names in TXT files
  :param new_height: h to which image is resized
  :param new_width: w to which image is resized
  :param crop_size: size of square crop from resized image
  :param mean_fn: mean binaryproto filename
  :param phase: phase of the layer (TRAIN / TEST)
  :param stage: stage of the layer (train / val / deploy)
  :return: netspec, tops
  """
  top_names=[k for k,_ in source.items()]
  param_dict = dict(batch_size=batch_size, shuffle=shuffle,
                    top_names=top_names, source=source, root_folder=root_folder,
                    new_height=new_height, new_width=new_width,
                    crop_size=crop_size, mean_file=mean_fn,
                    multithreaded_prefetch=True, multithreaded_preprocess=True)
  param_str = json.dumps(param_dict)
  tops = L.Python(name='data', module='layers.multi_input_data',
                  layer='MultiInputDataLayer', param_str=param_str,
                  include={'phase': phase, 'stage': stage}, ntop=len(top_names))
  if len(top_names) == 1:
    tops = [tops]

  for tn, top in zip(top_names, tops):
    ns[tn] = top

  return ns, tuple(top_names)

def deploy_data(ns, top_names, shapes):
  """
  adds a Input Layer to the netspec. phase = TEST, stage = deploy
  :param ns: netspec
  :param top_names: names of tops
  :param shapes: shapes of tops (list of lists)
  :return: netspec, tops
  """
  assert(len(top_names) == len(shapes))
  shape_list = [{'dim': s} for s in shapes]
  tops = L.Input(name='data', shape=shape_list, ntop=len(top_names),
                 include={'phase': caffe.TEST, 'stage': 'deploy'})
  if len(top_names) == 1:
    tops = [tops]

  for tn, top in zip(top_names, tops):
    ns[tn] = top

  return ns, tuple(top_names)
