# program to transfer weights from src_model to dst_model
# if dst_net is completely new, make dst_model = src_model
# transfer_names.txt has format:
# src_layer1 dst_layer1
# src_layer2 dst_layer2...

import os
import numpy as np
import caffe
import sys
from IPython.core.debugger import Tracer

if len(sys.argv) != 7:
  print 'Usage: python surgery.py dst_proto dst_model src_proto src_model \
      save_model transfer_names.txt'
  sys.exit(-1)

dst_proto  = sys.argv[1]
dst_model  = sys.argv[2]
src_proto  = sys.argv[3]
src_model  = sys.argv[4]
save_model = sys.argv[5]
transfer_names_file = sys.argv[6]

src_names = []
dst_names = []
with open(transfer_names_file, 'r') as f:
  for line in f:
    s, d = line.strip().split(' ')
    src_names.append(s)
    dst_names.append(d)

dst_net = caffe.Net(dst_proto, dst_model, caffe.TEST)
Tracer()()
src_net = caffe.Net(src_proto, src_model, caffe.TEST)

for dst_layer, src_layer in zip(dst_names, src_names):
  for i in xrange(len(src_net.params[src_layer])):
    dst_net.params[dst_layer][i].data[...] = src_net.params[src_layer][i].data
  print 'Transferred', src_layer, '->', dst_layer

dst_net.save(save_model)
print 'Surgery done, saved', save_model
Tracer()()
check_net = caffe.Net(dst_proto, save_model, caffe.TEST)
