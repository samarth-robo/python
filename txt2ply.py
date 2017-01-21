import sys
from IPython.core.debugger import Tracer

class TXTConverter:
  def __init__(self, input_txt_filename, output_ply_filename):
    self.input_txt_filename = input_txt_filename
    self.output_ply_filename = output_ply_filename
    self.ply_header = ('ply\n'
        'format ascii 1.0\n'
        'element vertex %d\n'
        'property float x\n'
        'property float y\n'
        'property float z\n'
        'property uchar red\n'
        'property uchar green\n'
        'property uchar blue\n'
        'end_header\n')
              
  def convert(self):
    with open(self.input_txt_filename, 'r') as fin, open(self.output_ply_filename, 'w') as fout:
      lines = [l for l in fin]
      fout.write(self.ply_header % len(lines))
      for line in lines:
        x, y, z, r, g, b = line.strip().split(' ')
        out_str = '%f %f %f %d %d %d\n' % (float(x), float(y), float(z),
            int(float(r)), int(float(g)), int(float(b)))
        fout.write(out_str)

if __name__ == '__main__':
  if len(sys.argv) is not 3:
    print 'Usage: python txt2ply.py input_txt.txt output_ply.ply'
    sys.exit(-1)

  c = TXTConverter(sys.argv[1], sys.argv[2])
  c.convert()
