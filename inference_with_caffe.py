import numpy as np
import sys
import caffe
import argparse




parser = argparse.ArgumentParser(description=' Inference benchmark for Intel Chainer')
parser.add_argument('--arch', '-a', default='alexnet', help='Convnet architecture \
                    (alexnet, googlenet, rcnn, caffenet)')
parser.add_argument('--batchsize', '-B', type=int, default=1, help='minibatch size')
parser.add_argument('--mean', '-m', default='ilsvrc_2012_mean.npy',
                    help='Path to the mean file')

args = parser.parse_args()

# Add caffe binary to $PATH
caffe_root = '../caffe'
sys.path.insert(0, caffe_root + 'python')


# Setup
caffe.set_mode_cpu()
model_def 
