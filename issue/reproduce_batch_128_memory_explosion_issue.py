# import Chainer related module
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.links.caffe import CaffeFunction

import time
from util import ProgressBar

# 
# model parameter
# 
W = 224
H = 224
C = 3
N = 128 # mini-batch size
model_path = './bvlc_googlenet.caffemodel'
estimate_load_time = 22

# switch Chainer configuration support inference
chainer.config.train = False

# dummy data
x_data = np.ndarray((N, C, H, W), dtype=np.float32)
x = Variable(x_data)

progress_bar = ProgressBar(estimate_load_time)
progress_bar.start()

func = CaffeFunction(model_path)

progress_bar.end()
time.sleep(1)

print 'start to calculate y'
y, = func(inputs={'data': x}, outputs=['loss3/classifier'],
                  disable=['loss1/ave_pool', 'loss2/ave_pool'])
print 'end to calculate y'
