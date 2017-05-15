#! /usr/bin/python

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from chainer.functions.connection import convolution_2d
from chainer.function_hooks import timer
from util import LayerTimer
from util import TimerHook
from util import ProgressBar

from chainer.links.caffe import CaffeFunction
import time
import argparse
import cv2
import os
import sys

parser = argparse.ArgumentParser(description=' Inference benchmark for Intel Chainer')
parser.add_argument('--arch', '-a', default='alexnet', help='Convnet architecture \
                    (alexnet, googlenet, rcnn, caffenet)')
parser.add_argument('--batchsize', '-B', type=int, default=1, help='minibatch size')
parser.add_argument('--mean', '-m', default='ilsvrc_2012_mean.npy',
                    help='Path to the mean file')

args = parser.parse_args()
print 'lol'
# N, C, W, H
W = 0
H = 0
C = 0
N = args.batchsize

# set dataset
dataset = './ILSVRC2012_img_val'
label_file = './val.txt'

# *.caffemodel file path
model_path = '' 

# set initial category number
label_category = 1000

# mean image
mean_image = np.load(args.mean)

if args.arch == 'alexnet':
	estimate_load_time = 120
	W = 227
	H = 227
	C = 3 
	label_category = 1000
	model_path = './bvlc_alexnet.caffemodel'
	def forward(x):
		y, = func(inputs={'data': x}, outputs=['fc8'])
		return y
elif args.arch == 'googlenet':
	estimate_load_time = 22
	W = 224
	H = 224
	C = 3
	model_path = './bvlc_googlenet.caffemodel'
	def forward(x):
		y, = func(inputs={'data': x}, outputs=['loss3/classifier'],
                  disable=['loss1/ave_pool', 'loss2/ave_pool'])
		return y
elif args.arch == 'rcnn':
	pass
elif args.arch == 'caffenet':
	estimate_load_time = 120
	W = 224
	H = 224
	C = 3
	model_path = './bvlc_reference_caffenet.caffemodel'
	def forward(x):
		y, = func(inputs={'data': x}, outputs=['fc8'])
		return y
else:
	raise ValueError('Oops, unknown arch name!')

cropwidth = 256 - W
start = cropwidth // 2
stop = start + W
mean_image = mean_image[:, start:stop, start:stop].copy()

def batch_generator():
	f = open(label_file, 'r')
	lines = f.readlines()
	for l in lines:
		pic = l.split(' ')[0]
		label = l.split(' ')[1]
		yield (pic, int(label))

def imresize(pic_data):
	output_side_length=256

	height = pic_data.shape[0]
	width = pic_data.shape[1]

	new_height = output_side_length
	new_width = output_side_length

	if height > width:
 		new_height = output_side_length * height / width
	else:
 		new_width = output_side_length * width / height

	resized_pic_data = cv2.resize(pic_data, (new_width, new_height))
	height_offset = (new_height - output_side_length) / 2
	width_offset = (new_width - output_side_length) / 2
	resized_pic_data = resized_pic_data[height_offset:height_offset + output_side_length,
	width_offset:width_offset + output_side_length]

	return resized_pic_data


def get_mini_batch(batch_size, bg):
	batch = np.ndarray((batch_size, C, H, W), dtype=np.float32)
	label_vec = np.zeros((batch_size, label_category))

	i = 0
	while i < batch_size:
		
		pic_path, label = bg.next()
		pic_data= cv2.imread(os.path.join(dataset, pic_path))

		# resize the image before cropping
		pic_data = imresize(pic_data)

		if pic_data.ndim < 3 or C < 3:
			continue
		
		# transpose image from (W, H, C) to (C, H, W)
		pic_data = np.transpose(pic_data, (2, 0, 1))


		if pic_data.shape[1] < H + start or pic_data.shape[2] < W + start:
			print pic_data.shape[1], pic_data.shape[2]
			continue
		

		# Convert label to one-hot vector
		label_vec[(i, label)] = 1

		batch[i] = pic_data[:, start:stop, start:stop]

		i += 1

	return (batch, label_vec, label)

layer_time_dict = {}

def print_layer_time(hook):
	global layer_time_dict
	for func, time in hook.call_history:
		# func_name = func.label
		func_name = func
		if layer_time_dict.get(func_name) is None:
			layer_time_dict[func_name] = {'time': time, 'number': 1}
		else:
			layer_time_dict[func_name]['time'] += time
			layer_time_dict[func_name]['number'] += 1

	print '================================================'
	for name, record in layer_time_dict.items():
		print '[{}]:'.format(name)
		print 'total time: {} ms'.format(record['time'] * 1000)
		print 'average time: {} ms\n'.format(float(record['time']) * 1000/ record['number'])
	print '================================================'

# lt = LayerTimer(model_path)
# configuration for inference
chainer.config.train = False
print 'here'
progress_bar = ProgressBar(estimate_load_time)
progress_bar.start()

print 'loading caffe model...'

start_time = time.time()
func = CaffeFunction(model_path)
end_time = time.time()

progress_bar.end()
time.sleep(1)
print '\nsuccessfully load caffe model, it costs %s seconds' % (end_time - start_time)

max_iter = 10
total_time = 0
average_time = 0


# global batch_geneartor
bg = batch_generator()

# count top 1 and top 5 accracy 
top5 = 0
top1 = 0

total_conv2d_time = 0
total_conv2d_layer = 0

timer_hook = TimerHook()

for i in xrange(max_iter):
	print 'start to get mini batch'
	x_data, label, l= get_mini_batch(N, bg)
	print 'end getting mini batch'
	# raise ValueError('debugging')

	# minus mean image
	x_data -= mean_image
	print x_data.shape

	x = Variable(x_data)

	with timer_hook:
		start_time = time.time()
		y = forward(x)
		end_time = time.time()

	y = F.softmax(y)

	inference_time = (end_time - start_time) * 1000
	total_time += inference_time

	# print '[%s] infernece time is %sms' % (str(i), str(inference_time))
	top5_y = np.argpartition(y.data[0], -5)[-5:]
	# print 'l=%s, h=%s, top5=%s' % (str(l), str(np.argmax(y.data)), str(top5))

	if np.argmax(label) in top5_y:
		top5 += 1

	if np.argmax(label) == np.argmax(y.data):
		top1 += 1

	sys.stdout.write('\r[{}/{}]'.format(i + 1, max_iter))
	sys.stdout.flush()


average_time = total_time / max_iter

print '\nTotal time is %s ms' % str(total_time)
print 'Average time is %s ms' % str(average_time)
print 'Top5 accuracy is %s%%' % str(float(top5) * 100 / max_iter)
print 'Top1 accuracy is %s%%' % str(float(top1) * 100 / max_iter)

# print 'Total conv2d time is %s ms' % str(total_conv2d_time)
# print 'Total conv2d time is %s ms' % str(float(total_conv2d_time) / total_conv2d_layer)

print_layer_time(timer_hook)
# lt.print_total_time()
