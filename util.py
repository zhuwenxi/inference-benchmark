from chainer.links.caffe.protobuf3 import caffe_pb2 as caffe_pb
from chainer.function_hooks import timer
from chainer.links.caffe import CaffeFunction
from chainer import cuda
from chainer import function

import collections
import threading
import time
import sys
import numpy

class ProgressBar(object):
	def __init__(self, time):
		self.progress = 0.0
		self.done = False
		self.current_time = 0
		self.time = time
		self.thread = threading.Thread(target=self._update_progress)

	def start(self):
		self.thread.start()

	def end(self):
		self.done = True

	def _update_progress(self):
		while True:
			if self.done is True:
				self.progress = 1
				self.show_progress()
				break
			else:
				time.sleep(1)
				self.progress += 1.0 / self.time
				if self.progress >= 1:
					self.progress = 0.99
				self.show_progress()

	def show_progress(self):
		sys.stderr.write('\r[{:6.2f}%]'.format(self.progress * 100))
		sys.stderr.flush()

class TimerHook(function.FunctionHook):
	"""Function hook for measuring elapsed time of functions.

	Attributes:
		call_history: List of measurement results. It consists of pairs of
			the function that calls this hook and the elapsed time
			the function consumes.
	"""

	name = 'TimerHook'

	def __init__(self):
		self.call_history = []

	def _preprocess(self):
		if self.xp == numpy:
			self.start = time.time()
		else:
			self.start = cuda.Event()
			self.stop = cuda.Event()
			self.start.record()

	def forward_preprocess(self, function, in_data):
		self.xp = cuda.get_array_module(*in_data)
		self._preprocess()

	def backward_preprocess(self, function, in_data, out_grad):
		self.xp = cuda.get_array_module(*(in_data + out_grad))
		self._preprocess()

	def _postprocess(self, function):
		if self.xp == numpy:
			self.stop = time.time()
			elapsed_time = self.stop - self.start
		else:
			self.stop.record()
			self.stop.synchronize()
			# Note that `get_elapsed_time` returns result in milliseconds
			elapsed_time = cuda.cupy.cuda.get_elapsed_time(
				self.start, self.stop) / 1000
		self.call_history.append((function.label, elapsed_time))

	def forward_postprocess(self, function, in_data):
		xp = cuda.get_array_module(*in_data)
		assert xp == self.xp
		self._postprocess(function)

	def backward_postprocess(self, function, in_data, out_grad):
		xp = cuda.get_array_module(*(in_data + out_grad))
		assert xp == self.xp
		self._postprocess(function)

	def total_time(self):
		"""Returns total elapsed time in seconds."""
		return sum(t for (_, t) in self.call_history)

	def print_layer_time(self):
		layer_time_dict = {}
		for func, time in self.call_history:
		# func_name = func.label
			func_name = func
			if layer_time_dict.get(func_name) is None:
				layer_time_dict[func_name] = {'time': time, 'number': 1}
			else:
				layer_time_dict[func_name]['time'] += time
				layer_time_dict[func_name]['number'] += 1

		print '================================================'
		keys = layer_time_dict.keys()
		keys.sort()
		for name in keys:
			record = layer_time_dict[name]
			print '[{}]:'.format(name)
			print 'total time: {} ms'.format(record['time'] * 1000)
			print 'average time: {} ms\n'.format(float(record['time']) * 1000/ record['number'])
		print '================================================'

# A sub-class of CaffeFunction which enables layer-by-layer time hooks 
class CaffeFunctionImpl(CaffeFunction):
	def __init__(self, model_path, timer_hook):
		super(CaffeFunctionImpl, self).__init__(model_path)
		self.timer_hook = timer_hook
	def __call__(self, inputs, outputs, disable=(), train=True):
		self.train = train
		variables = dict(inputs)
		for func_name, bottom, top in self.layers:
			if (func_name in disable or
				func_name not in self.forwards or
					any(blob not in variables for blob in bottom)):
				continue
			with self.timer_hook:
				func = self.forwards[func_name]
				input_vars = tuple(variables[blob] for blob in bottom)
				output_vars = func(*input_vars)

			self.timer_hook.call_history[-1] = (func_name, self.timer_hook.call_history[-1][1])
			if not isinstance(output_vars, collections.Iterable):
				output_vars = output_vars,
			for var, name in zip(output_vars, top):
				variables[name] = var

		self.variables = variables
		return tuple(variables[blob] for blob in outputs)

CaffeFunction = CaffeFunctionImpl