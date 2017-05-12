from chainer.links.caffe.protobuf3 import caffe_pb2 as caffe_pb
from chainer.function_hooks import timer
import threading
import time
import sys

_layer_impl_path_dict = {
	'Convolution': 'chainer.functions.connection.convolution_2d.Convolution2DFunction',
	# 'Concat': 'chainer.functions.concat.Concat',
	# 'Dropout': 'chainer.functions.dropout',
}

_v1_to_new_name_dict = {
	3: 'Concat',
	4: 'Convolution',
	5: 'Data',
	6: 'Dropout',
	14: 'InnerProduct',
	15: 'LRN',
	17: 'Pooling',
	18: 'ReLU',
	25: 'Eltwise',
	33: 'Slice',
	20: 'Softmax',
	21: 'SoftmaxWithLoss',
	22: 'Split',
}


class LayerTimer(object):
	def __init__(self, caffemodel):
		self.timer_hooks = []
		net = caffe_pb.NetParameter()
		with open(caffemodel, 'rb') as model_file:
			net.MergeFromString(model_file.read())
		
		if net.layer:
			for layer in net.layer:
				layer_impl_path = _layer_impl_path_dict.get(layer.type)
				if layer_impl_path:
					_setup_timer_by_layer(layer.type, layer_impl_path)
				else:
					print 'unimplemented layer timer: %s' % layer.type
		else: #v1 format
			for layer in net.layers:
				layer_type = _v1_to_new_name_dict[layer.type]
				layer_impl_path = _layer_impl_path_dict.get(layer_type)
				if layer_impl_path:
					self._setup_timer_by_layer(layer.name, layer_impl_path)
				else:
					print 'unimplemented layer timer: %s' % layer.type



	def _setup_timer_by_layer(self, layer_name, layer_path):
		# import layer impl
		layer_path_list = layer_path.split('.')
		module_path = '.'.join(layer_path_list[0:-1])
		function_name = layer_path_list[-1]

		module = __import__(module_path, globals(), locals(), [function_name])

		func = getattr(module, function_name)
		# Initilize a timer_hook
		timer_hook = TimerHook()
		self.timer_hooks.append((layer_name, timer_hook))

		# dirty implementation here:
		func._local_function_hooks = {}
		func._n_local_function_hooks = 1
		func._local_function_hooks[layer_name] = timer_hook

	def print_total_time(self):
		print '================================================'

		for layer, hook in self.timer_hooks:
			print '[{}]:'.format(layer)
			print 'total time: {}'.format(hook.total_time())

			layer_forward_time = 1 if len(hook.call_history) is 0 else len(hook.call_history)

			print 'average time: {}'.format(hook.total_time() / layer_forward_time)

		print '================================================'

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
		sys.stdout.write('\r[{:6.2f}%]'.format(self.progress * 100))
		sys.stdout.flush()





import time

import numpy

from chainer import cuda
from chainer import function


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
