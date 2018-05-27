"""
Description: A general structure of Neural Turing Machine
Author: Uriel Corona Bermudez

"""

from Components.General.Layer import *
from Components.Controller.FeedForwardController import FeedForwardController
from Components.OutputLayer.SigmoidOutputLayer import SigmoidOutputLayer
from Components.Operation.Read.ReadHeader import ReadHeader
from Components.Operation.Write.WriteHeader import WriteHeader
from Components.Operation.Read.Reader import Reader
from Components.Operation.Write.Writer import Writer
from Components.Operation.Memory.Memory import Memory
import tensorflow as tf
from tensorflow.python.util import nest



class NeuralTuringMachine():


	SCALAR_SIZE = 1

	def __init__(self, osize, hsize, mrows, mcolumns, epsilon=1e-6):

		self.osize = osize
		self.hsize = hsize
		self.mrows = mrows
		self.mcolumns = mcolumns
		self.epsilon = epsilon

		self.controller = FeedForwardController(
			layers_desc = [
				InputLayerDescription('input_header'), 
				OutputLayerDescription(self.hsize, 'header', tf.nn.tanh)
			],
			name='controller'
			)

		self.output_layer = SigmoidOutputLayer(
			layers_desc = [
				InputLayerDescription('input_y'), 
				OutputLayerDescription(self.osize, 'output', tf.nn.sigmoid, has_bias=False)
			],
			name='sigmoid_output'
			)

		self.read_header = ReadHeader(				 
			key_layers_desc=[
				InputLayerDescription('input_key'), 
				OutputLayerDescription(self.mcolumns, 'key', tf.nn.tanh)
			], 

			intensity_layers_desc=[
				InputLayerDescription('input_intensity'), 
				OutputLayerDescription(self.SCALAR_SIZE, 'intensity', tf.nn.softplus)
			],  

			interpolation_layers_desc=[
				InputLayerDescription('input_interpolation'), 
				OutputLayerDescription(self.SCALAR_SIZE, 'interpolation', tf.nn.sigmoid)
			], 

			shift_layers_desc=[
				InputLayerDescription('input_shift'), 
				OutputLayerDescription(self.mrows, 'shift', tf.nn.softmax)
			], 

			sharpen_layers_desc=[
				InputLayerDescription('input_sharpen'), 
				OutputLayerDescription(self.SCALAR_SIZE, 'sharpen', tf.nn.relu)
			], 

			name='read_header'

			)

		self.write_header = WriteHeader(
			key_layers_desc=[
				InputLayerDescription('input_key'), 
				OutputLayerDescription(self.mcolumns, 'key', tf.nn.tanh)
			], 

			intensity_layers_desc=[
				InputLayerDescription('input_intensity'), 
				OutputLayerDescription(self.SCALAR_SIZE, 'intensity', tf.nn.softplus)
			],  

			interpolation_layers_desc=[
				InputLayerDescription('input_interpolation'), 
				OutputLayerDescription(self.SCALAR_SIZE, 'interpolation', tf.nn.sigmoid)
			], 

			shift_layers_desc=[
				InputLayerDescription('input_shift'), 
				OutputLayerDescription(self.mrows, 'shift', tf.nn.softmax)
			], 

			sharpen_layers_desc=[
				InputLayerDescription('input_sharpen'), 
				OutputLayerDescription(self.SCALAR_SIZE, 'sharpen', tf.nn.relu)
			], 

			erase_layers_desc=[
				InputLayerDescription('input_erase'), 
				OutputLayerDescription(self.mcolumns, 'erase', tf.nn.sigmoid)
			], 

			add_layers_desc=[
				InputLayerDescription('input_add'), 
				OutputLayerDescription(self.mcolumns, 'add', tf.nn.tanh)
			],

			name='write_header'


			)

		self.reader = Reader()
		self.writer = Writer()



	def run_ntm(self, inputs, state):

		print('running...')
		header, memory, read_vec, write_weights, read_weights = state



		memory = tf.reshape(memory, [self.mrows, self.mcolumns], 'reshape_memory')

		n_header = self.controller.run_controller([inputs, read_vec])
		
		
		y = self.output_layer.run_output_layer([n_header])

		r_key, r_shift, r_intensity, r_interpolation, r_sharpen = self.read_header.run_header([n_header])
		w_key, w_shift, w_intensity, w_interpolation, w_sharpen, w_add, w_erase = self.write_header.run_header([n_header])

		n_read_vec, n_read_weights = self.reader.run_reader(memory, read_weights, r_key, r_shift, r_intensity, r_interpolation, r_sharpen, self.epsilon)
		n_memory, n_write_weights = self.writer.run_writer(memory, write_weights, w_key, w_shift, w_intensity, w_interpolation, w_sharpen, w_add, w_erase, self.epsilon)
		
		n_memory = tf.reshape(n_memory, [1, self.mrows * self.mcolumns])

		return y, (n_header, n_memory, n_read_vec, n_write_weights, n_read_weights)


class NTMCell(tf.contrib.rnn.RNNCell):
 

	def __init__(self, osize, hsize, mrows, mcolumns, epsilon=1e-6):
		tf.contrib.rnn.RNNCell.__init__(self)
		self.ntm = NeuralTuringMachine(osize=osize, hsize=hsize, mrows=mrows, mcolumns=mcolumns, epsilon=epsilon)

       
	@property
	def state_size(self):
		return (self.ntm.hsize, self.ntm.mrows * self.ntm.mcolumns, self.ntm.mcolumns, self.ntm.mrows, self.ntm.mrows)
            
	@property
	def output_size(self):
		return self.ntm.osize

	# def call(self, inputs, state):
	# 	return self.run_ntm(inputs, state)

	def __call__(self, inputs, state):
		#output, n_state = self.run_ntm(inputs, state)
		return self.ntm.run_ntm(inputs, state)