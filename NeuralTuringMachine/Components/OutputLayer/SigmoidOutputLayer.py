
import tensorflow as tf
from NeuralTuringMachine.Components.General.FeedForwardNN import FeedForwardNN

class SigmoidOutputLayer():

	def __init__(self, layers_desc, name='sigmoid_output'):
		self.ffnn = FeedForwardNN(layers_desc, name)

   	'''
   	input:
   		layers_desc = Description of each layer (LayerDescription object type )
   		error_func = Error function, must has this input:output, expected_output 
   		opt_func = Optimization function. Must have next input: error
   	'''
	def run_output_layer(self, inputs):
		return self.ffnn.run_feed_forward_nn(inputs)
				


