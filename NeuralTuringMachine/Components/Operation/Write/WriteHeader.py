from NeuralTuringMachine.Components.General.FeedForwardNN import FeedForwardNN

class WriteHeader():

	def __init__(self, key_layers_desc, intensity_layers_desc, interpolation_layers_desc, shift_layers_desc, sharpen_layers_desc, erase_layers_desc, add_layers_desc, name='write_header'):
		self.key_nn = FeedForwardNN(key_layers_desc, name)
		self.shift_nn = FeedForwardNN(shift_layers_desc, name)
		self.intensity_nn = FeedForwardNN(intensity_layers_desc, name)
		self.interpolation_nn = FeedForwardNN(interpolation_layers_desc, name)
		self.sharpen_nn = FeedForwardNN(sharpen_layers_desc, name)
		self.add_nn = FeedForwardNN(add_layers_desc, name)
		self.erase_nn = FeedForwardNN(erase_layers_desc, name)



	def run_header(self, inputs):

		key = self.key_nn.run_feed_forward_nn(inputs)
		shift = self.shift_nn.run_feed_forward_nn(inputs)
		intensity = self.intensity_nn.run_feed_forward_nn(inputs)
		interpolation = self.interpolation_nn.run_feed_forward_nn(inputs)
		sharpen = self.sharpen_nn.run_feed_forward_nn(inputs)
		add = self.add_nn.run_feed_forward_nn(inputs)
		erase = self.erase_nn.run_feed_forward_nn(inputs)


		return key, shift, intensity, interpolation, sharpen, add, erase

