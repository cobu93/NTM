from NeuralTuringMachine.Components.Operation.Addressing.ContentAddressing import ContentAddressing
from NeuralTuringMachine.Components.Operation.Addressing.LocationAddressing import LocationAddressing
import tensorflow as tf

class Writer():


	def run_writer(self, memory, w_weights, w_key, w_shift, w_intensity, w_interpolation, w_sharpen, w_add, w_erase, epsilon=1e-6):

		w = ContentAddressing.address(
			key=w_key, 
			intensity=w_intensity, 
			memory=memory,
			epsilon=epsilon
			)

		w = LocationAddressing.address(
			interpolation=w_interpolation, 
			shift=w_shift, 
			sharpen=w_sharpen, 
			weights=w_weights,
			cont_weights=w,
			epsilon=epsilon
			)

			# w=1XN
			# e=1XM

		mem_ = tf.multiply(memory, 1 - tf.multiply(tf.transpose(w), w_erase))
		return (mem_ + tf.multiply(tf.transpose(w), w_add)), w			