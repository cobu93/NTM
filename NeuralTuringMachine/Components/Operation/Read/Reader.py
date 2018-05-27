from NeuralTuringMachine.Components.Operation.Addressing.ContentAddressing import ContentAddressing
from NeuralTuringMachine.Components.Operation.Addressing.LocationAddressing import LocationAddressing
import tensorflow as tf

class Reader():


	def run_reader(self, memory, r_weights, r_key, r_shift, r_intensity, r_interpolation, r_sharpen, epsilon=1e-6):

		w = ContentAddressing.address(
			key=r_key, 
			intensity=r_intensity, 
			memory=memory,
			epsilon=epsilon
			)

		w = LocationAddressing.address(
			interpolation=r_interpolation, 
			shift=r_shift, 
			sharpen=r_sharpen, 
			weights=r_weights,
			cont_weights=w,
			epsilon=epsilon
			)

		# w=1XN
		read_vec = tf.matmul(w, memory, name="new_read_vect")
		return read_vec, w
