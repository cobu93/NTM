import tensorflow as tf
import numpy as np
import math

class LocationAddressing():

	@staticmethod
	# address(Scalar, 1XN, Scalar, 1XN, 1XN)			content_weights
	def address(interpolation, shift, sharpen, weights, cont_weights, epsilon=1e-6):


		wg = LocationAddressing.interpolation(interpolation, cont_weights, weights) #1xN
		wb = LocationAddressing.convolution(wg, shift) # 1 x N
		w = LocationAddressing.sharpen(wb, sharpen, epsilon)



		return w# 1 X N

	@staticmethod
	def interpolation(interpolation_gate, cont_weights, last_weights):
		return interpolation_gate * cont_weights + (1 - interpolation_gate) * last_weights


	@staticmethod
	def convolution(interpolation_weights, shift_weights):
		batch_size = interpolation_weights.get_shape()[0].value
		in_width = interpolation_weights.get_shape()[1].value
		
		wb = []

		def loop(idx, size):
			if idx < 0: return [0, size + idx]
			if idx >= size : return [0, idx - size]
			else: return [0, idx]
		
		for i in range(batch_size):
			values = tf.slice(interpolation_weights, [i, 0], [1, -1])
			kernel = tf.slice(shift_weights, [i, 0], [1, -1])

			values_size = values.get_shape()[1].value
			kernel_size = kernel.get_shape()[1].value
			kernel_shift = int(math.floor(kernel_size / 2.0))


			translations = []

			for i in range(values_size):
				indices = [loop(i + j, values_size) for j in range(kernel_shift, -kernel_shift, -1)]
				selected_values = tf.gather_nd(values, indices)
				
				translations.append(tf.reduce_sum(selected_values * kernel, 1))   

			wb.append(tf.transpose(tf.stack(translations)))
			
		return tf.reshape(tf.stack(wb), [batch_size, in_width])

	@staticmethod
	def sharpen(conv_weights, sharpen_fact, epsilon=1e-6):

		w = tf.pow(conv_weights, sharpen_fact + epsilon) 
		w_sum = tf.reduce_sum(conv_weights) + epsilon

		w /= w_sum
		
		return w





		
