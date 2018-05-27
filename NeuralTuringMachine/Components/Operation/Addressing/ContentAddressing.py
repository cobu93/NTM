import tensorflow as tf

class ContentAddressing():

	@staticmethod
	def address(key, intensity, memory, epsilon=1e-6):
		
		# Cosine similitude
		key_mem = tf.matmul(key, tf.transpose(memory)) # (1XM) (NXM)T = (1XM)(MXN) = 1XN
		norm_memory_row = tf.sqrt( tf.reduce_sum( tf.multiply(memory, memory) , 1) ) # Nx1
		norm_key = tf.sqrt( tf.reduce_sum( tf.multiply(key, key) , 1) )# Scalar

		norms_sum = tf.transpose( tf.multiply ( norm_key , norm_memory_row )) + epsilon # 1XN

		similitude = key_mem / norms_sum # 1XN

		return tf.nn.softmax(intensity * similitude) #1XN





		
