import tensorflow as tf

class Memory:
    
    def __init__(self, rows, columns):
    	self.memory = tf.zeros([rows, columns])



