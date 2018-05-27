#!/usr/bin/python

from NeuralTuringMachine.NeuralTuringMachine import NeuralTuringMachine
from NeuralTuringMachine.NeuralTuringMachine import NTMCell
import tensorflow as tf
import argparse, os, sys, inspect

		



def build_model(i_placeholder, o_placeholder, header_size, output_size, memory_rows, memory_columns, learning_rate, decay=0.9, momentum=0.9, epsilon=1e-10):


	HEADER_SIZE = header_size
	OUTPUT_SIZE = output_size

	MEMORY_ROWS = memory_rows
	MEMORY_COLUMNS = memory_columns

	LEARNING_RATE = learning_rate
	DECAY = decay
	MOMENTUM = momentum

	EPSILON = epsilon


	global_step = tf.Variable(0, name='global_step', trainable=False)

	ntm = NTMCell(
		osize=OUTPUT_SIZE, 
		hsize=HEADER_SIZE, 
		mrows=MEMORY_ROWS, 
		mcolumns=MEMORY_COLUMNS,
		epsilon=EPSILON
		)


	# End of associative recall config


	init_state = (
		tf.truncated_normal([1, HEADER_SIZE], stddev=0.1, mean=0.5, name='init_read_vector'),
		tf.truncated_normal([1, MEMORY_ROWS * MEMORY_COLUMNS], stddev=1.0, mean=0.5, name='init_memory'),
		tf.zeros([1, MEMORY_COLUMNS], name='init_read_vector'),
		tf.truncated_normal([1, MEMORY_ROWS], stddev=0.2, mean=0.5, name='init_write_weights'), # Must be initialized with positive values
		tf.truncated_normal([1, MEMORY_ROWS], stddev=0.2, mean=0.5, name='init_read_weights') # Must be initialized with positive values
		)

	
	output, _ = tf.nn.dynamic_rnn(ntm, i_placeholder, dtype=tf.float32, initial_state=init_state)

	cross_entropy = tf.multiply(o_placeholder, tf.log(output + EPSILON)) + tf.multiply(1 - o_placeholder, tf.log(1 - output + EPSILON))
	loss = -tf.reduce_mean(cross_entropy)

	optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, decay=DECAY, momentum=MOMENTUM)

	training_op = optimizer.minimize(loss, global_step=global_step)

	prediction_test = tf.round(output)

	return output, cross_entropy, loss, training_op, prediction_test, global_step






def train(i_placeholder, o_placeholder, num_vectors, num_bits, training_op, output, prediction_test, error, folder, global_step, gen_function, summary_merged, plot, restore, batch_size=1, epochs=20000):

	NUM_VECTORS = num_vectors
	NUM_BITS = num_bits
	EPOCHS = epochs
	BATCH_SIZE = batch_size

	last_mse = 10
	# Start of associative recall config
	
	#Create folder if not exists
	if not os.path.exists(PARAMETERS_FOLDER + '/' + folder):
		os.makedirs(PARAMETERS_FOLDER + '/' + folder)

	if not os.path.exists(PARAMETERS_FOLDER + '/' + folder + '/opt'):
		os.makedirs(PARAMETERS_FOLDER + '/' + folder + '/opt')


	if not os.path.exists(PLOTS_FOLDER + '/' + folder):
		os.makedirs(PLOTS_FOLDER + '/' + folder)
		
	if plot:
		train_writer = tf.summary.FileWriter(PLOTS_FOLDER + '/' + args.folder + '/train')
		test_writer = tf.summary.FileWriter(PLOTS_FOLDER + '/' + args.folder + '/test')


	restore_path = tf.train.latest_checkpoint(PARAMETERS_FOLDER + '/' + folder)
	
	if(restore_path and restore):
		print('Restoring model from %s.' % restore_path)
		saver = tf.train.Saver()
		saver.restore(sess, restore_path)
		print('Model restored.')
	else:
		init = tf.global_variables_initializer()	
		saver = tf.train.Saver()
		init.run()

	for epoch in range(1, EPOCHS):
		x_batch, y_batch = gen_function.generate(NUM_VECTORS, NUM_BITS, BATCH_SIZE)
		
		if(plot):
			_, out, prediction, summary = sess.run([training_op, output, prediction_test, summary_merged], feed_dict={i_placeholder: x_batch, o_placeholder: y_batch})
			train_writer.add_summary(summary, global_step.eval())
		else:
			_, out, prediction = sess.run([training_op, output, prediction_test], feed_dict={i_placeholder: x_batch, o_placeholder: y_batch})
			

		if (epoch % 100) == 0:
			if(plot):
				test_writer.add_summary(summary, global_step.eval())

			mse = error.eval(feed_dict={i_placeholder: x_batch, o_placeholder: y_batch})
			
			print('\n---------------------- Epoch %d ----------------------' % epoch)
			print('MSE:' + str(mse))

			print('\nInput: ')
			print(x_batch)
			print('\nNTM output: ')
			print(out)
			print('\nNTM output (rounded): ')
			print(prediction)
			print('\nTarget output: ')
			print(y_batch)

			save_path = saver.save(sess, PARAMETERS_FOLDER + '/' + folder + '/ntm', global_step=global_step)

			if(mse < last_mse):
				save_path = saver.save(sess, PARAMETERS_FOLDER + '/' + folder + '/opt/ntm', global_step=global_step)

			print('\nModel saved in file: %s' % save_path)


	train_writer.flush()
	test_writer.flush()
	
	train_writer.close()
	test_writer.close()


def test(i_placeholder, o_placeholder, prediction_test, num_vectors, num_bits, folder, gen_function, batch_size=1, num_tests=10):

	NUM_VECTORS = num_vectors
	NUM_BITS = num_bits
	NUM_TESTS=num_tests
	BATCH_SIZE = batch_size


	restore_path = tf.train.latest_checkpoint(PARAMETERS_FOLDER + '/' + folder)
	print('Restoring model from %s.' % restore_path)
	saver = tf.train.Saver()
	saver.restore(sess, restore_path)
	print('Model restored.')

	for i in range(0, NUM_TESTS):
		x_test, y_test = gen_function.generate(NUM_VECTORS, NUM_BITS, BATCH_SIZE)
		prediction = sess.run(prediction_test, feed_dict={i_placeholder: x_test, o_placeholder: y_test})


		print('\n---------------------- Test %d ----------------------' % i)
		print('\nInput: ')
		print(x_test)
		print('\nNTM output (rounded): ')
		print(prediction)
		print('\nTarget output: ')
		print(y_test)


def info():
	total_parameters = 0

	for variable in tf.trainable_variables():
		# shape is an array of tf.Dimension
		print('Name:\t\t%s' % (variable.name))
		shape = variable.get_shape()
		print('Shape:\t\t%s' % (shape))
		print('Shape length:\t%d' % len(shape))
		variable_parameters = 1
		idx_shape = 1
		for dim in shape:
			print('Dimension %d:\t%d' % (idx_shape, dim))
			idx_shape += 1
			variable_parameters *= dim.value
		print('Var parameters:\t%d' % (variable_parameters))
		total_parameters += variable_parameters
		print('\n')
	
	print('Total parameters:\t%d' %  (total_parameters))


















parser = argparse.ArgumentParser(description='A Neural Turing Machine.')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--train', action='store', dest='epochs', type=int, help='Use for train the NTM.')
group.add_argument('--info', action='store_true', dest='info', help='Use for get information about NTM.')
group.add_argument('--test', action='store', dest='examples', type=int, help='Use for test the NTM.')


parser.add_argument('--task_module', action='store', dest='task_module', help='Module for lookup task dataset generator.', required=True)
parser.add_argument('--task_class', action='store', dest='task_class', help='Class that will feed test and training information.', required=True)

parser.add_argument('--plot', action='store_true', dest='plot', help='Activates plot graphics.')
parser.add_argument('--restore', action='store_true', dest='restore', help='Restores last checkpoint from folder to continue processing.')

parser.add_argument('--folder', action='store', dest='folder', default='', help='Destination folder to save/recover test/train models.')
parser.add_argument('--vectors', action='store', dest='vectors', default=5, type=int, help='Vectors number to use in task.')
parser.add_argument('--bits', action='store', dest='bits', default=5, type=int, help='Bits on each vector used in task.')

parser.add_argument('--mem_rows', action='store', dest='rows', default=128, type=int, help='Rows in memory.')
parser.add_argument('--mem_cols', action='store', dest='cols', default=50, type=int, help='Columns in memory.')


parser.add_argument('--l_rate', action='store', dest='learning_rate', default=8e-5, type=float, help='Learning rate.')
parser.add_argument('--decay', action='store', dest='decay', default=0.7, type=float, help='Decay.')
parser.add_argument('--momentum', action='store', dest='momentum', default=0.7, type=float, help='Momentum.')


parser.set_defaults(plot=False)
parser.set_defaults(restore=False)


args = parser.parse_args()


PARAMETERS_FOLDER = 'Parameters'
PLOTS_FOLDER = 'Plots'

NUM_VECTORS = args.vectors
NUM_BITS = args.bits

HEADER_SIZE = 100
OUTPUT_SIZE = NUM_BITS

MEMORY_ROWS = args.rows # N
MEMORY_COLUMNS = args.cols # M

LEARNING_RATE = args.learning_rate
DECAY = args.decay
MOMENTUM = args.momentum


module = __import__(args.task_module, fromlist=[args.task_class])
TASK_GENERATOR = getattr(module, args.task_class)
task_generator = TASK_GENERATOR()


sess = tf.InteractiveSession()

X = task_generator.get_input_placeholder(NUM_VECTORS, NUM_BITS)
Y = task_generator.get_output_placeholder(NUM_VECTORS, NUM_BITS)


output, cross_entropy, loss, training_op, prediction_test, global_step = build_model(
																		i_placeholder = X,
																		o_placeholder = Y,
																		header_size=HEADER_SIZE, 
																		output_size=OUTPUT_SIZE, 
																		memory_rows=MEMORY_ROWS, 
																		memory_columns=MEMORY_COLUMNS, 
																		learning_rate=LEARNING_RATE, 
																		decay=DECAY, 
																		momentum=MOMENTUM
																		)
merged = None

if(args.plot):
	# tf.summary.histogram('Error', cross_entropy)
	tf.summary.scalar('Loss', loss)
	merged = tf.summary.merge_all()


if args.epochs:
	train(i_placeholder=X, 
			o_placeholder=Y, 
			num_vectors=NUM_VECTORS, 
			num_bits=NUM_BITS, 
			training_op=training_op, 
			output=output, 
			prediction_test=prediction_test,
			error=loss, 
			epochs=args.epochs,
			folder= args.folder,
			global_step=global_step,
			gen_function=task_generator,
			summary_merged=merged,
			plot=args.plot,
			restore=args.restore
			)

elif args.examples:
	test(i_placeholder=X, 
			o_placeholder=Y, 
			prediction_test=prediction_test, 
			num_vectors=NUM_VECTORS, 
			num_bits=NUM_BITS, 
			num_tests=args.examples,
			folder=args.folder,
			gen_function=task_generator)
	

elif args.info:
	info()