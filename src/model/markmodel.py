import numpy as np
import tflearn
import tensorflow as tf

from tflearn.data_utils import to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected, reshape
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

class MarkModel:
	def __init__(self):
		
		self.model = self.__construct_model__()
		self.marks, self.wgts = None, None
	
	def __construct_model__(self):
		tf.reset_default_graph()

		network = input_data(shape=[None, 14, 16, 2], name='input')

		# convolutional, max pool and normalization layer
		network = conv_2d(network, nb_filter=128, filter_size=[3, 16], 
						  padding='valid', activation='relu', 
						  regularizer="L2", weights_init='xavier')
		network = max_pool_2d(network, kernel_size=[12, 1], padding='valid')
		network = local_response_normalization(network)

		# reshape convolution output
		network = reshape(network, [-1, 128])

		# dense layer
		network = fully_connected(network, 1024, activation='relu')
		network = dropout(network, 0.5)

		# dense layer
		network = fully_connected(network, 512, activation='relu')
		network = dropout(network, 0.5)

		# dense layer
		network = fully_connected(network, 64, activation='relu')
		network = dropout(network, 0.5)

		#output layer
		network = fully_connected(network, 14, activation='softmax')

		# loss
		network = regression(network, optimizer='adam', learning_rate=0.001,
							 loss='categorical_crossentropy', name='target')

		# Training
		model = tflearn.DNN(network, tensorboard_verbose=0)
		model.load('./mark_cnn_20170812.tfl')
		return model
	
	def predict_mark(self, mark_wgt):
		# retrieve and reshape marks and wgts from `mark_wgt`
		self.marks, self.wgts = mark_wgt[:, 0].reshape(-1), mark_wgt[:, 1].reshape(-1)
		
		# preprocess model input
		model_input = self.preprocess_mark_row(self.marks, self.wgts)
		
		# reshape model input
		model_input = np.array(model_input).reshape(-1, 14, 16, 2)
		
		# make mark prediction
		predicted_mark = np.array(self.model.predict(model_input))
		
		return predicted_mark.reshape(-1)
	
	def grade_idx(self, mark):
		'''
		0 -> A+
		1 -> A
		2 -> A-
		  .
		  .
		  .
		12 -> F
		13 -> F-
		'''
		grade_ranges = [(150, 90), (89, 85), (84, 80), 
						 (79, 77), (76, 73), (72, 70), 
						 (69, 67), (66, 63), (62, 60), 
						 (59, 57), (56, 53), (52, 50), 
						 (49, 35), (34, 0)]
		mark = np.round(mark*100)
		return sum([i if rng[1] <= mark <= rng[0] else 0 for i, rng in enumerate(grade_ranges)])

	def preprocess_mark_row(self, marks, wgts, shuffle=False):
		output = np.zeros((14, 16, 2)).tolist()
		for i in range(len(marks)):
			mark, wgt = marks[i], wgts[i]
			gi = self.grade_idx(mark)
			output[gi][i][0], output[gi][i][1] = mark, wgt
		if shuffle: 
			output = [sample(letter_grade, len(letter_grade)) for letter_grade in output]
		return output