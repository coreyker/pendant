# A simple deep neural network created by initializing a multi-layer perceptron (MLP)
# using several restricted boltzman machines (RBM).
#
# There are two variants of fine-tuning implemented here:
# 	1. 'deep': The weights of the entire network are fine-tuned using back propagation
#	2. 'shallow': Only the weights of the deepest layer are fine-tuned using back propagation
#
# Author: Corey Kereliuk

import numpy as np
import rbm
import mlp
import pdb

class dnn:
	def __init__(self, layers, rng_seed=None):
		'''
		layers is a list specifying the dimension of each network layer:
			layers = [#n_visible, #n_hidden1, #n_hidden2, ...]
		The final layer is a softmax layer, with one node per class label
		E.g.,
			layers = [20,10,2]		
		implements binary classification with inputs of dimension 20, and 10 units in the first hidden layer

		The DNN is trained by first calling pre_train() to perform unsupervised pre-training, 
		and then fine_tune() to perform discriminative (supervised) fine tuning
		'''
		self.layers = layers
		self.n_hid  = len(layers) - 1 		
		self.mlp    = mlp.mlp(layers, rng_seed) # initialize multi-layer perceptron
		self.rbm    = (self.n_hid - 1) * [None]
		
		# initialize RBMs
		for k in xrange(self.n_hid - 1): # skip last layer which uses softmax instead of binary units
			self.rbm[k] = rbm.rbm(layers[k], layers[k+1], input_type='binary')

	def pre_train(self, data, batch_size=1, learning_rate=1e-1, epochs=10, cd_steps=1, momentum=0, weight_decay=0):
		'''
		Greedy layer-wise pre-training using RBMs
		'''		
		vis = np.copy(data)
		for k in xrange(self.n_hid - 1):
			# train rbm
			print 'Pre-training layer %d of %d' % (k+1, self.n_hid)
			print '----------------------------------'
			self.rbm[k].train(vis, batch_size, learning_rate, epochs, cd_steps, momentum, weight_decay)

            # copy new weights into mlp as they are learned
			self.mlp.W[k] = np.copy(self.rbm[k].W)
			self.mlp.b[k] = np.copy(self.rbm[k].h_bias)

            # hidden layer at level k becomes the visible layer at level k+1
			vis = self.rbm[k].propup(vis)

	def fine_tune(self, data, target, depth='deep', batch_size=1, learning_rate=0.1, epochs=10, momentum=0):
		'''
		Discriminative fine tuning of MLP weights. In most cases pre_train() should be called first
		'''		
		if depth=='deep': # fine-tune all layers			
			self.mlp.train(data, target, batch_size, learning_rate, epochs, momentum)

		elif depth=='shallow': # fine-tune deepest layer only
			perceptron = mlp.mlp([self.layers[-2], self.layers[-1]])
			perceptron.train( self.calc_rep(data), target, batch_size, learning_rate, epochs, momentum )

			# copy weights into last layer
			self.mlp.W[-1] = np.copy(perceptron.W[-1])
			self.mlp.b[-1] = np.copy(perceptron.b[-1])	
		else:
			raise Exception("Unknown depth.  Currently recognized options are 'deep' or 'shallow'")						

	def calc_rep(self, data):
		'''
		Forward propagate input to deepest layer
		'''
		for k in xrange(self.n_hid - 1):
			data = self.rbm[k].propup(data)
		return data

	def classify(self, data):
		return self.mlp.classify(data)

def test_dnn():
	'''
	A simple test of the DNN using synthetic data
	'''
	n_classes    = 10
	n_visible    = 100
	nnet         = dnn([n_visible,20,20,20,n_classes])#, rng_seed=1234)		

	# generate some synthetic train/test data
	# np.random.seed(1234)
	n_examples   = n_classes*10

	train_data   = np.zeros( (n_examples, n_visible) )
	train_target = np.zeros( n_examples, dtype='int' )
	
	test_data    = np.zeros( (n_examples, n_visible) )
	test_target  = np.zeros( n_examples, dtype='int' )

	for c in xrange(n_classes):		
		# choose random basis vector		
		s  = 0.1
		s2 = 3*s
		vec = s + (1-2*s)*np.random.rand(1,n_visible)

		for n in xrange(n_examples/n_classes):
			train_data[c + n*n_classes, :] = vec + s2*np.random.standard_normal(n_visible)
			train_target[c + n*n_classes]  = c
			
			test_data[c + n*n_classes, :]  = vec + s2*np.random.standard_normal(n_visible)
			test_target[c + n*n_classes]   = c

	# shuffle training data
	perm         = np.random.permutation(n_examples)
	train_data   = train_data[perm]
	train_target = train_target[perm]	

	# pre-train
	batch_size = n_examples/n_classes	
	nnet.pre_train(train_data, batch_size, learning_rate=1e-1, epochs=10, momentum=0.8) # set epochs=0 to see effect of pre-training
	
	# fine tune
	depth      = 'deep'
	nnet.fine_tune(train_data, train_target, depth, batch_size, learning_rate=1e-1, epochs=10, momentum=0.8)	

	# test
	test_labels = nnet.classify(test_data)

	print [(i,j) for i,j in zip(test_target, test_labels)]

	# report classification error
	per_error = np.sum(np.abs(test_labels - test_target)>0) / float(n_examples) * 100	
	print 'Error on test set is: %f percent' % per_error
	

# ------------------------
if __name__ == '__main__':
	test_dnn()