# A simple multi-layer perceptron neural network implementation
#
# See also:	
#	http://ufldl.stanford.edu/tutorial/index.php/Multi-Layer_Neural_Networks
#	http://www.iro.umontreal.ca/~bengioy/ift6266/H12/html.old/mlp_en.html
#	https://bitbucket.org/HugoLarochelle/mlpython
#
# TBD: allow input of validation set for training
#
# Author: Corey Kereliuk

import numpy as np
import copy
import pdb
from matplotlib import pyplot as plt

class mlp:
	def __init__(self, layers, rng_seed=None):		
		'''
		layers is a list specifying the dimension of each network layer:
			layers = [#n_visible, #n_hidden1, #n_hidden2, ...]
		The final layer is a softmax layer, with one node per class label
		E.g.,
			layers = [20,10,2]		
		implements binary classification with inputs of dimension 20, and 10 units in the first hidden layer
		'''
		if rng_seed:
			np.random.seed(rng_seed)

		self.layers = layers
		self.n_hid  = len(layers) - 1 # number of hidden layers				
		
		self.W      = self.n_hid * [None]
		self.dW     = self.n_hid * [None]
		self.b      = self.n_hid * [None]
		self.db     = self.n_hid * [None]
		
		self.pot    = self.n_hid * [None]
		self.act    = self.n_hid * [None]
		self.act0   =  None
		
		self.dE     = self.n_hid * [None]		

		for k in xrange(self.n_hid):
			self.W[k] = 1e-2 * np.random.standard_normal( (layers[k], layers[k+1]) )
			self.b[k] = 1e-6 * np.random.standard_normal( (1, layers[k+1]) ) #np.zeros( (1,layers[k+1]) )

	def sigmoid(self, x):
		return 1. / (1 + np.exp(-x))

	def sigmoid_derivative(self, x):
		return self.sigmoid(x) * (1 - self.sigmoid(x))

	def softmax(self, x):
		# tbd: add overflow protection...
		if 1:		
			Z = np.sum( np.exp(x), axis=1 )
			return np.exp(x) / Z.reshape( (x.shape[0],1) )
		else:
			xc = np.copy(x)
			xc[:,0] = 0 # unconnect first input
			Z = np.sum( np.exp(xc), axis=1 )
			return np.exp(xc) / Z.reshape( (xc.shape[0],1) )

	def f_prop(self, x):
		n_examples  = x.shape[0]
		self.act0   = np.copy(x) # input activation
		self.pot[0] = np.dot(self.act0, self.W[0]) + np.repeat(self.b[0], n_examples, axis=0) # neuron potential

		# forward propogate input through hidden layers
		for k in xrange(0,self.n_hid-1):						
			self.act[k]   = self.sigmoid( self.pot[k] ) # neuron activation
			self.pot[k+1] = np.dot(self.act[k], self.W[k+1]) + np.repeat(self.b[k+1], n_examples, axis=0) # neuron potential

		# output layer w/ softmax unit for multi-class classification
		self.act[-1] = self.softmax( self.pot[-1] )

		return self.act[-1]

	def b_prop(self, target):
		_, self.dE[-1] = self.cost(target)
		
		# back propogate errors
		for k in xrange(self.n_hid-1,0,-1):
			self.dE[k-1] = np.dot( self.dE[k], self.W[k].T ) * self.sigmoid_derivative( self.pot[k-1] )			

		# calculate parameter derivatives
		n_examples = self.act0.shape[0]
		self.dW[0] = np.dot( self.act0.T, self.dE[0] ) / n_examples
		self.db[0] = np.sum( self.dE[0], axis=0 ) / n_examples

		for k in xrange(1, self.n_hid):
			self.dW[k] = np.dot( self.act[k-1].T, self.dE[k] ) / n_examples
			self.db[k] = np.sum( self.dE[k], axis=0 ) / n_examples

	def cost(self, target):
		output     = np.copy( self.act[-1] )
		n_examples = output.shape[0]

		# cross-entropy cost and its derivative
		c = 0
		dC = output
		for n,k in enumerate(target):
			c += -np.log(output[n, k])
			dC[n,k] -= 1
		c /= n_examples

		return c, dC

	def grad(self, data, target):		
		self.f_prop(data)
		self.b_prop(target)

	def train(self, data, target, batch_size=1, learning_rate=1e-1, epochs=10, momentum=0, weight_decay=0):
		n_batches = data.shape[0] / batch_size

		W_inc = self.n_hid * [None]
		b_inc = self.n_hid * [None]

		for k in xrange(self.n_hid):
			W_inc[k] = np.zeros( self.W[k].shape )
			b_inc[k] = np.zeros( self.b[k].shape )
		
		error  = []
		for epoch in xrange(epochs):			
			
			for n in xrange(n_batches):
				index = np.arange(n * batch_size, (n + 1) * batch_size, dtype='int')
				self.grad(data[index], target[index])								

				for k in xrange(self.n_hid):
					W_inc[k] = momentum * W_inc[k] + learning_rate * (self.dW[k] + weight_decay*self.W[k])
					b_inc[k] = learning_rate * self.db[k] + momentum * b_inc[k]

					self.W[k] -= W_inc[k]
					self.b[k] -= b_inc[k]

				cost, _ = self.cost(target[index])
				error.append(cost)

				#print "epoch %d/%d, [%d/%d], cost=%f" % (epoch+1, epochs, n+1, n_batches, cost)

		return np.sum(error)

	def classify(self, data):
		act = self.f_prop(data)
		return np.argmax( act, axis=1 )
		
def test_gradient():	
	'''
	A script to verify the back propagation implementation by comparison with numerical gradients
	'''
	nnet        = mlp([100,20,10,4])	
	n_examples  = 100
	test_data   = np.random.standard_normal( (n_examples, nnet.layers[0]) )
	test_target = np.random.randint(0, nnet.layers[-1], n_examples)

	# numerical estimate of gradient
	epsilon = 1e-6	

	# Weights	
	dW = copy.deepcopy( nnet.W )
	for k in xrange(nnet.n_hid):
		for i in xrange(nnet.W[k].shape[0]):
			for j in xrange(nnet.W[k].shape[1]):
				
				# positive perturbation
				nnet.W[k][i,j] += epsilon
				nnet.f_prop( test_data )
				dC_plus, _ = nnet.cost( test_target )

				# negative perturbation		
				nnet.W[k][i,j] -= 2*epsilon
				nnet.f_prop( test_data )
				dC_minus, _ = nnet.cost( test_target )

				# numerical gradient
				dW[k][i,j] = (dC_plus - dC_minus) / (2*epsilon)

				nnet.W[k][i,j] += epsilon

	# Biases
	db = copy.deepcopy( nnet.b )	
	for k in xrange(nnet.n_hid):
		for i in xrange(nnet.b[k].shape[1]):
				
				# positive perturbation
				nnet.b[k][0,i] += epsilon
				nnet.f_prop( test_data )
				dC_plus, _ = nnet.cost( test_target )

				# negative perturbation		
				nnet.b[k][0,i] -= 2*epsilon
				nnet.f_prop( test_data )
				dC_minus, _ = nnet.cost( test_target )

				# numerical gradient
				db[k][0,i] = (dC_plus - dC_minus) / (2*epsilon)

				nnet.b[k][0,i] += epsilon

	# back propogation estimate of gradient
	nnet.grad( test_data, test_target )		

	# comparison
	delta_W = 0
	delta_b = 0
	for k in xrange(nnet.n_hid):
		delta_W += np.sum( np.abs(nnet.dW[k] - dW[k]) )
		delta_b += np.sum( np.abs(nnet.db[k] - db[k]) )

 	print 'W: The difference between the numerical and back propagation gradients is (should be 0): %f' % delta_W
 	print 'b: The difference between the numerical and back propagation gradients is (should be 0): %f' % delta_b

def test_mlp():
	'''
	A simple test of the MLP using synthetic data
	'''
	n_classes    = 10
	n_visible    = 100
	nnet         = mlp([n_visible,20,n_classes], rng_seed=1234)		

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
		s2 = 2*s
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

	# train
	batch_size = n_examples/n_classes
	nnet.train(train_data, train_target, batch_size, learning_rate=1e-1, epochs=200, momentum=0.7)
	# plt.plot(nnet.error)

	# test
	test_labels = nnet.classify(test_data)

	print [(i,j) for i,j in zip(test_target, test_labels)]

	# report classification error
	per_error = np.sum(np.abs(test_labels - test_target)>0) / float(n_examples) * 100	
	print 'Error on test set is: %f percent' % per_error

	#pdb.set_trace()

# -------------------------
if __name__ == '__main__':

	# verify gradients
	#test_gradient()	

	# dummy problem
	test_mlp()