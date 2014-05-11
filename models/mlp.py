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
	def __init__(self, layers, act_type='sigmoid', rng_seed=None):		
		'''
		layers is a list specifying the dimension of each network layer:
			layers = [#n_visible, #n_hidden1, #n_hidden2, ...]
		The final layer is a softmax layer, with one node per class label
		E.g.,
			layers = [20,10,2]		
		implements binary classification with inputs of dimension 20, and 10 units in the first hidden layer
		'''
		self.act_types = {'sigmoid':0, 'tanh':1, 'relu':2}

		if rng_seed:
			np.random.seed(rng_seed)

		if not act_type in self.act_types:
			raise Exception("Unknown activation type.  Currently recognized options are %s" % self.act_types)
		else:
			self.act_type = act_type

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
		
		self.W_inc  = self.n_hid * [None]
		self.b_inc  = self.n_hid * [None]

		self.weight_decay = 0	
			
		for k in xrange(self.n_hid):
			self.W[k]     = 1e-1 * np.random.standard_normal( (layers[k], layers[k+1]) )
			self.b[k]     = np.zeros( (1,layers[k+1]) )

			self.W_inc[k] = np.zeros( self.W[k].shape )
			self.b_inc[k] = np.zeros( self.b[k].shape )

	def activation_fn(self, x):
		if self.act_type=='sigmoid':
			return 1. / (1 + np.exp(-x))
		elif self.act_type=='tanh':
			return np.tanh(x)
		elif self.act_type=='relu':
			return x * (x>0) # ReLu

	def activation_derivative(self, x):
		if self.act_type=='sigmoid':
			return self.activation_fn(x) * (1 - self.activation_fn(x))
		elif self.act_type=='tanh':
			return 1 - np.tanh(x)**2
		elif self.act_type=='relu':
			return x>0 # ReLu

	def softmax(self, x):
		# tbd: add overflow protection...
		if 1:
			expx = np.exp(x)
			Z = np.sum(expx, axis=1 )
			return expx / Z.reshape( (x.shape[0],1) )
		else:
			xc = np.copy(x)
			xc[:,0] = 0 # unconnect first input
			Z = np.sum( np.exp(xc), axis=1 )
			return np.exp(xc) / Z.reshape( (xc.shape[0],1) )

	def f_prop(self, x, mask=None):
		if not mask:
			mask = np.ones(self.n_hid)

		n_examples  = x.shape[0]
		self.act0   = np.copy(x) * mask[0] # input activation
		self.pot[0] = np.dot(self.act0, self.W[0]) + np.repeat(self.b[0], n_examples, axis=0) # neuron potential

		# forward propogate input through hidden layers
		for k in xrange(0,self.n_hid-1):						
			self.act[k]   = self.activation_fn( self.pot[k] ) * mask[k+1] # neuron activation
			self.pot[k+1] = np.dot(self.act[k], self.W[k+1]) + np.repeat(self.b[k+1], n_examples, axis=0) # neuron potential

		# output layer w/ softmax unit for multi-class classification
		self.act[-1] = self.softmax( self.pot[-1] )

		return self.act[-1]

	def b_prop(self, target):
		_, self.dE[-1] = self.cost(target)
		
		# back propogate errors
		for k in xrange(self.n_hid-1,0,-1):
			self.dE[k-1] = np.dot( self.dE[k], self.W[k].T) * self.activation_derivative( self.pot[k-1] )			

		# calculate parameter derivatives
		self.dW[0] = np.dot( self.act0.T, self.dE[0] )
		self.db[0] = np.sum( self.dE[0], axis=0 )

		for k in xrange(1, self.n_hid):
			self.dW[k] = np.dot( self.act[k-1].T, self.dE[k] )
			self.db[k] = np.sum( self.dE[k], axis=0 )

	def f_prop_drop(self, x, drop_rates=None):
		if not drop_rates:
			drop_rates = np.zeros(self.n_hid)

		n_examples  = x.shape[0]
		self.act0   = np.copy(x) * (1-drop_rates[0]) # input activation
		self.pot[0] = np.dot(self.act0, self.W[0]) + np.repeat(self.b[0], n_examples, axis=0) # neuron potential

		# forward propogate input through hidden layers
		for k in xrange(0,self.n_hid-1):						
			self.act[k]   = self.activation_fn( self.pot[k] ) * (1-drop_rates[k+1]) # neuron activation
			self.pot[k+1] = np.dot(self.act[k], self.W[k+1]) + np.repeat(self.b[k+1], n_examples, axis=0) # neuron potential

		# output layer w/ softmax unit for multi-class classification
		self.act[-1] = self.softmax( self.pot[-1] )

		return self.act[-1]

	def cost(self, target):
		output     = np.copy( self.act[-1] )
		n_examples = output.shape[0]

		# cross-entropy cost and its derivative
		c = 0
		dC = output
		for n,k in enumerate(target):
			c += -np.log(output[n, k])
			dC[n,k] -= 1
		
		cost = c/n_examples + self.weight_decay * self.l2_penalty()
		return cost, dC/n_examples

	def l2_penalty(self):
		penalty = 0
		for k in xrange(self.n_hid):
			penalty += np.sum(self.W[k]**2)
			penalty += np.sum(self.b[k]**2)
		return 0.5 * penalty

	def l1_penalty(self):
		penalty = 0
		for k in xrange(self.n_hid):
			penalty += np.sum(np.abs(self.W[k]))
		return 0.5 * penalty

	def grad(self, data, target, drop_rates=None):		
		if drop_rates:
			mask = self.n_hid * [None]
			n_examples = data.shape[0]
			for k in xrange(self.n_hid):
				mask[k] = np.random.binomial( 1, 1-drop_rates[k], (n_examples, self.layers[k]) )
		else:
			mask = None

		self.f_prop(data, mask)
		self.b_prop(target)

		for k in xrange(self.n_hid):
			self.dW[k] += self.weight_decay * self.W[k]
			#self.db[k] += self.weight_decay * self.b[k].flatten()
		#return self.db, self.dW

	def train(self, data, target, batch_size=1, learning_rate=1e-1, epochs=10, momentum=0, weight_decay=0, drop_rates=None, verbose=False):
		n_batches = data.shape[0] / batch_size
		self.weight_decay = weight_decay

		error  = []
		for epoch in xrange(epochs):
			epoch_cost = 0				
			for n in xrange(n_batches):
				index = np.arange(n * batch_size, (n + 1) * batch_size, dtype='int')			
				self.grad(data[index], target[index], drop_rates)				

				for k in xrange(self.n_hid):
					self.W_inc[k] *= momentum
					self.b_inc[k] *= momentum

					self.W_inc[k] += (1-momentum) * learning_rate * (self.dW[k])# + weight_decay * self.W[k])
					self.b_inc[k] += (1-momentum) * learning_rate * (self.db[k])# + weight_decay * self.b[k] / float(n_batches))

					self.W[k]     -= self.W_inc[k]
					self.b[k]     -= self.b_inc[k]

				cost, _ = self.cost(target[index])
				epoch_cost += cost
			
			error.append(epoch_cost)

			if verbose:
				print "epoch %d/%d, cost=%f" % (epoch+1, epochs, epoch_cost / n_batches)

		return np.sum(error)

	def classify(self, data, drop_rates=None):
		act = self.f_prop_drop(data, drop_rates)
		return np.argmax( act, axis=1 )
	
def test_gradient():	
	'''
	A script to verify the back propagation implementation by comparison with numerical gradients
	'''
	nnet        = mlp([100,20,10,4])	
	n_examples  = 100
	test_data   = np.random.standard_normal( (n_examples, nnet.layers[0]) )
	test_target = np.random.randint(0, nnet.layers[-1], n_examples)
	nnet.weight_decay = 1e-1

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

def test_xor():
	n_classes = 2
	n_visible = 2
	nnet = mlp([n_visible, 2, n_classes], rng_seed=None)

	n_examples = 1000

	train_data = np.random.randint(0,n_classes,(n_examples,n_visible))
	train_target = np.array(np.sum(train_data, axis=1)==1, dtype='int')

	test_data = np.random.randint(0,n_classes,(n_examples,n_visible))
	test_target = np.array(np.sum(test_data, axis=1)==1, dtype='int')

	nnet.train(train_data, train_target, batch_size=10, learning_rate=1e-1, epochs=200, momentum=0.5, verbose=True)

	test_labels = nnet.classify(test_data)
	percent_error = np.sum(np.abs(test_labels - test_target)>0) / float(n_examples) * 100	
	print 'Error on test set is: %f percent' % percent_error

def test_xor_dropout():
	n_classes = 2
	n_visible = 2
	nnet = mlp([n_visible, 50, n_classes], act_type='relu', rng_seed=None)

	n_examples = 1000

	train_data = np.random.randint(0,n_classes,(n_examples,n_visible))
	train_target = np.array(np.sum(train_data, axis=1)==1, dtype='int')

	test_data = np.random.randint(0,n_classes,(n_examples,n_visible))
	test_target = np.array(np.sum(test_data, axis=1)==1, dtype='int')

	drop_rates=[0.5,0.5]
	nnet.train(train_data, train_target, batch_size=10, learning_rate=1e-1, epochs=200, momentum=0.5, drop_rates=drop_rates, verbose=True)

	test_labels = nnet.classify(test_data, drop_rates)
	percent_error = np.sum(np.abs(test_labels - test_target)>0) / float(n_examples) * 100	
	print 'Error on test set is: %f percent' % percent_error

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
	percent_error = np.sum(np.abs(test_labels - test_target)>0) / float(n_examples) * 100	
	print 'Error on test set is: %f percent' % percent_error

	#pdb.set_trace()

def test_mlp_dropout():
	'''
	A simple test of the MLP using synthetic data
	'''
	n_classes    = 2
	n_visible    = 50
	nnet         = mlp([n_visible,500,500,500,n_classes], act_type='relu', rng_seed=1234)		

	# generate some synthetic train/test data
	# np.random.seed(1234)
	n_examples   = n_classes*100

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
	drop_rates = [0.,0.25,0.25,0.25] 
	nnet.train(train_data, train_target, batch_size, learning_rate=1e-2, epochs=100, momentum=0.5, drop_rates=drop_rates, verbose=True)
	# plt.plot(nnet.error)

	# test
	test_labels = nnet.classify(test_data, drop_rates=drop_rates)

	print [(i,j) for i,j in zip(test_target, test_labels)]

	# report classification error
	percent_error = np.sum(np.abs(test_labels - test_target)>0) / float(n_examples) * 100	
	print 'Error on test set is: %f percent' % percent_error

# -------------------------
if __name__ == '__main__':

	# verify gradients
	test_gradient()	

	# dummy problem
	#test_mlp()
	#test_mlp_dropout()
	#test_xor()
	#test_xor_dropout()