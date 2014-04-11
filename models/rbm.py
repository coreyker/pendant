# A simple RBM implmentation
#
# See also: 
#	http://deeplearning.net/tutorial/code/rbm.py
#	http://www.cs.toronto.edu/~hinton/code/rbm.m
#
# NB!: persistent-cd and rlu units are not fully tested
# TBD: add sparsity-based regularization (sparsity of weights / sparsity of activations)
#
# Author: Corey Kereliuk

import pdb
import numpy as np
from matplotlib import pyplot as plt

class rbm:
	def __init__(self, n_visible=10, n_hidden=10, input_type='binary', output_type='binary', input_mean=None, rng_seed=None, persistent=False):				

		if rng_seed:
			np.random.seed(rng_seed)

        # model parameters
		self.n_visible    = n_visible
		self.n_hidden     = n_hidden		
		if not (input_type=='binary' or input_type=='gaussian' or input_type=='rlu'):
			raise Exception("Unknown input type.  Currently recognized options are 'binary' or 'gaussian'")			
		if not (output_type=='binary' or output_type=='softmax'):
			raise Exception("Unknown output type.  Currently recognized options are 'binary' or 'softmax'")			
		self.input_type   = input_type
		self.output_type  = output_type
		self.persistent   = persistent
		self.model_sample = None

		if not np.any(input_mean):
			self.input_mean     = np.zeros((1,n_visible))
		else:
			self.input_mean     = input_mean

		self.h_bias       = np.zeros( (1, n_hidden) )
		self.v_bias       = np.zeros( (1, n_visible) )
		self.W            = 1e-2 * np.random.randn( n_visible, n_hidden )

		self.W_inc        = np.zeros( (n_visible, n_hidden) )
		self.h_inc        = np.zeros( (1, n_hidden) )
		self.v_inc        = np.zeros( (1, n_visible) )

	def sigmoid(self, x):
		return 1. / (1 + np.exp(-x))

	def softmax(self, x):
		expx = np.exp(x)
		Z = np.sum(expx, axis=1 )
		return expx / Z.reshape( (x.shape[0],1) )

	def propup(self, vis):
		n_examples = vis.shape[0]
		pot = np.dot(vis, self.W) + np.repeat(self.h_bias, n_examples, axis=0)
		
		if self.output_type=='binary':
			return self.sigmoid( pot )
		elif self.output_type=='softmax':
			return self.softmax( pot )

	def propdown(self, hid):
		n_examples = hid.shape[0]
		pot = np.dot(hid, self.W.T) + np.repeat(self.v_bias, n_examples, axis=0)
		
		if self.input_type=='binary':
			return self.sigmoid(pot)
		elif self.input_type=='gaussian':
			return pot
		elif self.input_type=='rlu':
			return pot

	def sample_h_given_v(self, sample):	
		n_examples = sample.shape[0]
		pot = self.propup(sample)

		if self.output_type=='binary':
			return np.random.binomial( n=1, p=pot, size=(n_examples, self.n_hidden) )
		elif self.output_type=='softmax':
			hid = np.zeros(pot.shape)
			ind = np.argmax(pot, axis=1)
			for i,row in zip(ind,hid):
				row[i]=1
			return hid

	def sample_v_given_h(self, sample):
		n_examples = sample.shape[0]
		u = self.propdown( sample )

		if self.input_type=='binary':
			if 1:
				return u - np.repeat(self.input_mean, n_examples, axis=0) 
			else:
				return np.random.binomial( n=1, p=u ) - np.repeat(self.input_mean, n_examples, axis=0)
		elif self.input_type=='gaussian':
			return u + np.random.standard_normal( u.shape )
		elif self.input_type=='rlu':
			pot = u + np.random.standard_normal( u.shape )
			return pot * (pot>0)

	def gibbs_vhv(self, sample):
		return self.sample_v_given_h( self.sample_h_given_v( sample ) )
	
	def grad(self, data_sample, cd_steps):
		"""
		data_sample: A matrix with self.n_visible columns and one row per data sample
		"""
		batch_size = data_sample.shape[0]				

		# positive phase
		# --------------				
		hidden_prob = self.propup( data_sample )
		v_pos       = np.sum( data_sample, axis=0 )
		h_pos       = np.sum( hidden_prob, axis=0 )
		W_pos       = np.dot( data_sample.T, hidden_prob )

		# negative phase
		# --------------
		if self.persistent is False or self.model_sample is None:
			self.model_sample = np.copy( data_sample )

		for steps in xrange(cd_steps): # get new samples from model
			self.model_sample = self.gibbs_vhv( self.model_sample )
		
		hidden_prob = self.propup( self.model_sample )
		v_neg       = np.sum( self.model_sample, axis=0 )
		h_neg       = np.sum( hidden_prob, axis=0 )
		W_neg       = np.dot( self.model_sample.T, hidden_prob )

		# average gradient of batch
		v_delta = (v_pos - v_neg) / float(batch_size)
		h_delta = (h_pos - h_neg) / float(batch_size)
		W_delta = (W_pos - W_neg) / float(batch_size)

		Err = np.sum((data_sample - self.model_sample)**2)

		return W_delta, h_delta, v_delta, Err

	def train(self, data, batch_size=1, learning_rate=0.1, epochs=10, \
		cd_steps=1, momentum=0, weight_decay=0, verbose=False):

		"""
		Train RBM with stocastic gradient descent
		"""
		n_batches = data.shape[0] / batch_size

		Err = []
		for epoch in xrange(epochs):

			Err.append(0)
			for index in xrange(n_batches):				
				batch = data[index * batch_size : (index + 1) * batch_size]
				
				W_grad, h_grad, v_grad, err = self.grad(batch, cd_steps)
				Err[-1] += err

				# is this use of momentum correct? should there be (1-momentum) attached to the second term?
				self.W_inc  = self.W_inc * momentum + learning_rate * (W_grad - weight_decay*self.W)
				self.h_inc  = self.h_inc * momentum + learning_rate * h_grad
				self.v_inc  = self.v_inc * momentum + learning_rate * v_grad								

				self.W      += self.W_inc
				self.h_bias += self.h_inc
				self.v_bias += self.v_inc

				if verbose:
					print "Epoch %d/%d, [%d/%d], error=%f" % (epoch+1, epochs, index+1, n_batches, err)		
		
		return np.sum(Err)/epochs #return Err
		

def test_rbm():
	input_type = 'gaussian'
	n_visible = 2
	n_hidden  = 3

	my_rbm = rbm(n_visible, n_hidden, input_type, rng_seed=1234)

	# make training data
	n_samples = 200 * n_hidden
	data      = np.zeros( (n_samples, n_visible) )
	vec       = np.zeros((n_hidden,n_visible))

	# synthetic data	
	for n in xrange(n_hidden):
		# choose basis vector
		if input_type == 'gaussian':
			vec[n,:] = 10 * np.random.standard_normal(n_visible)
		elif input_type == 'rlu':
			vec[n,:] = 40 * np.random.rand(1,n_visible)
		elif input_type == 'binary':
			vec[n,:] = np.random.binomial(n=1, p=0.5, size=(1,n_visible))

		# generate noisy data around  basis vector
		for k in xrange(n_samples/n_hidden):
			if input_type == 'gaussian' or input_type == 'rlu':
				data[n + k*n_hidden, :] = vec[n,:] + 1*np.random.standard_normal(n_visible)
			elif input_type == 'binary':
				data[n + k*n_hidden, :] = vec[n,:] + 1e-2*np.random.standard_normal(n_visible)

	# center data
	if input_type == 'gaussian':
		mu    = np.mean(data, axis=0)
		data -= mu
		vec  -= mu

	# train model	
	err = my_rbm.train(data, batch_size=n_hidden, epochs=100, learning_rate=1e-2, cd_steps=1, momentum=0, weight_decay=0, verbose=True)

	# plot data samples
	plt.plot(data[:,0], data[:,1], 'b.')

	# plot model samples
	seed = np.random.binomial(n=1, p=0.5, size=(n_samples, n_hidden) )
	model_sample = my_rbm.sample_v_given_h( seed )

	for k in xrange(25):
		model_sample = my_rbm.gibbs_vhv( model_sample )
		
	plt.plot(model_sample[:,0], model_sample[:,1], 'r.')
	plt.show()
	
if __name__ == "__main__":
	test_rbm()