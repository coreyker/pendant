# A simple RBM implmentation
#
# See also: 
#	http://deeplearning.net/tutorial/code/rbm.py
#	http://www.cs.toronto.edu/~hinton/code/rbm.m
#
# NB!: persistent-cd does not seem to be working correctly
# TBD: add sparsity-based regularization (sparsity of weights / sparsity of activations)
#
# Author: Corey Kereliuk

import pdb
import numpy as np
from matplotlib import pyplot as plt

class rbm:
	def __init__(self, n_visible=10, n_hidden=10, input_type='binary', persistent=False):				

        # model parameters
		self.n_visible    = n_visible
		self.n_hidden     = n_hidden		
		if not (input_type == 'binary' or input_type == 'gaussian'):
			raise Exception("Unknown input type.  Currently recognized options are 'binary' or 'gaussian'")			
		self.input_type   = input_type		
		self.persistent   = persistent
		self.model_sample = None

		self.h_bias       = np.zeros( (1, n_hidden) )
		self.v_bias       = np.zeros( (1, n_visible) )
		self.W            = 1e-1 * np.random.randn( n_visible, n_hidden )

		self.W_inc        = np.zeros( (n_visible, n_hidden) )
		self.h_inc        = np.zeros( (1, n_hidden) )
		self.v_inc        = np.zeros( (1, n_visible) )

	def sigmoid(self, x):
		return 1. / (1 + np.exp(-x))

	def propup(self, vis):
		return self.sigmoid( np.dot(vis, self.W) + np.repeat(self.h_bias, vis.shape[0], axis=0) )

	def propdown(self, hid):
		u = np.dot(hid, self.W.T) + np.repeat(self.v_bias, hid.shape[0], axis=0)
		if self.input_type=='binary':
			return self.sigmoid( u )
		elif self.input_type=='gaussian':
			return u

	def sample_h_given_v(self, sample):	
		return np.random.binomial( n=1, p=self.propup( sample ) )
	
	def sample_v_given_h(self, sample):
		u = self.propdown( sample )
		if self.input_type=='binary':
			return u #np.random.binomial( n=1, p=u )
		elif self.input_type=='gaussian':
			return u + np.random.standard_normal( u.shape )

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
		v_delta = (v_pos - v_neg) / batch_size
		h_delta = (h_pos - h_neg) / batch_size
		W_delta = (W_pos - W_neg) / batch_size					

		Err = np.sum((data_sample - self.model_sample)**2)

		return W_delta, h_delta, v_delta, Err

	def train(self, data, batch_size=1, learning_rate=0.1, epochs=10, \
		cd_steps=1, momentum=0, weight_decay=0):

		"""
		Train RBM with stocastic gradient descent
		"""
		n_batches = data.shape[0] / batch_size

		Err = []
		for epoch in xrange(epochs):
			print "epoch %d of %d" % (epoch, epochs)

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

				print "Mini batch %d/%d, Error=%f" % (index, n_batches, err)
		return Err

def test_rbm():
	n_visible = 2
	n_hidden  = 3
	r         = rbm(n_visible, n_hidden)

	# make training data
	n_samples = 500
	data      = np.zeros( (n_samples, n_visible) )

	# bi-modal data
	for n in xrange(n_samples):
		p = np.random.binomial(n=1, p=0.5)
		x = p * (0.0 + 0.1*np.random.standard_normal(1)) + (1-p) * (1 + 0.1*np.random.standard_normal(1))
		y = p * (0.0 + 0.1*np.random.standard_normal(1)) + (1-p) * (1 + 0.1*np.random.standard_normal(1))
		data[n,:] = [x,y]

	# train model	
	r.train(data, batch_size=20, epochs=100, learning_rate=1e-1, cd_steps=1, momentum=0, weight_decay=0)

	# plot data samples
	plt.plot(data[:,0], data[:,1], 'b.')

	# plot model samples
	model_sample = np.zeros( (n_samples, n_visible) )

	for n in xrange(n_samples):	
		seed = np.random.binomial(n=1, p=0.5*np.ones(r.h_bias.shape) )
		tmp  = r.propdown( seed )

		for k in xrange(10):
			tmp = r.gibbs_vhv( tmp )
		
		model_sample[n,:] = tmp + 0.1*np.random.standard_normal(2)

	plt.plot(model_sample[:,0], model_sample[:,1], 'r.')
	plt.show()
	
if __name__ == "__main__":
	test_rbm()