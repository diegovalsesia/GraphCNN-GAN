import numpy as np
import tensorflow as tf
import time
import random
from tf_utils import safe_log

from tensorflow.python.client import timeline
from tflearn.initializations import uniform_scaling

import scipy.io as sio


class GAN:


	def __init__(self, config):

		self.config = config 
		self.N = config.N
		self.radius = config.radius

		######### not running out gpu sources ##########
		tf_config = tf.ConfigProto()
		tf_config.gpu_options.allow_growth = True
		self.sess = tf.Session(config = tf_config)
		
		######### profiling #############################
		#self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		#self.run_metadata = tf.RunMetadata()

		############ define variables ##################
		self.gen_FC_layers = config.gen_FC_layers
		self.gen_FC_layers_num = len(self.gen_FC_layers)
		self.gen_conv_layers = config.gen_conv_layers
		self.gen_conv_layers_num = len(self.gen_conv_layers)
		self.gen_fnet_layers = config.gen_fnet_layers
		self.gen_fnet_layers_num = len(self.gen_fnet_layers[0])
		self.dis_SFC_layers = config.dis_SFC_layers
		self.dis_SFC_layers_num = len(self.dis_SFC_layers)
		self.dis_FC_layers = config.dis_FC_layers
		self.dis_FC_layers_num = len(self.dis_FC_layers)
		self.W = {}
		self.b = {}
		self.filter = {}
		self.scale={}
		self.beta={}
		self.pop_mean={}
		self.pop_var={}
		self.Sigma={}
		#self.hc={}
		# generator
		gen_vars = []

		for i in range(self.gen_FC_layers_num-1):
			name = "gen_dense_" + str(i)
			#self.W[name] = tf.Variable(tf.random_normal([self.gen_FC_layers[i], self.gen_FC_layers[i+1]], stddev=0.02), name=name)
			#self.W[name] = tf.get_variable(name, [self.gen_FC_layers[i], self.gen_FC_layers[i+1]], dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
			self.W[name] = tf.get_variable(name, [self.gen_FC_layers[i], self.gen_FC_layers[i+1]], dtype=tf.float32, initializer=uniform_scaling())
			self.b[name] = tf.Variable(tf.zeros([1, self.gen_FC_layers[i+1]]))
			gen_vars = gen_vars + [self.W[name], self.b[name]]

		for i in range(self.gen_conv_layers_num-1):
			for j in range(self.gen_fnet_layers_num-1):
				name = "gen_conv_" + str(i) + "flayer_" + str(j)
				self.W[name] = tf.Variable(tf.random_normal([self.gen_fnet_layers[i][j], self.gen_fnet_layers[i][j+1]], stddev=0.01), name=name)
				#self.W[name] = tf.get_variable(name, [self.gen_fnet_layers[j], self.gen_fnet_layers[j+1]], dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
				#self.W[name] = tf.get_variable(name, [self.gen_fnet_layers[i][j], self.gen_fnet_layers[i][j+1]], dtype=tf.float32, initializer=uniform_scaling())
				self.b[name] = tf.Variable(tf.zeros([1, self.gen_fnet_layers[i][j+1]]))
				gen_vars = gen_vars + [self.W[name], self.b[name]]
			name = "gen_conv_" + str(i)
			self.b[name] = tf.Variable(tf.zeros([1, self.gen_conv_layers[i+1]]))
			gen_vars = gen_vars + [self.b[name]]
			### bn
			self.scale['bn_scale_'+name] = tf.get_variable('bn_scale_'+name, [self.N[i+1], self.gen_conv_layers[i+1]], initializer=tf.ones_initializer())
			self.beta['bn_beta_'+name]  = tf.get_variable('bn_beta_'+name , [self.N[i+1], self.gen_conv_layers[i+1]], initializer=tf.constant_initializer(0.0))
			self.pop_mean['bn_pop_mean_'+name] = tf.get_variable('bn_pop_mean_'+name, [self.N[i+1], self.gen_conv_layers[i+1]], initializer=tf.constant_initializer(0.0), trainable=False)
			self.pop_var['bn_pop_var_'+name ]  = tf.get_variable('bn_pop_var_'+name , [self.N[i+1], self.gen_conv_layers[i+1]], initializer=tf.ones_initializer(), trainable=False)
			gen_vars = gen_vars + [self.scale['bn_scale_'+name], self.beta['bn_beta_'+name]]
			## dense3
			name = "gen_dense3_" + str(i)
			self.W[name] = tf.get_variable(name, [self.gen_conv_layers[i], self.gen_conv_layers[i+1]], dtype=tf.float32, initializer=uniform_scaling())
			self.b[name] = tf.Variable(tf.zeros([1,self.gen_conv_layers[i+1]]))
			gen_vars = gen_vars + [self.W[name], self.b[name]]
			## upsampling
			if self.config.upsampling[i]:
				# perturb method
				if self.config.upsamp_method == 'perturb':
					name = "upsamp_" + str(i)
					self.Sigma[name] = tf.Variable(config.sigma_init*tf.ones([self.gen_conv_layers[i],]))
					gen_vars = gen_vars + [self.Sigma[name]]
				# aggr method
				if self.config.upsamp_method == 'aggr':
					if self.config.upsamp_aggr_method == 'full':
						name = "upsamp_" + str(i) + '_0'
						self.W[name] = tf.Variable(tf.random_normal([self.gen_conv_layers[i], self.gen_conv_layers[i]], stddev=0.01), name=name)
						self.b[name] = tf.Variable(tf.zeros([1, self.gen_conv_layers[i]]))
						gen_vars = gen_vars + [self.W[name], self.b[name]]
						name = "upsamp_" + str(i) + '_1'
						self.W[name] = tf.Variable(tf.random_normal([self.gen_conv_layers[i], self.gen_conv_layers[i]*self.gen_conv_layers[i]], stddev=0.01), name=name)
						self.b[name] = tf.Variable(tf.zeros([1, self.gen_conv_layers[i]*self.gen_conv_layers[i]]))
						gen_vars = gen_vars + [self.W[name], self.b[name]]
					if self.config.upsamp_aggr_method == 'diag':
						name = "upsamp_" + str(i) + '_0'
						self.W[name] = tf.Variable(tf.random_normal([self.gen_conv_layers[i], self.gen_conv_layers[i]], stddev=0.01), name=name)
						self.b[name] = tf.Variable(tf.zeros([1, self.gen_conv_layers[i]]))
						gen_vars = gen_vars + [self.W[name], self.b[name]]
						name = "upsamp_" + str(i) + '_1'
						self.W[name] = tf.Variable(tf.random_normal([self.gen_conv_layers[i], self.gen_conv_layers[i]], stddev=0.01), name=name)
						self.b[name] = tf.Variable(tf.zeros([1, self.gen_conv_layers[i]]))
						gen_vars = gen_vars + [self.W[name], self.b[name]]
					if self.config.upsamp_aggr_method == 'scalar':
						name = "upsamp_" + str(i) + '_0'
						self.W[name] = tf.Variable(tf.random_normal([self.gen_conv_layers[i], self.gen_conv_layers[i]], stddev=0.01), name=name)
						self.b[name] = tf.Variable(tf.zeros([1, self.gen_conv_layers[i]]))
						gen_vars = gen_vars + [self.W[name], self.b[name]]
						name = "upsamp_" + str(i) + '_1'
						self.W[name] = tf.Variable(tf.random_normal([self.gen_conv_layers[i], 1], stddev=0.01), name=name)
						self.b[name] = tf.Variable(tf.zeros([1, 1]))
						gen_vars = gen_vars + [self.W[name], self.b[name]]
					name = "upsamp_dense3_" + str(i)
					self.W[name] = tf.get_variable(name, [self.gen_conv_layers[i], self.gen_conv_layers[i]], dtype=tf.float32, initializer=uniform_scaling())
					self.b[name] = tf.Variable(tf.zeros([1,self.gen_conv_layers[i]]))
					gen_vars = gen_vars + [self.W[name], self.b[name]]
					# bn
					#name = "upsamp_"+str(i)
					#self.scale['bn_scale_'+name] = tf.get_variable('bn_scale_'+name, [self.N[i], self.gen_conv_layers[i]], initializer=tf.ones_initializer())
					#self.beta['bn_beta_'+name]  = tf.get_variable('bn_beta_'+name , [self.N[i], self.gen_conv_layers[i]], initializer=tf.constant_initializer(0.0))
					#self.pop_mean['bn_pop_mean_'+name] = tf.get_variable('bn_pop_mean_'+name, [self.N[i], self.gen_conv_layers[i]], initializer=tf.constant_initializer(0.0), trainable=False)
					#self.pop_var['bn_pop_var_'+name ]  = tf.get_variable('bn_pop_var_'+name , [self.N[i], self.gen_conv_layers[i]], initializer=tf.ones_initializer(), trainable=False)
					#gen_vars = gen_vars + [self.scale['bn_scale_'+name], self.beta['bn_beta_'+name]]

		# discriminator
		dis_vars = []
		for i in range(self.dis_SFC_layers_num-1):
			name = "dis_shareddense_" + str(i)
			#self.W[name] = tf.Variable(tf.random_normal([self.dis_FC_layers[i], self.dis_FC_layers[i+1]], stddev=0.02), name=name)
			#self.W[name] = tf.get_variable(name, [self.dis_FC_layers[i], self.dis_FC_layers[i+1]], dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
			self.W[name] = tf.get_variable(name, [self.dis_SFC_layers[i], self.dis_SFC_layers[i+1]], dtype=tf.float32, initializer=uniform_scaling())
			self.b[name] = tf.Variable(tf.zeros([1, self.dis_SFC_layers[i+1]]))		
			dis_vars = dis_vars + [self.W[name], self.b[name]]
		for i in range(self.dis_FC_layers_num-1):
			name = "dis_dense_" + str(i)
			#self.W[name] = tf.Variable(tf.random_normal([self.dis_FC_layers[i], self.dis_FC_layers[i+1]], stddev=0.02), name=name)
			#self.W[name] = tf.get_variable(name, [self.dis_FC_layers[i], self.dis_FC_layers[i+1]], dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
			self.W[name] = tf.get_variable(name, [self.dis_FC_layers[i], self.dis_FC_layers[i+1]], dtype=tf.float32, initializer=uniform_scaling())
			self.b[name] = tf.Variable(tf.zeros([1, self.dis_FC_layers[i+1]]))		
			dis_vars = dis_vars + [self.W[name], self.b[name]]

		############ define placeholders ##############
		self.x = tf.placeholder("float", [None, config.signal_size[0], config.signal_size[1]], name="true_points")
		self.z = tf.placeholder("float", [None, config.z_size], name="latent_code")
		self.is_training = tf.placeholder(tf.bool, (), name="is_training")

		########### computational graph ###############
		self.__make_compute_graph()
		#self.clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, self.config.dis_clip[0], self.config.dis_clip[1])) for var in dis_vars]

		################## losses #####################
		self.__make_loss()

		################ optimizer ops ################
		#update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		#with tf.control_dependencies(update_ops):
		if self.config.wgan:
			self.opt_g = tf.train.RMSPropOptimizer(config.learning_rate_gen).minimize(self.loss_g, var_list=gen_vars)
			self.opt_d = tf.train.RMSPropOptimizer(config.learning_rate_dis).minimize(self.loss_d, var_list=dis_vars)
		else:
			self.opt_g = tf.train.AdamOptimizer(config.learning_rate_gen, config.beta).minimize(self.loss_g, var_list=gen_vars)
			self.opt_d = tf.train.AdamOptimizer(config.learning_rate_dis, config.beta).minimize(self.loss_d, var_list=dis_vars)

		################# summaries ###################
		tf.summary.scalar('loss_d', self.loss_d)
		tf.summary.scalar('loss_g', self.loss_g)
		#tf.summary.histogram('p_real', self.real_prob)
		#tf.summary.histogram('p_syn', self.synthetic_prob)
		#for tf_op in dis_vars:
		#	tf.summary.histogram(tf_op.name, tf_op)  
		#tf.summary.histogram('x', self.x)
		#tf.summary.histogram('x_hat', self.x_hat)
		tf.summary.histogram('synthetic_logits', self.synthetic_logits)
		tf.summary.histogram('real_logits', self.real_logits)
		#for i in range(self.gen_conv_layers_num-1):
		#	if self.config.upsampling[i]:
		#		tf.summary.histogram('Sigma_'+str(i), self.Sigma['upsamp_'+str(i)])
		self.summaries = tf.summary.merge_all()
		# Check if log_dir exists, if so delete contents
		if tf.gfile.Exists(self.config.log_dir):
			tf.gfile.DeleteRecursively(self.config.log_dir)
		self.summaries_writer = tf.summary.FileWriter(self.config.log_dir, self.sess.graph)


	def dense(self, h, name):

		return tf.matmul(h, self.W[name]) + self.b[name]


	def dense3(self, h, name):

		return tf.tensordot(h, self.W[name], axes=1) + self.b[name]


	def dense3_ein(self, h, name):

		return tf.einsum('aij,jk->aik', h, self.W[name])



	# h: (B,N,dlm1), D_prev: (B,N/2,N/2)
	# 	  +-----------------+---------------+
	# 	  | 	D_prev 	    | D_new_and_old |
	# D = +-----------------+---------------+
	# 	  | D_new_and_old^T | 	  D_new     |
	# 	  +-----------------+---------------+
	def gconv(self, h, name, i, partial_graph=False, D_prev=[]):
	
		h = tf.cast(h, tf.float64)

		if partial_graph:
			h_old = h[:,:self.N[i],:]
			h_new = h[:,self.N[i]:,:]
			sq_norms_old = tf.reduce_sum(h_old*h_old,2) # (B,N/2)
			sq_norms_new = tf.reduce_sum(h_new*h_new,2) # (B,N/2)
			D_new = tf.abs( tf.expand_dims(sq_norms_new, 2) + tf.expand_dims(sq_norms_new, 1) - 2*tf.matmul(h_new, h_new, transpose_b=True) ) # (B, N/2, N/2)
			D_new_and_old = tf.abs( tf.expand_dims(sq_norms_old, 2) + tf.expand_dims(sq_norms_new, 1) - 2*tf.matmul(h_old, h_new, transpose_b=True) ) # (B, N/2, N/2)
			D_new = tf.cast(D_new, tf.float32)
			D_new_and_old = tf.cast(D_new_and_old, tf.float32)
			D = tf.concat( [tf.concat([D_prev, D_new_and_old], axis=2), tf.concat([tf.matrix_transpose(D_new_and_old), D_new], axis=2)], axis=1 )# (B, N, N)
		else:	
			sq_norms = tf.reduce_sum(h*h,2) # (B,N)
			D = tf.abs( tf.expand_dims(sq_norms, 2) + tf.expand_dims(sq_norms, 1) - 2*tf.matmul(h, h, transpose_b=True) ) # (B, N, N)
			D = tf.cast(D, tf.float32)	
		
		h = tf.cast(h, tf.float32) 

		ret_list = tf.map_fn(lambda feat: self.gconv_inner_knn(feat[0], feat[1], name, i), [h, D], parallel_iterations=16, swap_memory=False)
		return ret_list[0]



	def gconv_inner_knn(self, x_tilde, D, name, i):
		
		# N: no. of nodes
		# K: no. of actual edges
		# Z: no. of non-isolated nodes
		# d: no. of neighbors (K=Nd)

		_, top_idx = tf.nn.top_k(-D, self.config.min_nn[i]+1) # (N, d+1)
		top_idx2 = tf.reshape(tf.tile(tf.expand_dims(top_idx[:,0],1), [1, self.config.min_nn[i]]), [-1])
		top_idx = tf.reshape(top_idx[:,1:],[-1]) # (N*d,)

		x_tilde1 = tf.gather(x_tilde, top_idx) # (K, dlm1)		
		x_tilde2 = tf.gather(x_tilde, top_idx2) # (K, dlm1)
		labels = x_tilde1 - x_tilde2 # (K, dlm1)

		j=-1
		for j in range(self.gen_fnet_layers_num-2):
			name_flayer = name + "flayer_" + str(j)
			labels = tf.nn.relu(tf.matmul(labels, self.W[name_flayer]) + self.b[name_flayer])
		theta = tf.matmul(labels, self.W[name + "flayer_" + str(j+1)]) + self.b[name + "flayer_" + str(j+1)] # (K, dlm1*dl)
		theta = tf.reshape(theta, [-1, self.config.gen_conv_layers[i], self.config.gen_conv_layers[i+1]]) # (K, dlm1, dl)

		#x = tf.einsum( 'aj,ajk->ak', x_tilde1, theta ) # (K, dl)
		x = tf.reduce_sum(tf.multiply( tf.expand_dims(x_tilde1,2), theta ), 1) # (K, dl)

		x = tf.reshape(x, [-1, self.config.min_nn[i], self.config.gen_conv_layers[i+1]]) # (N, d, dl)
		x = tf.reduce_mean(x, 1) # (N, dl)

		return [x, D]


	# # Input is (B, N*dlm1), output is (B, N*dl)
	# def gconv_old(self, h, name, i, use_radius=False):

	# 	return tf.map_fn(lambda img: self.gconv_inner_old(img, name, i), h, parallel_iterations=16, swap_memory=False)
		

	# # Input is (B, N, dlm1), output is (B, N, dl)
	# def gconv_inner_old(self, x_tilde, name, i):
		
	# 	# N: no. of nodes
	# 	# K: no. of actual edges
	# 	# Z: no. of non-isolated nodes
	# 	# d: no. of neighbors (K=Nd)

	# 	x_tilde = tf.cast(x_tilde, tf.float64)
		
	# 	# compute graph
	# 	sq_norms = tf.reduce_sum(x_tilde*x_tilde,1) # (N,)
	# 	D = tf.sqrt( tf.abs( tf.expand_dims(sq_norms, 1) + sq_norms - 2*tf.matmul(x_tilde, x_tilde, transpose_b=True) ) )# (N, N)
	# 	x_tilde = tf.cast(x_tilde, tf.float32) # (N,)

	# 	_, top_idx = tf.nn.top_k(-D, self.config.min_nn[i]+1) # (N, d+1)
	# 	top_idx2 = tf.reshape(tf.tile(tf.expand_dims(top_idx[:,0],1), [1, self.config.min_nn[i]]), [-1])
	# 	top_idx = tf.reshape(top_idx[:,1:],[-1]) # (N*d,)

	# 	x_tilde1 = tf.gather(x_tilde, top_idx) # (K, dlm1)		
	# 	x_tilde2 = tf.gather(x_tilde, top_idx2) # (K, dlm1)
	# 	labels = x_tilde1 - x_tilde2 # (K, dlm1)

	# 	j=-1
	# 	for j in range(self.gen_fnet_layers_num-2):
	# 		name_flayer = name + "flayer_" + str(j)
	# 		labels = tf.nn.relu(tf.matmul(labels, self.W[name_flayer]) + self.b[name_flayer])
	# 	theta = tf.matmul(labels, self.W[name + "flayer_" + str(j+1)]) + self.b[name + "flayer_" + str(j+1)] # (K, dlm1*dl)
	# 	theta = tf.reshape(theta, [-1, self.config.gen_conv_layers[i], self.config.gen_conv_layers[i+1]]) # (K, dlm1, dl)

	# 	x = tf.einsum( 'aj,ajk->ak', x_tilde1, theta ) # (K, dl)
		
	# 	x = tf.reshape(x, [-1, self.config.min_nn[i], self.config.gen_conv_layers[i+1]]) # (N, d, dl)
	# 	x = tf.reduce_mean(x, 1) # (N, dl)

	# 	return x


	def batch_norm_wrapper(self, inputs, name, decay = 0.999):
		
		def bn_train():
			if len(inputs.get_shape())==4:
				# for convolutional activations of size (batch, height, width, depth)
				batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
			if len(inputs.get_shape())==3:
				# for activations of size (batch, points, features)
				batch_mean, batch_var = tf.nn.moments(inputs,[0,1])
			if len(inputs.get_shape())==2:
				# for fully connected activations of size (batch, features)
				batch_mean, batch_var = tf.nn.moments(inputs,[0])
			train_mean = tf.assign(self.pop_mean['bn_pop_mean_'+name], self.pop_mean['bn_pop_mean_'+name] * decay + batch_mean * (1 - decay))
			train_var = tf.assign(self.pop_var['bn_pop_var_'+name], self.pop_var['bn_pop_var_'+name] * decay + batch_var * (1 - decay))
			with tf.control_dependencies([train_mean, train_var]):
				return tf.nn.batch_normalization(inputs, batch_mean, batch_var, self.beta['bn_beta_'+name], self.scale['bn_scale_'+name], 1e-3)

		def bn_test():
			return tf.nn.batch_normalization(inputs, self.pop_mean['bn_pop_mean_'+name], self.pop_var['bn_pop_var_'+name], self.beta['bn_beta_'+name], self.scale['bn_scale_'+name], 1e-3)

		normalized = tf.cond( self.is_training, bn_train, bn_test )
		return normalized


	def conv1(self, h, name, i):
		
		return tf.nn.conv1d(h, filters=self.filter[name], stride=1, padding='SAME', name=name)


	# def upsamp_old(self, h, i=0, name=None):
		
	# 	if self.config.upsamp_method == 'triangle':
	# 		return tf.map_fn(lambda img: self.upsamp_triangle_inner(img), h, parallel_iterations=16, swap_memory=False)
	# 	if self.config.upsamp_method == 'gauss':
	# 		return tf.map_fn(lambda img: self.upsamp_gaussian_inner(img), h, parallel_iterations=16, swap_memory=False)
	# 	if self.config.upsamp_method == 'perturb':
	# 		h2 = h + tf.multiply(tf.random_normal(tf.shape(h)), self.Sigma[name])
	# 		h2 = tf.concat([h, h2], axis=1)
	# 		return h2
	# 	if self.config.upsamp_method == 'aggr':
	# 		return tf.map_fn(lambda img: self.upsamp_aggr_inner_old(img, i), h, parallel_iterations=16, swap_memory=False)


	def upsamp(self, h, i=0, name=None):
		
		if self.config.upsamp_method == 'triangle':
			return tf.map_fn(lambda img: self.upsamp_triangle_inner(img), h, parallel_iterations=16, swap_memory=False)
		if self.config.upsamp_method == 'gauss':
			return tf.map_fn(lambda img: self.upsamp_gaussian_inner(img), h, parallel_iterations=16, swap_memory=False)
		if self.config.upsamp_method == 'perturb':
			h2 = h + tf.multiply(tf.random_normal(tf.shape(h)), self.Sigma[name])
			h2 = tf.concat([h, h2], axis=1)
			return h2
		if self.config.upsamp_method == 'aggr':
			h = tf.cast(h, tf.float64)
			sq_norms = tf.reduce_sum(h*h,2) # (B,N)
			D = tf.abs( tf.expand_dims(sq_norms, 2) + tf.expand_dims(sq_norms, 1) - 2*tf.matmul(h, h, transpose_b=True) ) # (B, N, N)
			D = tf.cast(D, tf.float32)
			h = tf.cast(h, tf.float32)
			ret_list = tf.map_fn(lambda img: self.upsamp_aggr_inner(img[0], img[1], i), [h, D], parallel_iterations=16, swap_memory=False)
			return ret_list[0], D


	# h: (N, dl), Output is (2N, dl)
	# new point is centroid of a triangle with current point and its 2-NN or a random point in the triangle
	def upsamp_triangle_inner(self, h):

		x_tilde = tf.cast(h, tf.float64)
		
		# compute graph
		sq_norms = tf.reduce_sum(x_tilde*x_tilde,1) # (N,)
		D = tf.sqrt( tf.abs( tf.expand_dims(sq_norms, 1) + sq_norms - 2*tf.matmul(x_tilde, x_tilde, transpose_b=True) ) )# (N, N)
		x_tilde = tf.cast(x_tilde, tf.float32) # (N,)
		_, top_idx = tf.nn.top_k(-D, 3) # (N, 3)

		h = tf.cast(h, tf.float32)

		nn0 = tf.reshape(top_idx[:,0],[-1]) # (N,)
		nn1 = tf.reshape(top_idx[:,1],[-1]) # (N,)
		nn2 = tf.reshape(top_idx[:,2],[-1]) # (N,)
		# centroid
		#h2 = tf.div(tf.gather(h, nn0) + tf.gather(h, nn1) + tf.gather(h, nn2), 3) # (N, dl)
		# random 
		a0 = tf.random_uniform([], 0.0, 1.0)
		a1 = tf.random_uniform([], 0.0, 1.0)
		a2 = tf.random_uniform([], 0.0, 1.0)
		asum = a0+a1+a2
		a0 = tf.div(a0, asum)
		a1 = tf.div(a1, asum)
		a2 = tf.div(a2, asum)
		h2 = a0*tf.gather(h, nn0) + a1*tf.gather(h, nn1) + a2*tf.gather(h, nn2) # (N, dl)

		return tf.concat([h, h2], axis=0)


	# h: (N, dl), Output is (2N, dl)
	# new point is sampled from gaussian with mean and variance estimated from the neighborhood
	def upsamp_gaussian_inner(self, h, n_neigh=10):

		x_tilde = tf.cast(h, tf.float64)
		x_shape = tf.shape(x_tilde)
		
		# compute graph
		sq_norms = tf.reduce_sum(x_tilde*x_tilde,1) # (N,)
		D = tf.sqrt( tf.abs( tf.expand_dims(sq_norms, 1) + sq_norms - 2*tf.matmul(x_tilde, x_tilde, transpose_b=True) ) )# (N, N)
		x_tilde = tf.cast(x_tilde, tf.float32) # (N, dl)
		_, top_idx = tf.nn.top_k(-D, n_neigh) # (N, n_neigh)
		top_idx = tf.reshape(top_idx,[-1]) # (N*n_n,)

		x_tilde1 = tf.gather(x_tilde, top_idx) # (N*n_n, dl)	
		h2 = tf.reshape(x_tilde1, [-1, n_neigh, x_shape[1]]) # (N, n_n, dl)
		
		mu, ss = tf.nn.moments(h2, axes=1) # (N, dl)

		h3 = tf.multiply(tf.sqrt(ss), tf.random_normal(x_shape)) + mu # sample from gaussian

		return tf.concat([h, h3], axis=0)


	# Input is (B, N, dl), output is (B, N, dl)
	def upsamp_aggr_inner(self, x_tilde, D, i):
		
		# N: no. of nodes
		# K: no. of actual edges
		# Z: no. of non-isolated nodes
		# d: no. of neighbors (K=Nd)

		_, top_idx = tf.nn.top_k(-D, self.config.upsamp_aggr_nn+1) # (N, d+1)
		top_idx2 = tf.reshape(tf.tile(tf.expand_dims(top_idx[:,0],1), [1, self.config.upsamp_aggr_nn]), [-1])
		top_idx = tf.reshape(top_idx[:,1:],[-1]) # (N*d,)

		x_tilde1 = tf.gather(x_tilde, top_idx) # (K, dl)		
		x_tilde2 = tf.gather(x_tilde, top_idx2) # (K, dl)
		labels = x_tilde1 - x_tilde2 # (K, dl)

		labels = tf.nn.relu(tf.matmul(labels, self.W["upsamp_"+str(i)+"_0"]) + self.b["upsamp_"+str(i)+"_0"]) # (K, dl)
		theta = tf.matmul(labels, self.W["upsamp_"+str(i)+"_1"]) + self.b["upsamp_"+str(i)+"_1"] # (K,dl*dl) or (K, dl) or (K,1)

		if self.config.upsamp_aggr_method == 'full':
			theta = tf.reshape(theta, [-1, self.config.gen_conv_layers[i], self.config.gen_conv_layers[i]]) # (K, dl, dl)
			x = tf.reduce_sum(tf.multiply( tf.expand_dims(x_tilde1,2), theta ), 1) # (K, dl)
		else:
			x = tf.multiply(x_tilde1, theta) # (K, dl)
		
		x = tf.reshape(x, [-1, self.config.upsamp_aggr_nn, self.config.gen_conv_layers[i]]) # (N, d, dl)
		x = tf.reduce_mean(x, 1) # (N, dl)

		return [x, D]


	# # Input is (B, N, dl), output is (B, N, dl)
	# def upsamp_aggr_inner_old(self, x_tilde, i):
		
	# 	# N: no. of nodes
	# 	# K: no. of actual edges
	# 	# Z: no. of non-isolated nodes
	# 	# d: no. of neighbors (K=Nd)

	# 	x_tilde = tf.cast(x_tilde, tf.float64)
		
	# 	# compute graph
	# 	sq_norms = tf.reduce_sum(x_tilde*x_tilde,1) # (N,)
	# 	D = tf.sqrt( tf.abs( tf.expand_dims(sq_norms, 1) + sq_norms - 2*tf.matmul(x_tilde, x_tilde, transpose_b=True) ) )# (N, N)
	# 	x_tilde = tf.cast(x_tilde, tf.float32) # (N,)

	# 	_, top_idx = tf.nn.top_k(-D, self.config.upsamp_aggr_nn+1) # (N, d+1)
	# 	top_idx2 = tf.reshape(tf.tile(tf.expand_dims(top_idx[:,0],1), [1, self.config.upsamp_aggr_nn]), [-1])
	# 	top_idx = tf.reshape(top_idx[:,1:],[-1]) # (N*d,)

	# 	x_tilde1 = tf.gather(x_tilde, top_idx) # (K, dl)		
	# 	x_tilde2 = tf.gather(x_tilde, top_idx2) # (K, dl)
	# 	labels = x_tilde1 - x_tilde2 # (K, dl)

	# 	labels = tf.nn.relu(tf.matmul(labels, self.W["upsamp_"+str(i)+"_0"]) + self.b["upsamp_"+str(i)+"_0"]) # (K, dl)
	# 	theta = tf.matmul(labels, self.W["upsamp_"+str(i)+"_1"]) + self.b["upsamp_"+str(i)+"_1"] # (K, dl)

	# 	x = tf.multiply(x_tilde1, theta) # (K, dl)
		
	# 	x = tf.reshape(x, [-1, self.config.upsamp_aggr_nn, self.config.gen_conv_layers[i]]) # (N, d, dl)
	# 	x = tf.reduce_mean(x, 1) # (N, dl)

	# 	return x


	def __make_compute_graph(self):

		def generator(h):

			for i in range(self.gen_FC_layers_num-1):
				name = "gen_dense_" + str(i)
				h = self.dense(h, name)
				#h = self.batch_norm_wrapper(h, name)
				h = tf.nn.leaky_relu(h)

			h = tf.reshape(h, [-1, self.N[0], self.config.gen_conv_layers[0]])
			if self.config.upsampling[0]:
				if self.config.upsamp_method == 'triangle':
					h = self.upsamp(h)				
				if self.config.upsamp_method == 'perturb' or self.config.upsamp_method == 'gauss':
					h = self.upsamp(h, name='upsamp_0')
				if self.config.upsamp_method == 'aggr':
					[h2, D] = self.upsamp(h, i=0)
					h2 = h2 + self.b["upsamp_dense3_0"] + self.dense3_ein(h, "upsamp_dense3_0")
					#h2 = self.batch_norm_wrapper(h2, "upsamp_0")
					h2 = tf.nn.leaky_relu( h2 )
					h = tf.concat([h, h2], axis=1) 

			i=-1
			for i in range(self.gen_conv_layers_num-2):
				name = "gen_conv_" + str(i)
				if self.config.upsampling[i]:
					h = self.gconv(h, name, i, partial_graph=True, D_prev=D) + self.b[name] + self.dense3_ein(h, "gen_dense3_" + str(i))
				else:
					h = self.gconv(h, name, i) + self.b[name] + self.dense3_ein(h, "gen_dense3_" + str(i))
				#h = self.batch_norm_wrapper(h, name)
				h = tf.nn.leaky_relu(h)
				if self.config.upsampling[i+1]:
					if self.config.upsamp_method == 'triangle':
						h = self.upsamp(h)				
					if self.config.upsamp_method == 'perturb' or self.config.upsamp_method == 'gauss':
						h = self.upsamp(h, 'upsamp_'+str(i+1))
					if self.config.upsamp_method == 'aggr':
						[h2, D] = self.upsamp(h, i=i+1)
						#h2 = h2 + self.b["upsamp_dense3_"+str(i+1)] + self.dense3_ein(h, "upsamp_dense3_"+str(i+1))
						h2 = h2 + self.b["upsamp_dense3_"+str(i+1)] + h
						#h2 = self.batch_norm_wrapper(h2, "upsamp_"+str(i+1))
						h2 = tf.nn.leaky_relu(h2)
						h = tf.concat([h, h2], axis=1)	

			if self.config.upsampling[-1]:
				h = self.gconv(h, "gen_conv_" + str(i+1), i+1, partial_graph=True, D_prev=D) + self.b[name] + self.dense3_ein(h, "gen_dense3_" + str(i))	
			else:
				h = self.gconv(h, "gen_conv_" + str(i+1), i+1) + self.b["gen_conv_" + str(i+1)] + self.dense3_ein(h, "gen_dense3_" + str(i+1))

			return h

		# Input: BxNx3
		def discriminator(h):
			
			for i in range(self.dis_SFC_layers_num-1):
				name = "dis_shareddense_" + str(i)
				h = tf.nn.leaky_relu( self.dense3(h, name) ) 

			h = tf.reduce_max(h, axis=1, name='dis_maxpool')
			h = tf.reshape(h, [-1, self.dis_FC_layers[0]])

			for i in range(self.dis_FC_layers_num-2):
				name = "dis_dense_" + str(i)
				h = tf.nn.leaky_relu( self.dense(h, name) ) 

			h = self.dense(h, "dis_dense_" + str(i+1))
			
			return h

		self.x_hat = generator( self.z ) 
		self.synthetic_logits = discriminator( self.x_hat )
		self.synthetic_prob = tf.sigmoid( self.synthetic_logits )
		self.real_logits = discriminator( self.x )
		self.real_prob = tf.sigmoid( self.real_logits )

		if self.config.wgan:
			epsilon = tf.random_uniform([], 0.0, 1.0)
			self.x_int = epsilon * self.x + (1 - epsilon) * self.x_hat
			self.d_hat = discriminator(self.x_int)


	def __make_loss(self):
		
		if self.config.wgan:
			# cost for discriminator
			self.loss_d = -tf.reduce_mean(self.real_logits) + tf.reduce_mean(self.synthetic_logits)
			# cost for generator
			self.loss_g = -tf.reduce_mean(self.synthetic_logits)
			# for gradient penalty
			ddx = tf.gradients(self.d_hat, self.x_int)[0]
			ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
			ddx = tf.reduce_mean(tf.square(ddx - 1.0) * self.config.scale)
			self.loss_d = self.loss_d + ddx
		else:
			# cost for discriminator
			dis_cost_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logits, labels=tf.ones_like(self.real_logits)))
			dis_cost_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.synthetic_logits, labels=tf.zeros_like(self.synthetic_logits)))
			self.loss_d = dis_cost_real + dis_cost_gen
			# cost for generator
			self.loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.synthetic_logits, labels=tf.ones_like(self.synthetic_logits)))


	def do_variables_init(self):

		init = tf.global_variables_initializer()       
		self.sess.run(init)


	def save_model(self, path):

		saver = tf.train.Saver()
		saver.save(self.sess, path)


	def restore_model(self, path):

		saver = tf.train.Saver()
		saver.restore(self.sess, path)
		self.is_Init = True


	def __get_feed_dict(self, data, noise, is_training):
		return {self.x: data, self.z: noise, self.is_training: is_training}


	def fit(self, data, noise, iter_no, dis_iter):


		feed_dict = self.__get_feed_dict(data, noise, True)
			
		#self.sess.run(self.opt_d, feed_dict = feed_dict, options=self.options, run_metadata=self.run_metadata)
		self.sess.run(self.opt_d, feed_dict = feed_dict)
		if dis_iter < self.config.dis_n_iter:
			return

		#pc_gen, _, summaries_val = self.sess.run((self.x_hat, self.opt_g, self.summaries), feed_dict = feed_dict, options=self.options, run_metadata=self.run_metadata)
		if iter_no % self.config.summaries_every_iter == 0:
			pc_gen, _, summaries_val = self.sess.run((self.x_hat, self.opt_g, self.summaries), feed_dict = feed_dict)
		else:
			pc_gen, _ = self.sess.run((self.x_hat, self.opt_g), feed_dict = feed_dict)
		
		#pc_gen, _, summaries_val, h_val_d, h_val_0, h_val_1, h_val_2, h_val_3, D_val, x_val = self.sess.run((self.x_hat, self.opt_g, self.summaries, self.hc["gen_dense"], self.hc["gen_conv_0"], self.hc["gen_conv_1"], self.hc["gen_conv_2"], self.hc["gen_conv_3"], self.DD, self.xx), feed_dict = feed_dict)
		#sio.savemat('%sdebug_%d.mat' % (self.config.render_dir,iter_no),{'h_val_d': h_val_d, 'h_val_0': h_val_0, 'h_val_1': h_val_1, 'h_val_2': h_val_2, 'h_val_3': h_val_3, 'D': D_val, 'x_tilde': x_val})

		# for gconv_test
		#pc_gen, _, summaries_val, K_val = self.sess.run((self.x_hat, self.opt_g, self.summaries, self.K), feed_dict = feed_dict, options=self.options, run_metadata=self.run_metadata)
		#print K_val

		# for gconv_knn test
		#pc_gen, _, summaries_val, x_tilde_val, D_val = self.sess.run((self.x_hat, self.opt_g, self.summaries, self.x_tilde_debug, self.DD), feed_dict = feed_dict, options=self.options, run_metadata=self.run_metadata)
		#sio.savemat('%sdebug_%d.mat' % (self.config.render_dir,iter_no),{'x_tilde':x_tilde_val , 'D':D_val})


		#if self.config.wgan:
			#self.sess.run(self.clip_discriminator_var_op)

		if iter_no % self.config.summaries_every_iter == 0:
			self.summaries_writer.add_summary(summaries_val, iter_no)

		return pc_gen


	def generate(self, noise):

		return self.sess.run(self.x_hat, feed_dict={self.z: noise, self.is_training: False})
