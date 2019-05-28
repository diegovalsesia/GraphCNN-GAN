#!/home/valsesia/tensorflow-python2.7/bin/python
import os
import os.path as osp
import numpy as np
import shutil
import sys

from config import Config
from in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, load_all_point_clouds_under_folder
from gan import GAN
from general_utils import *
from PIL import Image

import scipy.io as sio

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--class_name', default='', help='Shapenet class')
parser.add_argument('--start_iter', type=int, default=1, help='Start iteration (ex: 10001)')
parser.add_argument('--render_dir', default='', help='Renders directory')
parser.add_argument('--log_dir', default='', help='Tensorboard log directory')
parser.add_argument('--save_dir', default='', help='Trained model directory')
param = parser.parse_args()


# import config
config = Config()
config.render_dir = param.render_dir
config.log_dir = param.log_dir
config.save_dir = param.save_dir


#class_name = raw_input('Give me the class name (e.g. "chair"): ').lower()
class_name = param.class_name

# import data
syn_id = snc_category_to_synth_id()[class_name]
class_dir = osp.join(config.top_in_dir , syn_id)
all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)

model = GAN(config)
model.do_variables_init()

# if len(sys.argv)==1 or int(sys.argv[1])==1:
# 	start_iter=0
# 	config.N_iter = config.N_iter+1
# 	# delete old renders
# 	shutil.rmtree(config.render_dir)
# 	os.mkdir(config.render_dir)
# else:
# 	start_iter=int(sys.argv[1])
# 	model.restore_model(config.save_dir+'model.ckpt')
# 	print 'Resuming training from iter %d' % start_iter

if param.start_iter==1:
	start_iter = 0
	config.N_iter = config.N_iter+1
	# delete old renders
	shutil.rmtree(config.render_dir)
	os.mkdir(config.render_dir)
else:
	start_iter = param.start_iter
	model.restore_model(config.save_dir+'model.ckpt')
	print 'Resuming training from iter %d' % start_iter



# training
for iter_no in range(start_iter, config.N_iter):

	for dis_iter in range(config.dis_n_iter):
		noise = np.random.normal(size=[config.batch_size, config.z_size], scale=0.2)
		data = all_pc_data.next_batch(config.batch_size)[0]
		model.fit(data, noise, iter_no, dis_iter)

	noise = np.random.normal(size=[config.batch_size, config.z_size], scale=0.2)
	data = all_pc_data.next_batch(config.batch_size)[0]
	pc_gen = model.fit(data, noise, iter_no, config.dis_n_iter)

	if iter_no % 1000 == 0:
		sio.savemat('%srender_%d.mat' % (config.render_dir,iter_no),{'X_hat':pc_gen})

	# save image of point cloud
	if iter_no % config.renders_every_iter == 0:
		pc_gen = np.reshape(pc_gen[0,:], [2048,3])
		im_array = point_cloud_three_views(pc_gen)
		img = Image.fromarray(np.uint8(im_array*255.0))
		img.save('%srender_%d.jpg' % (config.render_dir,iter_no))

	if iter_no % config.save_every_iter == 0:
		model.save_model(config.save_dir+'model.ckpt')

	if iter_no % 10000 == 0:
		os.mkdir(config.save_dir+str(iter_no))
		model.save_model(config.save_dir+str(iter_no)+'/model.ckpt')

	if iter_no % 10:
		with open(config.log_dir+'start_iter', "w") as text_file:
			text_file.write("%d" % iter_no)


# testing
for test_no in range(config.N_test):

	noise = np.random.normal(size=[config.batch_size, config.z_size], scale=0.2)
	img = model.generate(noise)
