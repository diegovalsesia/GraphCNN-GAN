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
parser.add_argument('--render_dir', default='', help='Renders directory')
parser.add_argument('--save_dir', default='', help='Trained model directory')
param = parser.parse_args()


# import config
config = Config()
config.render_dir = param.render_dir
config.save_dir = param.save_dir

#class_name = raw_input('Give me the class name (e.g. "chair"): ').lower()
class_name = param.class_name

model = GAN(config)
model.do_variables_init()
model.restore_model(config.save_dir+'model.ckpt')


# testing
for test_no in range(config.N_test):

	noise = np.random.normal(size=[config.batch_size, config.z_size], scale=0.2)
	pc_gen = model.generate(noise)
	sio.savemat('%srender.mat' % (config.render_dir,),{'X_hat':pc_gen})
