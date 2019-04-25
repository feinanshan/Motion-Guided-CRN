import ConfigParser
import numpy as np
from ast import literal_eval

class Params(object):
	def __init__(self,config_file='./config.ini',for_test=False):
		parser = ConfigParser.ConfigParser()
		parser.read(config_file)

		self.gpu_nums = 1
		# model
		self.network =  parser.get("model", "network")
		self.prefix = parser.get("model","network")
		self.backbone =  parser.get("model", "backbone")
		self.frame_num = parser.getint("model","frame_num")
		self.img_size = parser.getint("model","img_size")

		# epoch
		self.begin_epoch = parser.getint("epoch", "begin_epoch")
		self.end_epoch = parser.getint("epoch", "end_epoch")
		self.frequence = parser.getint("epoch", "frequence")

		# iterator
		self.batch_size = parser.getint("iterator", "batch_size")
		self.num_thread = parser.getint("iterator", "num_thread")
		self.gt_path = parser.get("iterator", "gt_path")
		self.img_path = parser.get("iterator", "img_path")
		self.list_path = parser.get("iterator", "list_path")
		self.gt_path_val = parser.get("iterator", "gt_path_val")
		self.img_path_val = parser.get("iterator", "img_path_val")
		self.list_path_val = parser.get("iterator", "list_path_val")
		self.data_aug = parser.getboolean("iterator", "data_aug")
		self.use_global_stats = parser.getboolean("iterator", "use_global_stats")
		self.update_iter = parser.getint("iterator", "update_iter")

		# optimizer
		self.optimizer = parser.get("optimizer", "name")
		self.learning_rate = parser.getfloat("optimizer", "learning_rate")
		self.wd = parser.getfloat("optimizer", "wd")
		self.momentum = parser.getfloat("optimizer", "momentum")

		# misc
		self.description = parser.get("misc", "description")
