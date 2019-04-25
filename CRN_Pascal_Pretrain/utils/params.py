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
		self.backbone =  parser.get("model", "backbone")
		self.prefix = parser.get("model","network")
		self.pretrained = parser.getboolean("model","pretrained")
		self.frame_num = parser.getint("model","frame_num")
		self.img_size = parser.getint("model","img_size")

		# epoch
		self.begin_epoch = parser.getint("epoch", "begin_epoch")
		self.end_epoch = parser.getint("epoch", "end_epoch")
		self.frequence = parser.getint("epoch", "frequence")

		# iterator
		itr_set = "iterator_test" if for_test else 'iterator_train'
		self.batch_size = parser.getint(itr_set, "batch_size")
		self.num_thread = parser.getint(itr_set, "num_thread")
		self.gt_path = parser.get(itr_set, "gt_path")
		self.img_path = parser.get(itr_set, "img_path")
		self.list_path = parser.get(itr_set, "list_path")
		self.data_aug = parser.getboolean(itr_set, "data_aug")
		self.use_global_stats = parser.getboolean(itr_set, "use_global_stats")
		self.updateIter = parser.getint(itr_set, "updateIter")

		# optimizer
		self.optimizer = parser.get("optimizer", "name")
		self.learning_rate = parser.getfloat("optimizer", "learning_rate")
		self.wd = parser.getfloat("optimizer", "wd")
		self.momentum = parser.getfloat("optimizer", "momentum")

		# misc
		self.description = parser.get("misc", "description")
