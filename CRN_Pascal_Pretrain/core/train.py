import sys
import os
import cv2

sys.path.append('../')


import time
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import argparse
from datetime import datetime
import logging
import math
import importlib
from utils.params import Params
from utils import util
from loader.CRN_VOC_iter import dataIter
from CRN import CRN, Loss_calc



def train_CRN_Net(args):
	cfg = Params(os.path.join("../config",args.config))
	# set up params
	cfg.ctx = [int(i) for i in args.gpu.split(',')]
	if len(cfg.ctx)>1:
		Exception('Only for 1-GPU mode')


	cfg_attr = vars(cfg)
	cfg_attr.update(vars(args))


	# set up data loader
	trainIter = dataIter(cfg)

	# set up model
	if cfg.backbone == 'CRN_Res101':
		model = CRN(mtype=101,num_classes=1)
	if cfg.backbone == 'CRN_Res50':
		model = CRN(mtype=50,num_classes=1)

	if args.resume:
		mdl_dir = args.model_dir
		pre_pfx = args.pretrained_prefix
		pre_epc = args.pretrained_epoch
		net_pfx = cfg.network
		saved_state_dict = torch.load(os.path.join(mdl_dir,pre_pfx,net_pfx+'_'+pre_epc+'.pth'))
		model.load_state_dict(saved_state_dict)

	if cfg.use_global_stats == True:
		model.eval() # 

	if len(cfg.ctx)>0:
		model.cuda(cfg.ctx[0])

	# set up optimizer
	if cfg.optimizer == 'SGD':
		optimizer = optim.SGD(model.parameters(),lr = cfg.learning_rate, 
												momentum = cfg.momentum, weight_decay = cfg.wd)#
	elif cfg.optimizer == 'Adam':
		optimizer = optim.Adam(model.parameters(),lr = cfg.learning_rate, weight_decay = cfg.wd)
	else:
		Exception('SGD or Adam')

	optimizer.zero_grad()


	# set up model path
	model_path = os.path.join(args.model_dir, cfg.prefix)
	if not os.path.isdir(model_path):
		os.mkdir(model_path)
	model_full_path = os.path.join(
		model_path, datetime.now().strftime('%Y_%m_%d_%H_%M'))
	if not os.path.isdir(model_full_path):
		os.mkdir(model_full_path)


	# set up log
	util.save_log(cfg.prefix, model_full_path)
	logging.info(
		'---------------------------TIME-------------------------------')
	logging.info('-------------------{}------------------------'.format(
		datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
	for k, v in sorted(cfg_attr.items(), key=lambda x: x[0]):
		logging.info("%s : %s", k, v)

	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
 	
	# training phase
	for epoch in range(cfg.begin_epoch,cfg.end_epoch):
		trainIter.reset()
		totalLoss = 0.0
		tic = time.time()
		for iter in tqdm(range(trainIter.iter_cnt)):

			img,msk_16,msk_32,gt0,gt1,gt2,gt3,gt4,gt5= trainIter.next()
			
			out = model([img,msk_32])
			loss0 = Loss_calc(out[0],gt0)
			loss1 = Loss_calc(out[1],gt1)
			loss2 = Loss_calc(out[2],gt2)
			loss3 = Loss_calc(out[3],gt3)
			loss4 = Loss_calc(out[4],gt4)
			loss5 = Loss_calc(out[5],gt5)

			loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5

			loss.backward()
			if iter%cfg.updateIter == 0:
				optimizer.step()
				optimizer.zero_grad()
			totalLoss += loss.data.cpu().numpy()/trainIter.iter_cnt

		logger.info('Epoch[%d] Train-Loss=%.5f', epoch, totalLoss)
		toc = time.time()
		logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc - tic))

		if epoch%cfg.frequence == 0:
			print 'taking snapshot: '+ os.path.join(cfg.network+'_'+str(epoch)+'.pth')
			torch.save(model.cpu().state_dict(),os.path.join(model_full_path,cfg.network+'-'+str(epoch)+'.pth'))
			model.cuda(cfg.ctx[0])




if __name__ == '__main__':

	parser = argparse.ArgumentParser(
					description="Train Model for TubeNet")
	parser.add_argument('--model_dir', type=str, default='./model/',
					help='model folder')
	parser.add_argument('--gpu', type=str, default='1',
					help='GPU devices to train with, e.g. \'0,1,2 \'')
	parser.add_argument('--config', type=str, default='TubeNet_VOC.cfg',
					help='cofing file for train')

	parser.add_argument('--resume', action='store_true',
					help='continue training')
	parser.add_argument('--pretrained_prefix', type=str, default=None,
					help='prefix of pretrained model')
	parser.add_argument('--pretrained_epoch', type=str, default=None,
					help='training epoch of pretrained model')

	args = parser.parse_args()

	cudnn.enabled = True

	train_CRN_Net(args)