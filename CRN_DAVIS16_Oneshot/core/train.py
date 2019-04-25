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
from CRN import CRN, Loss_calc
from OneShot_util import read_vos_test_list,compute_iou_for_binary_segmentation,get_testing_batch,get_training_batch
import numpy as np


def train_Tube_Net(args):
	cfg = Params(os.path.join("../config",args.config))
	# set up params
	cfg.ctx = [int(i) for i in args.gpu.split(',')]
	if len(cfg.ctx)>1:
		Exception('Only for 1-GPU mode')


	cfg_attr = vars(cfg)
	cfg_attr.update(vars(args))

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






	name_list, f1_img, f1_gt, img_list, label_list = read_vos_test_list(cfg)
	mIoU = 0.0
	MAE = 0.0
	seq_num = len(name_list)


	for seq_i in range(seq_num):

		seq = name_list[seq_i]
		f1_i = f1_img[seq_i]
		f1_g = f1_gt[seq_i]
		imgs = img_list[seq_i]
		labels =  label_list[seq_i]


		# set up model
		if cfg.backbone == 'CRN_Res101':
			model = CRN(mtype=101,num_classes=1)
		if cfg.backbone == 'CRN_Res50':
			model = CRN(mtype=50,num_classes=1)

		if args.resume==1:		
			print('load model:'+args.trained_path)
			logging.info('load model:'+args.trained_path)
			saved_state_dict = torch.load(args.trained_path)
			model.load_state_dict(saved_state_dict)

		if cfg.use_global_stats == True:
			model.eval() # use_global_stats = True 

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



		totalLoss = 0.0
		tic = time.time()

		logger.info('Finetuning on the set- %s', seq)

		for epoch in tqdm(range(cfg.finetune_epoch)):
			img,msk_16,msk_32,gt0,gt1,gt2,gt3,gt4,gt5= get_training_batch(cfg,f1_i,f1_g)
			
			out = model([img,msk_32])
			loss0 = Loss_calc(out[0],gt0)
			loss1 = Loss_calc(out[1],gt1)
			loss2 = Loss_calc(out[2],gt2)
			loss3 = Loss_calc(out[3],gt3)
			loss4 = Loss_calc(out[4],gt4)
			loss5 = Loss_calc(out[5],gt5)
			loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			totalLoss += loss.data.cpu().numpy()/cfg.finetune_epoch

		model.eval()
		torch.save(model.cpu().state_dict(),os.path.join(model_full_path,seq+'_iter'+str(cfg.finetune_epoch)+'.pth'))
		model.cuda(cfg.ctx[0])

		toc = time.time()
		logger.info('Train-Loss=%.5f, Time cost=%.3f', totalLoss,(toc - tic))

		iouu = 0.0
		mae = 0.0


	
		img_ = imgs[0]
		label_ = labels[0]

		img_test,lbl_test,mask_test=get_testing_batch(cfg,img_,label_)

		mask__ = lbl_test[0,0].cpu().data.numpy()

		for test_ in range(len(imgs)):
			model.eval()
			torch.cuda.empty_cache()
			img_ = imgs[test_]
			label_ = labels[test_]

			img_test,lbl_test,mask_test=get_testing_batch(cfg,img_,label_)

			kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(8,8))  
			mask__ = cv2.dilate(mask__,kernel) 

			mask_temp = mask__>0
			msk_atn0 = cv2.resize(mask_temp.astype(np.float),(256,256) , interpolation = cv2.INTER_NEAREST)
			msk_atn0 = Variable(torch.from_numpy(msk_atn0).float().view(1,1,256,256)).cuda(cfg.ctx[0])	
			msk_atn1 = cv2.resize(mask_temp.astype(np.float),(128,128) , interpolation = cv2.INTER_NEAREST)
			msk_atn1 = Variable(torch.from_numpy(msk_atn1).float().view(1,1,128,128)).cuda(cfg.ctx[0])	
			msk_atn2 = cv2.resize(mask_temp.astype(np.float),(64,64) , interpolation = cv2.INTER_NEAREST)
			msk_atn2 = Variable(torch.from_numpy(msk_atn2).float().view(1,1,64,64)).cuda(cfg.ctx[0])	
			msk_atn3 = cv2.resize(mask_temp.astype(np.float),(32,32) , interpolation = cv2.INTER_NEAREST)
			msk_atn3 = Variable(torch.from_numpy(msk_atn3).float().view(1,1,32,32)).cuda(cfg.ctx[0])	
			msk_atn4 = cv2.resize(mask_temp.astype(np.float),(16,16) , interpolation = cv2.INTER_NEAREST)
			msk_atn4 = Variable(torch.from_numpy(msk_atn4).float().view(1,1,16,16)).cuda(cfg.ctx[0])	

			mask_ = cv2.resize(mask__.astype(np.float),(16,16) )
			mask_t = Variable(torch.from_numpy(mask_).float().view(1,16,16))

			out = model([img_test,mask_t.cuda(cfg.ctx[0])])
			mask_out = (out[0][0,0].cpu().data.numpy()>0.5).astype(np.float)

			kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(64, 64))  
			mask__ = cv2.dilate(mask__,kernel) 
 
		 	mask__ = mask_out*mask__

			lbl_test = (lbl_test).cpu().data.numpy()

			#cv2.imshow("mask__",mask__)  
			#cv2.waitKey (50) 

			iouu += compute_iou_for_binary_segmentation(mask__>0.5,lbl_test[:,:]>0.5)
			mae += (np.abs(mask__-lbl_test[:,:])).sum()
		mIoU += iouu/len(imgs)
		MAE += mae/len(imgs)/(cfg.frame_num*cfg.img_size/8*cfg.img_size/8)

		logger.info('Testing: mIoU-%.5f,  MAE-%.5f', iouu/len(imgs), mae/len(imgs)/(cfg.frame_num*cfg.img_size/8*cfg.img_size/8))
		logger.info('---------------------------------')
	
	logger.info('####################################')
	logger.info('Total Testing: mIoU-%.5f,  MAE-%.5f', mIoU/seq_num, MAE/seq_num)
	logger.info('####################################')
		


if __name__ == '__main__':

	parser = argparse.ArgumentParser(
					description="Train Model for TubeNet")
	parser.add_argument('--model_dir', type=str, default='/scratch2/PingHu/VOS/Single/model/',
					help='model folder')
	parser.add_argument('--gpu', type=str, default='1',
					help='GPU devices to train with, e.g. \'0,1,2 \'')
	parser.add_argument('--config', type=str, default='TubeNet_VOC.cfg',
					help='cofing file for train')

	parser.add_argument('--resume', type=int, default=0,
					help='model resume')
	parser.add_argument('--trained_path', type=str, default=None,
					help='pretrained model')


	args = parser.parse_args()

	cudnn.enabled = True

	train_Tube_Net(args)