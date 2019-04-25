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
import numpy as np
import random
random.seed(0)
PIXEL_MEANS = np.array([[[0.485, 0.456, 0.406]]])


def compute_iou_for_binary_segmentation(y_argmax, target):
  I = np.logical_and(y_argmax == 1, target == 1).sum()
  U = np.logical_or(y_argmax == 1, target == 1).sum()
  if U == 0:
    IOU = 1.0
  else:
    IOU = float(I) / U
  return IOU

def read_vos_test_list(cfg):
	path_to_gt = cfg.gt_path_val
	path_to_img = cfg.img_path_val
	path_to_file = cfg.list_path_val

	name_list = []
	img_list = []
	label_list = []
	f1_img = []
	f1_gt = []

	with open(path_to_file) as f:
	#f =['train ']
		for line in f:
			frame_num = len(os.listdir(os.path.join(path_to_img,line[:-1])))

			imgs = []
			labels = []


			f1_img.append(os.path.join(path_to_img,line[:-1],'00000.jpg'))
			f1_gt.append(os.path.join(path_to_gt,line[:-1],'00000.png'))

			for i in range(0,len(os.listdir(os.path.join(path_to_img,line[:-1])))):

				ID_cur0 = i

				name = '%05d'%ID_cur0
				imgs.append(os.path.join(path_to_img,line[:-1],name+'.jpg'))
				labels.append(os.path.join(path_to_gt,line[:-1],name+'.png'))
	

			name_list.append(line[:-1])
			img_list.append(imgs)
			label_list.append(labels)

	return name_list,f1_img,f1_gt,img_list,label_list


def get_testing_batch(cfg, data, label):
	"""
	build databatch
	"""
	dim = (cfg.img_size,cfg.img_size) #256

	img_test = np.zeros((cfg.img_size,cfg.img_size,3,1))
	lbl_test = np.zeros((cfg.img_size,cfg.img_size,1,1))
	mask_test = np.zeros((cfg.img_size/32,cfg.img_size/32,1))

	img_temp0 = cv2.imread(data).astype(float)/255
	img_temp0 -= PIXEL_MEANS
	img_temp = cv2.resize(img_temp0,dim).astype(float)

	gt_temp = cv2.imread(label).astype(float)
	if len(gt_temp.shape)==3:
		gt_temp = gt_temp[:,:,0]
	gt_temp = (gt_temp>0).astype(float)



	img_test[:,:,:,0] = img_temp
	lbl_test[:,:,0,0] = cv2.resize(gt_temp,(dim[1],dim[0]) , interpolation = cv2.INTER_NEAREST)
	mask_test[:,:,0] = cv2.resize(gt_temp,(dim[1]/32,dim[0]/32) , interpolation = cv2.INTER_NEAREST)



	img_test = img_test.transpose((3,2,0,1))
	lbl_test = lbl_test.transpose((3,2,0,1))
	mask_test = mask_test.transpose((2,0,1))

	img_test = Variable(torch.from_numpy(img_test).float())
	lbl_test = Variable(torch.from_numpy(lbl_test).float())
	mask_test = Variable(torch.from_numpy(mask_test).float())

	if len(cfg.ctx)>0:
		img_test = img_test.cuda(cfg.ctx[0])
		lbl_test = lbl_test.cuda(cfg.ctx[0])
		mask_test = mask_test.cuda(cfg.ctx[0])

	return img_test,lbl_test,mask_test


def get_training_batch(cfg, data, label):
	"""
	build databatch
	"""
	Sizes = [[352,352],[416,416],[448,448],[480,480],[512,512],[544,544]]
	Slice = random.sample(range(len(Sizes)),1)
	sizeInput = Sizes[Slice[0]]
	dim = (sizeInput[1],sizeInput[0])

	img = np.zeros((dim[0],dim[1],3,cfg.batch_size))
	msk_16 = np.zeros((dim[0]/16,dim[1]/16,cfg.batch_size))
	msk_32 = np.zeros((dim[0]/32,dim[1]/32,cfg.batch_size))
	gt0 = np.zeros((dim[0],dim[1],1,cfg.batch_size))
	gt1 = np.zeros((dim[0]/2,dim[1]/2,1,cfg.batch_size))
	gt2 = np.zeros((dim[0]/4,dim[1]/4,1,cfg.batch_size))
	gt3 = np.zeros((dim[0]/8,dim[1]/8,1,cfg.batch_size))
	gt4 = np.zeros((dim[0]/16,dim[1]/16,1,cfg.batch_size))
	gt5 = np.zeros((dim[0]/32,dim[1]/32,1,cfg.batch_size))

	for ib in range(cfg.batch_size):

		flip_p = random.uniform(0, 1)		
		erORdi = random.uniform(0, 1)
		kernalShape = random.uniform(0, 3)
		if kernalShape<=1:
			kernal_type = cv2.MORPH_RECT
		elif kernalShape<=2:
			kernal_type = cv2.MORPH_ELLIPSE
		else:
			kernal_type = cv2.MORPH_CROSS

		gt_temp = cv2.imread(label).astype(float)
		if len(gt_temp.shape)==3:
			gt_temp = gt_temp[:,:,0]
		gt_temp[gt_temp >=0.5] = 1
		gt_temp[gt_temp <0.5] = 0
		gt_temp.flags.writeable = False

		img_temp= cv2.imread(data).astype(float)/255
		img_temp -= PIXEL_MEANS

		if flip_p>0.5:
			img_temp = np.fliplr(img_temp)
			gt_temp = np.fliplr(gt_temp)

		[top,left,bottom,right] = bboxs_from_mask(gt_temp,1)

		crop_p  = random.uniform(0, 1)

		img_ = img_temp
		gt = gt_temp

		if crop_p>0.75:
			img_ = img_temp[top[0]:bottom[0],left[0]:right[0],:]
			gt = gt_temp[top[0]:bottom[0],left[0]:right[0]]

		img_ = cv2.resize(img_,dim)
		gt = cv2.resize(gt,(dim[1],dim[0]),interpolation = cv2.INTER_NEAREST)

		gt0[:,:,0,ib] = cv2.resize(gt,(dim[1],dim[0]) , interpolation = cv2.INTER_NEAREST)
		gt1[:,:,0,ib] = cv2.resize(gt,(dim[1]/2,dim[0]/2) , interpolation = cv2.INTER_NEAREST)
		gt2[:,:,0,ib] = cv2.resize(gt,(dim[1]/4,dim[0]/4) , interpolation = cv2.INTER_NEAREST)
		gt3[:,:,0,ib] = cv2.resize(gt,(dim[1]/8,dim[0]/8) , interpolation = cv2.INTER_NEAREST)
		gt4[:,:,0,ib] = cv2.resize(gt,(dim[1]/16,dim[0]/16) , interpolation = cv2.INTER_NEAREST)
		gt5[:,:,0,ib] = cv2.resize(gt,(dim[1]/32,dim[0]/32) , interpolation = cv2.INTER_NEAREST)

		if erORdi >= 0.85:
			kernel_size =  random.sample(range(1,dim[1]/32),1)
			kernel = cv2.getStructuringElement(kernal_type,(kernel_size[0], kernel_size[0]))  
			gt = cv2.erode(gt,kernel)  
		if erORdi <= 0.15:
			kernel_size =  random.sample(range(1,dim[1]/32),1)
			kernel = cv2.getStructuringElement(kernal_type,(kernel_size[0], kernel_size[0]))  
			gt = cv2.dilate(gt,kernel)  

		img[:,:,:,ib] = img_
		msk_16[:,:,ib] = cv2.resize(gt,(dim[1]/16,dim[0]/16) , interpolation = cv2.INTER_NEAREST)
		msk_32[:,:,ib] = cv2.resize(gt,(dim[1]/32,dim[0]/32) , interpolation = cv2.INTER_NEAREST)

	img = img.transpose((3,2,0,1))
	msk_16 = msk_16.transpose((2,0,1))
	msk_32 = msk_32.transpose((2,0,1))
	gt0 = gt0.transpose((3,2,0,1))
	gt1 = gt1.transpose((3,2,0,1))
	gt2 = gt2.transpose((3,2,0,1))
	gt3 = gt3.transpose((3,2,0,1))
	gt4 = gt4.transpose((3,2,0,1))
	gt5 = gt5.transpose((3,2,0,1))

	img = Variable(torch.from_numpy(img).float())
	msk_16 = Variable(torch.from_numpy(msk_16).float())
	msk_32 = Variable(torch.from_numpy(msk_32).float())
	gt0 = Variable(torch.from_numpy(gt0).float())
	gt1 = Variable(torch.from_numpy(gt1).float())
	gt2 = Variable(torch.from_numpy(gt2).float())
	gt3 = Variable(torch.from_numpy(gt3).float())
	gt4 = Variable(torch.from_numpy(gt4).float())
	gt5 = Variable(torch.from_numpy(gt5).float())

	if len(cfg.ctx)>0:
		img = img.cuda(cfg.ctx[0])
		msk_16 = msk_16.cuda(cfg.ctx[0])
		msk_32 = msk_32.cuda(cfg.ctx[0])
		gt0 = gt0.cuda(cfg.ctx[0])
		gt1 = gt1.cuda(cfg.ctx[0])
		gt2 = gt2.cuda(cfg.ctx[0])
		gt3 = gt3.cuda(cfg.ctx[0])
		gt4 = gt4.cuda(cfg.ctx[0])
		gt5 = gt5.cuda(cfg.ctx[0])

	return img,msk_16,msk_32,gt0,gt1,gt2,gt3,gt4,gt5




def bboxs_from_mask(mask, smpl_num):
	hrzn = mask.sum(0)
	vrtc = mask.sum(1)
	left = 0
	right = len(hrzn)-1
	top = 0
	bottom = len(vrtc)-1
	for i in range(smpl_num*2,len(hrzn)):
		if hrzn[i] > 0:
			left = i
			break

	for i in range(smpl_num*2,len(hrzn)):
		if hrzn[len(hrzn)-i-1] > 0:
			right = len(hrzn)-i-1
			break

	for i in range(smpl_num*2,len(vrtc)):
		if vrtc[i] > 0:
			top = i
			break

	for i in range(smpl_num*2,len(vrtc)):
		if vrtc[len(vrtc)-i-1] > 0:
			bottom = len(vrtc)-i-1
			break

	t = random.sample(range(0,50),smpl_num)
	l = random.sample(range(0,50),smpl_num)
	b = random.sample(range(len(vrtc)-50,len(vrtc)),smpl_num)
	r = random.sample(range(len(hrzn)-50,len(hrzn)),smpl_num)
	#t.sort()
	#l.sort()
	#b.sort()
	#r.sort()

	return [t,l,b,r]
