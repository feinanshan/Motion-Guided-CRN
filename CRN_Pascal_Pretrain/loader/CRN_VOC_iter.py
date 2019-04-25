import torch
import random
import sys
import time
import pyarrow as pa
sys.path.insert(0, '../')
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from PIL import Image
import cv2
import os
import numpy as np
from multiprocessing import Process
from multiprocessing import Queue as MPQueue

from threading import Thread, Lock

PIXEL_MEANS = np.array([[[0.485, 0.456, 0.406]]])

class dataIter(object):
	def __init__(self, cfg=None):
		self.cfg = cfg
		self.ctx = cfg.ctx
		self.batch_size = cfg.batch_size
		self.img_path = cfg.img_path
		self.gt_path = cfg.gt_path
		self.list_path = cfg.list_path
		self.data_list = []
		self.label_list = []
		self.order = []
		self.cursor = 0
		self.cnt = 0
		self.data = None
		self.label = None
		dataIter.read_voc_train_list(self,cfg)

		self.iter_cnt = self.cnt/self.batch_size 
		self.size = len(self.data_list)

		self.index_queue = MPQueue()
		self.result_queue = MPQueue(maxsize=cfg.num_thread)
		self.workers = None

		#self.reset()
		self._thread_start(cfg.num_thread)
		#self.next()


	def reset(self):
		self._shuffle()
		self.cursor = 0
		self._insert_queue()

	def _shuffle(self):
		random.shuffle(self.order)


	def _insert_queue(self):
		for i in range(0, len(self.order), self.batch_size):
			if i + self.batch_size <= len(self.order):
				self.index_queue.put(self.order[i:i + self.batch_size])

	def iter_next(self):
		return self.cursor + self.batch_size <= self.size

	def _thread_start(self, num_thread):

		self.workers = [Process(target=dataIter._worker,
						args=[self.cfg, self.data_list, self.label_list,
						self.index_queue, self.result_queue])
						for _ in range(num_thread)]

		for worker in self.workers:
			worker.daemon = True 
			worker.start()


	def next(self):
		if self.iter_next():
			self.cursor += self.batch_size
			img,msk_16,msk_32,gt0,gt1,gt2,gt3,gt4,gt5 =  self.result_queue.get()
			img = Variable(torch.from_numpy(img).float())
			msk_16 = Variable(torch.from_numpy(msk_16).float())
			msk_32 = Variable(torch.from_numpy(msk_32).float())
			gt0 = Variable(torch.from_numpy(gt0).float())
			gt1 = Variable(torch.from_numpy(gt1).float())
			gt2 = Variable(torch.from_numpy(gt2).float())
			gt3 = Variable(torch.from_numpy(gt3).float())
			gt4 = Variable(torch.from_numpy(gt4).float())
			gt5 = Variable(torch.from_numpy(gt5).float())

			if len(self.ctx)>0:
				img = img.cuda(self.ctx[0])
				msk_16 = msk_16.cuda(self.ctx[0])
				msk_32 = msk_32.cuda(self.ctx[0])
				gt0 = gt0.cuda(self.ctx[0])
				gt1 = gt1.cuda(self.ctx[0])
				gt2 = gt2.cuda(self.ctx[0])
				gt3 = gt3.cuda(self.ctx[0])
				gt4 = gt4.cuda(self.ctx[0])
				gt5 = gt5.cuda(self.ctx[0])

			return img,msk_16,msk_32,gt0,gt1,gt2,gt3,gt4,gt5

		else:
			raise StopIteration


	@staticmethod
	def _worker(cfg, data_list, label_list, index_queue, result_queue):

		while True:
			indexes = index_queue.get()
			if indexes is None:
				return
			img,msk_16,msk_32,gt0,gt1,gt2,gt3,gt4,gt5 = dataIter.get_batch(cfg, data_list,label_list, indexes)
			data = [img,msk_16,msk_32,gt0,gt1,gt2,gt3,gt4,gt5]

			result_queue.put(data)


	@staticmethod
	def get_batch(cfg, data_list, label_list, indexes):
		"""
		build databatch
		"""
		Sizes = [[352,352],[416,416],[448,448],[480,480],[512,512]]
		Slice = random.sample(range(len(Sizes)),1)
		sizeInput = Sizes[Slice[0]]
		dim = (sizeInput[1],sizeInput[0])

		img = np.zeros((dim[0],dim[1],3,cfg.batch_size))
		msk_16 = np.zeros((dim[0]/16,dim[1]/16,1,cfg.batch_size))
		msk_32 = np.zeros((dim[0]/32,dim[1]/32,1,cfg.batch_size))
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


			idx = indexes[ib]

			gt_temp = Image.open(label_list[idx])
			gt_temp=np.array(gt_temp)
			if len(gt_temp.shape)==3:
				gt_temp = gt_temp[:,:,0]
			gt_temp[gt_temp==225] = 0
			obj_num = np.unique(gt_temp)
			obj_set = random.sample(range(1,len(obj_num)),1) 

			while (gt_temp==obj_set[0]).sum()<1024:
				idx = random.sample(range(len(data_list)),1)[0]
				gt_temp = Image.open(label_list[idx])
				gt_temp=np.array(gt_temp)
				if len(gt_temp.shape)==3:
					gt_temp = gt_temp[:,:,0]
				gt_temp[gt_temp==225] = 0
				obj_num = np.unique(gt_temp)
				obj_set = random.sample(range(1,len(obj_num)),1) 

			img_temp= cv2.imread(data_list[idx]).astype(float)/255
			img_temp -= PIXEL_MEANS

			gt_temp = (gt_temp==obj_set[0]).astype(float)

			if flip_p>0.5:
				img_temp = np.fliplr(img_temp)
				gt_temp = np.fliplr(gt_temp)

			[top,left,bottom,right] = dataIter.bboxs_from_mask(gt_temp,1)

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

			if erORdi >= 0.333:
				kernel_size =  random.sample(range(1,10),1)
				kernel = cv2.getStructuringElement(kernal_type,(kernel_size[0], kernel_size[0]))  
				gt = cv2.erode(gt,kernel)  
			else: 
				kernel_size =  random.sample(range(1,16),1)
				kernel = cv2.getStructuringElement(kernal_type,(kernel_size[0], kernel_size[0]))  
				gt = cv2.dilate(gt,kernel)  

			img[:,:,:,ib] = img_
			msk_16[:,:,0,ib] = cv2.resize(gt,(dim[1]/16,dim[0]/16) , interpolation = cv2.INTER_NEAREST)
			msk_32[:,:,0,ib] = cv2.resize(gt,(dim[1]/32,dim[0]/32) , interpolation = cv2.INTER_NEAREST)



		img = img.transpose((3,2,0,1))
		msk_16 = msk_16.transpose((3,2,0,1))
		msk_32 = msk_32.transpose((3,2,0,1))
		gt0 = gt0.transpose((3,2,0,1))
		gt1 = gt1.transpose((3,2,0,1))
		gt2 = gt2.transpose((3,2,0,1))
		gt3 = gt3.transpose((3,2,0,1))
		gt4 = gt4.transpose((3,2,0,1))
		gt5 = gt5.transpose((3,2,0,1))

		return img,msk_16,msk_32,gt0,gt1,gt2,gt3,gt4,gt5



	@staticmethod
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

		t = random.sample(range(0,top//2+1),smpl_num)
		l = random.sample(range(0,left//2+1),smpl_num)
		b = random.sample(range((bottom+len(vrtc))//2-1,len(vrtc)),smpl_num)
		r = random.sample(range((right+len(hrzn))//2-1,len(hrzn)),smpl_num)
		t.sort()
		l.sort()
		b.sort()
		r.sort()

		return [t,l,b,r]



	def read_voc_train_list(self,cfg):
		path_to_gt = self.gt_path
		path_to_img = self.img_path
		path_to_file = self.list_path

		with open(path_to_file) as f:
			for line in f:
				self.data_list.append(os.path.join(path_to_img,line[:-1]+".jpg"))
				self.label_list.append(os.path.join(path_to_gt,line[:-1]+".png"))
				self.order.append(self.cnt)
				self.cnt += 1
			return

