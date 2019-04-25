#!/bin/zsh

train='train.py'
model_dir='./model/'
gpu=1
config='CRN_DAVIS16.cfg'
resume=1
trained_path='./trained_model/CRN_Res101_Pascal-48.pth'

cd ./core && python $train 	--gpu $gpu\
							--config $config\
							--model_dir $model_dir\
							--resume $resume\
							--trained_path $trained_path

