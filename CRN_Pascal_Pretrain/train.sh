#!/bin/zsh

train='train.py'
model_dir='./model/'
gpu=0
config='CRN_Pascal.cfg'


cd ./core && python $train 	--gpu $gpu\
							--config $config\
							--model_dir $model_dir

