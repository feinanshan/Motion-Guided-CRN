# Motion-Guided-CRN (under construction...)

This is a PyTorch implementation of **Cascaded Refinement Network** described in
>[Ping Hu, Gang Wang, Xiangfei Kong , Jason Kuen, Yap-Peng Tan;. "Motion-Guided Cascaded Refinement Network for Video Object Segmentation." CVPR, 2018](https://sites.google.com/view/pinghu/projects/video-object-segmentation)

Results on DAVIS2016 can be download [here](https://github.com/feinanshan/Motion-Guided-CRN/blob/master/doc/CRN.zip)

### Prerequisites
1. PyTorch
2. Opencv

### Pretraining on Pascal_VOC
1. Edit `img_path`, `gt_path`, `list_path`  in the file `./CRN_Pascal_Pretrain/config/CRN_Pascal.cfg`.
2. Edit file `./CRN_Pascal_Pretrain/train.sh`.
3. Run `sh train.sh`.

### Pretraining on the DAVIS16 Training Split
1. Edit `img_path`, `gt_path`, `list_path`  in the file `./CRN_DAVIS16_Pretrain/config/CRN_DAVIS16.cfg`.
2. Copy a [Pascal-pretrained](https://drive.google.com/open?id=1kEBnETlgNzws8neVo7zK5I6jt5CS0PIE) model  to `./CRN_DAVIS16_Pretrain/trained_model/`. 
3. Edit file `./CRN_DAVIS16_Pretrain/train.sh`.
4. Run `sh train.sh`.

### Online Finetuning
1. Edit `img_path`, `gt_path`, `list_path`  in the file `./CRN_DAVIS16_Oneshot/config/CRN_DAVIS16.cfg`.
2. Copy a  [DAVIS16-pretrained](https://drive.google.com/open?id=1uITKDKtzeBiIKNiAhTztHQr1RwFpUu5n) model to `./CRN_DAVIS16_Oneshot/trained_model/`. 
3. Edit file `./CRN_DAVIS16_Pretrain/train.sh`.
4. Run `sh train.sh`.


### Bibtex
@InProceedings{Hu_2018_CVPR,

author = {Hu, Ping and Wang, Gang and Kong , Xiangfei and Kuen,  Jason and Tan, Yap-Peng},

title = {Motion-Guided Cascaded Refinement Network for Video Object Segmentation},

booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},

year = {2018}

}
