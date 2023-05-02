#import setup logger
from detectron2.utils.logger import setup_logger

setup_logger()

#register the dataset with detectron2
from detectron2.data.datasets import register_coco_instances

#import default trainer to train the models
from detectron2.engine import DefaultTrainer

#need to import other libraries
import os
import pickle

#import utils.py
from utils import *

#Custom-Object Detection Code - for example using: R50-FPN3x (fast RCNN model)
#Speed and accuracy is a common trade-off between models
#Higher performance/accuracy - but lower speed
#Higher speed - but lower accuracy

#Corresponding link: 
#https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml
#but remove ALL up to COCO-Detection, and enter this as config_file_path

#-->EDIT:
config_file_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
#-->EDIT:
checkpoint_url = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

#where to save output model specified below:
output_dir = "./output/object_detection"

#--> EDIT: Define classes in the dataset
num_classes = 3

#train on CUDA (since CUDA present) (can also use cpu)
device = "cuda"

#-->EDIT:
train_dataset_name = "LP_train" #LP_train <-- :EDIT
train_images_path = "train" #define train images path
train_json_annot_path = "train.json" #define train.json path

#-->EDIT:
test_dataset_name = "LP_test" #LP_test 
test_images_path = "test" #define test images path
test_json_annot_path = "test.json" #define test.json path

cfg_save_path = "OD_cfg.pickle" #EDIT: for object detection

########################

#Register dataset train and test with Detectron2

#Registers using dataset name, dataset metadata, path to annotations file,
#and where traindataset is located (image_root)

#Registers image dataset for 'training' model = train
register_coco_instances(name = train_dataset_name, metadata={}, json_file=train_json_annot_path, image_root=train_images_path)

#Registers image dataset for 'test' model = test
register_coco_instances(name = test_dataset_name, metadata={}, 
json_file=test_json_annot_path, image_root=test_images_path)

#Run at this stage to check for errors [7]

#To check whether annotations detected correctly by Detectron2
#For this, write function utils.py

#run following to check verification:
plot_samples(dataset_name=train_dataset_name, n=2)

###########

def main():
  cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir)

  with open(cfg_save_path, 'wb') as f:
    pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)
  
  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

  trainer = DefaultTrainer(cfg)
  trainer.resume_or_load(resume=False)

  trainer.train()

if __name__ =='__main__':
  main()

