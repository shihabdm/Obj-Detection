from detectron2.engine import DefaultPredictor

import os
import pickle

from utils import *

cfg_save_path = "OD_cfg.pickle" #the saved pickle file

#load cfg

with open(cfg_save_path, 'rb') as f:
  cfg = pickle.load(f) #load cfg

#defined output director in configuration file - the cfg.OUTPUT_DIR
#Trainer saves final model as model_final.pth - will be in directory output folder
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") 

#define threshold for detection 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 #any object of confidence below 0.5 not displayed

predictor = DefaultPredictor(cfg) #default predictor will load custom model and use cfg file

#Define another function to test image from test folder
image_path = "test/d9.jpeg"
on_image(image_path, predictor)