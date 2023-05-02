#To check whether annotations detected correctly by Detectron2
#For this, write function utils.py

#import dataset catalogue and metadata catalogue from Detectron2
from detectron2.data import DatasetCatalog, MetadataCatalog

#import Visualizer to visualize annotations and predictions
from detectron2.utils.visualizer import Visualizer

#import getconfig method to load configuration for object detection model
from detectron2.config import get_cfg

#import modelzoo to load pre-trained model checkpoints for object detection
from detectron2 import model_zoo

#import colour modes
from detectron2.utils.visualizer import ColorMode

#import other libraries to load images and plot images
import random
import cv2
import matplotlib.pyplot as plt

#define function which takes dataset_name as input (n=1, for 1 image)
def plot_samples(dataset_name, n=1):
  #function to retrieve dataset
  dataset_custom = DatasetCatalog.get(dataset_name)
  #function to retrieve metadata releated to dataset
  dataset_custom_metadata = MetadataCatalog.get(dataset_name)

  #to plot images runs a loop
  for s in random.sample(dataset_custom, n):
    img = cv2.imread(s["file_name"]) #loads image for each particular sample
    #initialize visualizer

    #RGB format and cv2 format changing for -1
    v = Visualizer(img[:,:,::-1], metadata=dataset_custom_metadata, scale=0.5)
    v = v.draw_dataset_dict(s)
    plt.figure(figsize=(15,20))
    plt.imshow(v.get_image())
    plt.show()
  
 ##############

def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir):
  cfg = get_cfg()

  cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
  cfg.DATASETS.TRAIN = (train_dataset_name,)
  cfg.DATASETS.TEST = (test_dataset_name,)

  cfg.DATALOADER.NUM_WORKERS = 2

  cfg.SOLVER.IMS_PER_BATCH = 2 #number of images per batch
  cfg.SOLVER.BASE_LR = 0.00025
  cfg.SOLVER.MAX_ITER = 1000 #number of iterations
  cfg.SOLVER.STEPS = []

  cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
  cfg.MODEL.DEVICE = device
  cfg.OUTPUT_DIR = output_dir

  return cfg

###test.py

def on_image(image_path, predictor):
  im = cv2.imread(image_path)
  outputs = predictor(im)
  v = Visualizer(im[:,:,::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
  v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

  plt.figure(figsize=(14,10))
  plt.imshow(v.get_image())
  plt.show()

  #Video Method not included:(31mins 51sec)
  





