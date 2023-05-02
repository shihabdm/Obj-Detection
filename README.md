# Obj-Detection
 Object detection test for cat and dog images

Complete the following Initialisation Steps on Google Colab for files in repo

**Step 1: Following code required to install pyyaml and Detectron2 GoogleColab**

!pip install pyyaml==5.1

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

**Step 2: Install labelme**

!pip install labelme

**Step 3: Convert 'train' dataset to COCO JSON**

run labelme2coco.py train --output train.json

**Step 4: Convert 'test' dataset to COCO JSON**

run labelme2coco.py test --output test.json


Complete the following steps to train and test the model on Google Colab for files in repo
- utils.py contains configuration for model visualiser
- train.py contains configuration for training dataset
- test.py contains configuration for test dataset

**Step 1: Train dataset**

run train.py

**Step 2: Test dataset**

run test.py
