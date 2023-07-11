'''
    Zero_shot_learning
    ------------------
    > Here we have a anomaly dataset.
        > classes are defects and non-detects dataset.
        
    > Here we will use the pretrained models 
'''
## importing the modules
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sbn
import cv2
import keras



## Configurations
class CNF:
    seed = 42
    data_folder = "dataset"  ### replace with your training data folder
    target_shape = (224, 224)
    

### setting the reproducability
def seeding():
    np.random.seed = CNF.seed
    tf.random.set_seed(CNF.seed)
    
    print("| Seeding Done.")

seeding

### Detecting the GPU and Setting distribution strategy based on the GPU or CPU.
gpus = tf.config.experimental.list_logical_devices('GPU')
if(len(gpus)>0):
    CNF.device = "GPU"
    strategy = tf.distribute.MirroredStrategy(devices=gpus)
    print("> Running on {} | Number of devices {}".format(CNF.device, len(gpus)))
    
else:
    CNF.device = "CPU"
    strategy = tf.distribute.get_strategy() ## here we are getting the default strategy.
    print("> Running on {}".format(CNF.device))

print("> Number of replicas : ", strategy.num_replicas_in_sync)


## preparing the dataset
class Prepare_dataset:
    '''
        Here we will prepare the dataset using
        > Here we will create a data loaders
        > and data generator.
        > we will also apply data augmentation too
    '''
    
    def __init__(self):
        self.training_data_paths = None
        self.evaluation_data_paths = None
        
    def get_training_data_paths(self):
        paths = glob(os.path.join(CNF.data_folder,"**","train", "good","**.png"))
        self.training_data_paths = paths
    
    def get_evaluation_data_paths(self):
        paths = glob(os.path.join(CNF.data_folder, "**", "test", "*","**.png"))
        self.evaluation_data_paths = paths
        
    def train_data_generator(self, train_paths):
        ## paths can be ==> ./dataset/class_name/train/good/imag.png
        def load_preprocess(path):
            image = tf.keras.utils.load_img(path)
            image = tf.keras.utils.img_to_array(image)
            image = tf.reshape(image, shape=CNF.target_shape)
            return image
        
        def get_label(path):
            parts = tf.strings.split(path, os.path.sep)
            label = parts[-4]
            ### convert it into the hard label using char to int dictonary
            
            
    
    def test_data_generator(self):
        pass
    
    def dataloader(self):
        pass

    
    
#### Sanity check 

if __name__ == "__main__":
    
    PD = Prepare_dataset()
    PD.get_training_data_paths()
    PD.get_evaluation_data_paths()
    print(PD.training_data_paths[:10])
    print("**"*10)
    print(PD.evaluation_data_paths[:10])
    
    
    
    
    