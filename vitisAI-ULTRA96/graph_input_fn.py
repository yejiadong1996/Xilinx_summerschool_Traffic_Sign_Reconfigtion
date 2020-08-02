import cv2
import numpy as np
import os
import gzip
import struct

calib_batch_size = 50


sorts_list = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42']
img_size_net = 32
CONV_INPUT = "x_input_input"
#CONV_INPUT = "x_input"
calib_batch_size = 30

def load_valid_data(data_path):
    label_cnt = 0
    test_images = []
    test_lables = []
    for sort_path in sorts_list:    
        flower_list = os.listdir(data_path + sort_path)
        for img_name in flower_list:
            img_path = data_path + sort_path + "/" + img_name
            img = cv2.imread(img_path,0)  
            img_scale = cv2.resize(img,(img_size_net, img_size_net), interpolation = cv2.INTER_CUBIC)
            if not img is None:
                test_images.append(img_scale / 255.)
                test_lables.append(label_cnt)
        label_cnt += 1 
    test_images=np.array(test_images)
    test_images=test_images[..., None]             
    return test_images, test_lables

dataset_valid_path = './dataset_valid/'
(validSet_images, validSet_lables) = load_valid_data(dataset_valid_path)

validSet_images = np.array(validSet_images)
print(validSet_images.shape)
validSet_lables = np.array(validSet_lables)
def calib_input(iter):
    images = []
    for index in range(0, calib_batch_size):
        images.append(validSet_images[index])

    return {CONV_INPUT: images}

