#测试代码中一些模块的功能
import tensorflow as tf
import sys
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
image_name = "./data/train/00001/00025_00001.png"

def lm_get_image_size(file_name):
    print('load %s as ...' % file_name)
    img = cv2.imread(file_name)
    sp = img.shape
    print(sp)
    sz1 = sp[0]  # height(rows) of image
    sz2 = sp[1]  # width(colums) of image
    sz3 = sp[2]  # channels
    print('height: %d \nwidth: %d \nchannels: %d' % (sz1, sz2, sz3))
    return sp

def load_image(filepath):
    data = cv2.imread(filepath)
    data = cv2.resize(data, (32, 32))

    #cv2.imshow("1", data)
    #cv2.waitKey(0)
    data = img_to_array(data)
    label=filepath.split('/')[-2]
    data = np.array(data, dtype="float")

    label = np.array(label)
    label = to_categorical(label, num_classes=62)  # 转换成类别矩阵

    data=tf.convert_to_tensor(data)
    label=tf.convert_to_tensor(label)
    print(label.shape,data.shape)
    #data = np.expand_dims(data, 0)
    #print(data.shape)


def main():
    #lm_get_image_size(image_name)
    load_image(image_name)


if __name__ == '__main__':
    sys.exit(main())