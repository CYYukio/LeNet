# encoding: utf-8

from keras.preprocessing.image import img_to_array
#图片转换为矩阵

from keras.utils import to_categorical
from imutils import paths
import numpy as np
import random
import cv2
import os


def load_data(path,norm_size,class_num):
    data=[]
    label=[]

    image_paths=sorted(list(paths.list_images(path)))  #imutls读取文件路径下所有图片
    random.seed(0)   #随机种子，保证每次顺序一样
    random.shuffle(image_paths)

    for each_path in image_paths:
        image=cv2.imread(each_path)
        image=cv2.resize(image,(norm_size,norm_size))
        image=img_to_array(image)

        data.append(image)
        #maker = int(each_path.split('/')[-2])
        maker=int(each_path.split(os.path.sep)[-2])
        # sep切分文件目录，标签类别为文件夹名称的变化，从0-61.如train文件下00014，label=14

        label.append(maker)

    data=np.array(data,dtype="float") / 255.0  #归一化
    #data = np.expand_dims(data, 0)
    #data.reshape(-1, 32, 32, 3)
    #label=np.array(label)
    #np.ravel(data)
    #data.reshape(-1, 32, 32, 3)
    label=to_categorical(label,num_classes=class_num)  #转换成类别矩阵

    return data,label