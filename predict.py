import keras
from keras.preprocessing.image import img_to_array
import imutils.paths as paths
import cv2
import os
import numpy as np
import matplotlib.pylab as plt

HEIGHT = 32
WIDTH = 32
CLASS_NUM = 62
NORM_SIZE = 32
BATCH_SIZE = 32
EPOCHS = 40

test_path="./data/test"
image_paths = sorted(list(paths.list_images(test_path)))

model = keras.models.load_model("./save/model.h5")

for each_path in image_paths:
    image=cv2.imread(each_path)
    image=cv2.resize(image,(NORM_SIZE,NORM_SIZE))
    image=img_to_array(image) / 255.0
    image=np.expand_dims(image,axis=0)

    result = model.predict(image)
    proba = np.max(result)#提取最大概率
    predict_label = np.argmax(result)  # 提取最大概率下标
    label = int(each_path.split(os.path.sep)[-2])  # 提取标签
    plt.imshow(image[0])  # 显示图片
    plt.title("label:{},predict_label:{}, proba:{:.2f}".format(label, predict_label, proba))
    plt.show()