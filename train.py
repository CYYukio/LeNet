# encoding: utf-8

import matplotlib.pylab as plt
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from lenet import LeNet
import data_input
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

HEIGHT = 32
WIDTH = 32
CLASS_NUM = 62
NORM_SIZE = 32
BATCH_SIZE = 32
EPOCHS = 40

def train(aug, train_X, train_Y, test_X, test_Y):
    print("[INFO] compiling model...")
    model = LeNet.build(width=WIDTH,height=HEIGHT,depth=4,classes=CLASS_NUM)
    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

    print("[INFO] training network...")
    _history=model.fit_generator(aug.flow(train_X,train_Y,batch_size=BATCH_SIZE),
                                 validation_data=(test_X,test_Y),
                                 steps_per_epoch=len(train_X),
                                 epochs=EPOCHS,
                                 verbose=1)
    #steps_per_epoch是每次迭代，需要迭代多少个batch_size，validation_data为test数据，直接做验证，不参与训练
    model.save("./save/model.h5")#保存模型
    plt.style.use("ggplot")
    plt.figure()
    N= EPOCHS
    plt.plot(np.arange(0, N), _history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), _history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), _history.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), _history.history["val_acc"], label="val_acc")
    plt.title("loss and accuracy")
    plt.xlabel("epoch")
    plt.ylabel("loss/acc")
    plt.legend(loc="best")
    plt.savefig("./result/result.png")
    plt.show()



if __name__ =="__main__":

    train_x, train_y = data_input.load_data("../data/train", NORM_SIZE, CLASS_NUM)
    test_x, test_y = data_input.load_data("../data/test", NORM_SIZE, CLASS_NUM)

    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2,
                             zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

    train(aug,train_x,train_y,test_x,test_y)
