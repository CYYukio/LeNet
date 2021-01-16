from keras.preprocessing.image import img_to_array
#图片转换为矩阵
import matplotlib.pylab as plt
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import random
import cv2
from lenet import LeNet
import os
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

HEIGHT = 32
WIDTH = 32
CLASS_NUM = 62
NORM_SIZE = 32
BATCH_SIZE = 32
EPOCHS = 40
INIT_LR = 1e-3
def load_data(path):
    data=[]
    label=[]

    image_paths=sorted(list(paths.list_images(path)))  #imutls读取文件路径下所有图片
    random.seed(0)   #随机种子，保证每次顺序一样
    random.shuffle(image_paths)

    for each_path in image_paths:
        image=cv2.imread(each_path)
        image=cv2.resize(image,(NORM_SIZE,NORM_SIZE))
        image=img_to_array(image)

        data.append(image)
        maker = int(each_path.split('/')[-2])
        # sep切分文件目录，标签类别为文件夹名称的变化，从0-61.如train文件下00014，label=14

        label.append(maker)

    data=np.array(data,dtype="float") / 255.0  #归一化
    data = np.expand_dims(data, 0)
    label=np.array(label)
    label=to_categorical(label,num_classes=CLASS_NUM)  #转换成类别矩阵

    return data,label

def train(aug, train_X, train_Y, test_X, test_Y):
    print("[INFO] compiling model...")
    model = LeNet.build(width=WIDTH,height=HEIGHT,depth=3,classes=CLASS_NUM)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

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

    train_x, train_y = load_data("../data/train")
    test_x, test_y = load_data("../data/test")

    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2,
                             zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

    train(aug,train_x,train_y,test_x,test_y)
