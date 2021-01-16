from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class LeNet:
    @staticmethod
    #静态方法，可不实例化对象调用函数
    def build(width, height, depth,classes):

        inputshape=(height, width, depth)
        #图片格式
        if K.image_data_format() == "channels_first":  # for tensorflow
            inputshape = (depth, height, width)

        model = Sequential()
        '''卷积->池化'''
        model.add(Conv2D(20,(5,5),padding="same",input_shape=inputshape))
        #Con2D(filter数量->卷积后输出的维度，（卷积核大小5*5），padding填充边缘，inputshape不解释了)
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        #poolsize(2,2)缩小2倍-xy方向

        '''卷积->池化'''
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        '''全连接'''
        model.add(Flatten())  #压平
        model.add(Dense(500)) #全连接
        model.add(Activation("relu"))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
