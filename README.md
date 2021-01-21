# LeNet

环境：python -3.8.5
      keras  -2.4.3


>>>>>>>>>>>>>>>>>2021.01.16<<<<<<<<<<<<<<<<<<<<
数据集：交通标志，每类约100张图片，62类

链接：https://pan.baidu.com/s/1xkJipbYldL6EBV_oMqqttQ 
提取码：l68l 
复制这段内容后打开百度网盘手机App，操作更方便哦

数据集较大，放在网盘


目前遇到错误:
ValueError: ('Input data in `NumpyArrayIterator` should have rank 4. You passed an array with shape', (0,))

data升一维依然没有调通

input_data.py  ————  数据预处理（剪裁、读入等）
lenet.py       ————  网络结构
train.py       ————  训练
main.py        ————  没有把input，train分开
channel.py     ————  测试写代码过程中一些小bug（通道数读入等等）

跑train.py就可以


>>>>>>>>>>>>>>>>>2021.01.21<<<<<<<<<<<<<<<<<<<<
解决bug,实现预测图片，模型训练准确率为94%左右
