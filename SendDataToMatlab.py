#包文件导入区
import pickle
from zipfile import ZipFile
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
import matlab.engine

#自己的包定义区
import Support

#%%数据文件导入区
FileName = './traffic-signs-data.zip'
zipf = ZipFile(FileName)
# 长生一个zip内文件名集合
filenames_pbar = tqdm(zipf.namelist(), unit='files')
# 循环读取ZIP文件内所有数据
for filename in filenames_pbar:
    # 检查是否是目录，防止有其他东西
    if not filename.endswith('/'):
        image_file = zipf.open(filename)
        #按照需要打开文件
        if filename == 'test.p':
            test = pickle.load(image_file)     
        if filename == 'train.p':
            train = pickle.load(image_file)
        if filename == 'valid.p':
            valid = pickle.load(image_file)
        image_file.close()
zipf.close()
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

#%%训练数据
#显示统计数据
#训练集的个数
n_train = np.shape(y_train)[0]

#验证集的个数
n_validation = np.shape(y_valid)[0]

#测试集的个数
n_test = np.shape(y_test)[0]

#交通标志图像的大小
image_shape = np.shape(X_train)[1::]

#不同标志的个数
n_classes = np.max(y_train)+1

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#%%训练图像处理过程
#随机排列图像
X_train, y_train = shuffle(X_train, y_train)


#将数据转移到Matlab做再处理
#Support.FigureDataToMatlab(X_valid,0,4410,'X_valid')
#Support.FigureDataToMatlab(y_valid,0,4410,'y_valid')

Support.FigureDataToMatlab(X_test,0,12630,'X_test')
Support.FigureDataToMatlab(y_test,0,12630,'y_test')
'''
Support.FigureDataToMatlab(X_train,0,34799,'X_train')
Support.FigureDataToMatlab(y_train,0,34799,'y_train')
'''

