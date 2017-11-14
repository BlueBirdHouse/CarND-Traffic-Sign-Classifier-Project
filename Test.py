import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
import time
import matlab.engine
from skimage.color import rgb2gray

#%%函数定义区
def evaluate(X_data, y_data,sess,logits,one_hot_y,BATCH_SIZE):
    '''
    logits:网络的直接输出
    one_hot_y：验证信息的输入经过变化以后的输出
    '''
    num_examples = len(X_data)
    total_accuracy = 0
    #生成验证
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        batch_x_Gray = rgb2gray(batch_x)
        batch_x_Gray = batch_x_Gray[:,:,:,np.newaxis]
        accuracy = sess.run(accuracy_operation, feed_dict={x_RGB: batch_x, x_Gray:batch_x_Gray, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def analyseSoftMaxOutPut(DataBase_X,DataBase_y,sess):
    '''
    这个函数统计由训练后的网络输出的关于DataBase_X中的相同类别的样本的平均SoftMax值
    以及相同样本的个数
    '''
    
    SoftMaxOut = tf.nn.softmax(logits)
    n = np.size(DataBase_y)
    
    #每一行对应一种样本的SoftMax值
    TotalSoftMax = np.zeros([43,43])
    #每一个元素对应一种样本的个数
    TotalExample = np.zeros([1,43])
    
    for Counter in range(n):
        AFig_RGB = np.squeeze(DataBase_X[Counter])
        AFig_Gray = rgb2gray(AFig_RGB)
        AFig_RGB = AFig_RGB[np.newaxis,:]
        AFig_Gray = AFig_Gray[np.newaxis,:,:,np.newaxis]
        feed_dict={x_RGB: AFig_RGB, x_Gray:AFig_Gray}
        Out = sess.run(SoftMaxOut, feed_dict)
        SampleNum = DataBase_y[Counter]
        
        TotalSoftMax[SampleNum,:] = np.add(TotalSoftMax[SampleNum,:],Out)
        TotalExample[0,SampleNum] += 1
        
        #print(Counter)
    
    TotalSoftMax = TotalSoftMax/(np.transpose(TotalExample))

    return (TotalSoftMax,TotalExample)


#%%数据读取区
pickle_file = './valid_Data'
with open(pickle_file, 'rb') as f:
  pickle_data = pickle.load(f)
  
  X_valid_Matlab = pickle_data['X_valid_Matlab']
  y_valid_Matlab = pickle_data['y_valid_Matlab']
  
  del pickle_data  # Free up memory
  
pickle_file = './test_Data'
with open(pickle_file, 'rb') as f:
  pickle_data = pickle.load(f)
  
  X_test_Matlab = pickle_data['X_test_Matlab']
  y_test_Matlab = pickle_data['y_test_Matlab']
  
  del pickle_data  # Free up memory

#%% 重新构建神经网络
tf.reset_default_graph()
#彩色图像输入数值格式转换层
x_RGB = tf.placeholder(tf.uint8, (None, 32, 32, 3))
x_RGB_float = tf.cast(x_RGB, tf.float32)
#%链接数据归一化层
Batch_mean, Batch_variance = tf.nn.moments(x_RGB_float,axes = [0, 1, 2])
beat_part = tf.constant([0, 0, 0],dtype=tf.float32)
#beat_part = tf.Variable(tf.zeros(3,dtype=tf.float32))
gamma_part = tf.constant([1, 1, 1],dtype=tf.float32)
#gamma_part = tf.Variable(tf.ones(3,dtype=tf.float32))
Normed_RGB_x = tf.nn.batch_norm_with_global_normalization(x_RGB_float,Batch_mean,Batch_variance,beat_part,gamma_part,variance_epsilon = 0.0001,scale_after_normalization = False)

#灰度图像输入数值格式转换层
x_Gray = tf.placeholder(tf.float64, (None, 32, 32, 1))
x_Gray_float = tf.cast(x_Gray, tf.float32)

#%链接数据归一化层
Batch_mean_Gray, Batch_variance_Gray = tf.nn.moments(x_Gray_float,axes = [0, 1, 2])
beat_part_Gray = tf.constant([0],dtype=tf.float32)
gamma_part_Gray = tf.constant([1],dtype=tf.float32)
Normed_Gray_x = tf.nn.batch_norm_with_global_normalization(x_Gray_float,Batch_mean_Gray,Batch_variance_Gray,beat_part_Gray,gamma_part_Gray,variance_epsilon = 0.0001,scale_after_normalization = False)

#卷积层1：输入32*32*3->28*28*10
mu = 0
sigma = 0.1

weight1_RGB = tf.Variable(tf.truncated_normal([5, 5, 3, 9], mean = mu, stddev = sigma,dtype=tf.float32),name = 'weight1_RGB')
bias1_RGB = tf.Variable(tf.zeros(9,dtype=tf.float32),name = 'bias1_RGB')
conv1_RGB = tf.nn.conv2d(Normed_RGB_x,weight1_RGB,strides=[1, 1, 1, 1],padding='VALID')
conv1_RGB = tf.nn.bias_add(conv1_RGB, bias1_RGB)


weight1_Gray = tf.Variable(tf.truncated_normal([5, 5, 1, 6], mean = mu, stddev = sigma),name = 'weight1_Gray')
bias1_Gray = tf.Variable(tf.zeros(6),name = 'bias1_Gray')
conv1_Gray = tf.nn.conv2d(Normed_Gray_x,weight1_Gray,strides=[1, 1, 1, 1],padding='VALID')
conv1_Gray = tf.nn.bias_add(conv1_Gray, bias1_Gray)

conv1 = tf.concat([conv1_RGB,conv1_Gray],3)

#输出：28*28*15
conv1_OutPut = tf.nn.sigmoid(conv1)

#降维层1:28*28*15->14*14*15
pool1 = tf.nn.max_pool(conv1_OutPut, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')


#卷积层1：输入14*14*12->10*10*27
weight2 = tf.Variable(tf.truncated_normal([5, 5, 15, 27], mean = mu, stddev = sigma),name = 'weight2')
bias2 = tf.Variable(tf.zeros(27),name = 'bias2')
conv2 = tf.nn.conv2d(pool1, weight2, strides=[1, 1, 1, 1], padding='VALID')
conv2 = tf.nn.bias_add(conv2, bias2)
conv2_OutPut = tf.nn.sigmoid(conv2)

#降维层1:10*10*27->5*5*27
pool2 = tf.nn.max_pool(conv2_OutPut, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')

#展开层
flatten1 = flatten(pool2)

#3全连层1
weight3 = tf.Variable(tf.truncated_normal([675,516],mean = mu, stddev = sigma),name = 'weight3')
bias3 = tf.Variable(tf.zeros(516),name = 'bias3')
Mux3 = tf.matmul(flatten1,weight3)
logits3 = tf.add(Mux3,bias3)
OutPut3 = tf.nn.sigmoid(logits3)

#4全连层2
weight4 = tf.Variable(tf.truncated_normal([516,362],mean = mu, stddev = sigma),name = 'weight4')
bias4 = tf.Variable(tf.zeros(362),name = 'bias4')
Mux4 = tf.matmul(OutPut3,weight4)
logits4 = tf.add(Mux4,bias4)
OutPut4 = tf.nn.sigmoid(logits4)

#5全连层2
weight5 = tf.Variable(tf.truncated_normal([362,43],mean = mu, stddev = sigma),name = 'weight5')
bias5 = tf.Variable(tf.zeros(43),name = 'bias5')
Mux5 = tf.matmul(OutPut4,weight5)
logits = tf.add(Mux5,bias5)

#代价函数层
y = tf.placeholder(tf.uint8, (None))
one_hot_y = tf.one_hot(y,43,dtype = tf.float32)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)

#%%启动场景恢复过程
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, './TrainRecord/20171111/SavedSession_ACC955')

#通过验证模型准确性来判断是否正确恢复
BATCH_SIZE = 128
validation_accuracy = evaluate(X_valid_Matlab,y_valid_Matlab,sess,logits,one_hot_y,BATCH_SIZE)
print('请验证这个Valid准确度是否正确：')
print(validation_accuracy)

#%%测试过程
print('测试以后的结果是：')
test_accuracy = evaluate(X_test_Matlab,y_test_Matlab,sess,logits,one_hot_y,BATCH_SIZE)
print(test_accuracy)
print('这个结果一锤定音！')

#%%计算平均SoftMax值的过程
TotalSoftMax,TotalExample = analyseSoftMaxOutPut(X_test_Matlab,y_test_Matlab,sess)

#下面的式子说明，从平均意义上讲，SoftMax的最大值发生在对应的位置上
RowMax = np.max(TotalSoftMax,axis=0)
_,Positon = np.where(TotalSoftMax == RowMax)

#%%传送到Matlab制图
eng = matlab.engine.connect_matlab()
eng.workspace['TotalSoftMax'] = matlab.double(TotalSoftMax.tolist())
eng.workspace['TotalExample'] = matlab.double(TotalExample.tolist())


