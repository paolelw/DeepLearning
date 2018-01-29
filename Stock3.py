
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# In[2]:


# ---- 参数定义---- 
cell_size=10       # 隐藏层cell个数
input_size=7       # 输入层维数
time_step = 20     # 步长窗口
output_size=1      # 输出层维数
lr=0.0006         # 学习率
batch_size=60      # 每批次大小
split_line=5800 #前5800行作为训练集


# In[3]:


# ---- 数据导入 ---- 
df = pd.read_csv('dataset_2.csv') # 导csv
origin = df.iloc[:,2:].values     # #取第3～10列，一共8列，原始数据(6109, 8)


# In[4]:


# 标准化，工具函数
def normal(data):
    mean=np.mean(data,axis=0)
    std=np.std(data,axis=0)
    return (data-mean)/std, mean, std #有三个返回值


# In[5]:


# 训练集
#使用前5800行,其中每个输入的x是(20,7)的矩阵，每个输出的y是(20,1)的矩阵，它是20条x对应的y合成的
train,_,_ = normal(origin[:split_line]) # (5800, 8),取normal函数第1个返回值，舍弃后两个返回值
x_train,y_train=[],[]
for i in range(len(train)//time_step): # i=0~5780
    x_train.append(train[i*time_step:(i+1)*time_step,:-1]) #往x_train里放入一个(20,7)的矩阵
    y_train.append(train[i*time_step:(i+1)*time_step,-1,np.newaxis]) 
    #x_train.append( train[i:i+time_step   ,:-1  ] ) #往x_train里放入一个(20,7)的矩阵
    #y_train.append( train[i:i+time_step   ,-1   ,np.newaxis] )


# In[6]:


print(len(x_train))
print(np.array(x_train).shape)

#for i in range(len(x_train)):
#    print(i, np.array(x_train[i]).shape)
#for j in range(len(y_train)):
#    print(j, np.array(y_train[j]).shape)

#必须要加各种print才可以看出各种矩阵形状的变化
#for batch in range( len(x_train)//batch_size+1): # 1个周期内批次数
#    start = batch*batch_size 
#    end = min( (batch+1)*batch_size , len(x_train)) 
#    batch_xs = x_train[start : end ]# 获得本批次的x,y
#    batch_ys = y_train[start : end ]
#    print(np.array(batch_xs).shape)


# In[7]:


# 测试集
test,mean,std= normal(origin[split_line:])
x_test,y_test=[],[]
for i in range(len(test)//time_step): # (6109-5801)/20 ~= 15
    x_test.append(test[i*time_step:(i+1)*time_step,:-1])
    #y_test.append(test[i*time_step:(i+1)*time_step,-1])
    y_test.extend( test[i*time_step:(i+1)*time_step,-1] ) 


# In[8]:


#for i in range(len(x_test)):
#    print(i, np.array(x_test[i]).shape)
#for j in range(len(y_test)):
#    print(j, np.array(y_test[j]).shape)
print(len(x_test))
print(np.array(x_test).shape)
print(len(y_test))
print(np.array(y_test).shape)


# In[9]:


# 定义网络
#X就是输入的数据张量Tensor了，形状为(#data_example, time_step, input_size)的Tensor
X=tf.placeholder(tf.float32, shape=[None,time_step,input_size]) # (Any, 20, 7)
Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size]) # (Any, 20, 1)

#输入层权重、偏置
w1 = tf.Variable(tf.random_normal([input_size,cell_size])) # (7,10)
b1 = tf.Variable(tf.constant(0.1,shape=[cell_size,])) # (10,1)

#输出层权重、偏置
w2 = tf.Variable(tf.random_normal([cell_size,1]))# (10,1)
b2 = tf.Variable(tf.constant(0.1,shape=[1,]))# (1,1)


# In[10]:


#定义LSTM
def RNN(x):
    input=tf.reshape(x,[-1,input_size]) # x降维
    input_rnn=tf.matmul(input,w1)+b1
    input_rnn=tf.reshape(input_rnn,[-1,time_step,cell_size])  #将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.LSTMCell(cell_size) #定义LSTM， block数为cell_size
    output, final_state = tf.nn.dynamic_rnn(cell, input_rnn, dtype=tf.float32)
    output = tf.reshape(output,[-1,cell_size]) 
    result = tf.matmul(output,w2)+b2
    return result


# In[11]:


#定义TensorFlow计算
prediction = RNN(X) # 预测
loss = tf.reduce_mean(tf.square(tf.reshape(prediction,[-1])-tf.reshape(Y, [-1]))) # 误差 mean_square(prediction,Y)
train_op=tf.train.AdamOptimizer(lr).minimize(loss) # 训练


# In[12]:


#训练
# ---- 训练 ---- 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1000):     #训练10个周期
        #for batch in range( len(train)//batch_size  ): # 这个地方把原来的改掉了
        for batch in range( len(x_train)//batch_size+1): # 1个周期内批次数
            start = batch*batch_size 
            end = min( (batch+1)*batch_size , len(x_train)) 
            batch_xs, batch_ys =  x_train[start : end ], y_train[start : end ] # 获得本批次的x,y
            _,loss_= sess.run([train_op, loss],feed_dict={X:batch_xs, Y:batch_ys}) # 训练
        print("Number of iterations:",epoch," loss:",loss_)
    print("The train has finished")
    # 开始测试
    y_pred=[]
    for x in  x_test :
        y=sess.run(prediction,feed_dict={X:[x]})
        y_pred.extend(y.reshape((-1)))
    y_test=np.array(y_test)*std[7]+mean[7]
    y_pred=np.array(y_pred)*std[7]+mean[7]
    acc=np.average(np.abs(y_pred-y_test[:len(y_pred)])/y_test[:len(y_pred)])  # 精确度
    print("The accuracy of this predict:",acc)
    #以折线图表示结果
    plt.figure()
    plt.plot(list(range(len(y_pred))), y_pred, color='b',)
    plt.plot(list(range(len(y_test))), y_test,  color='r')
    plt.show()

