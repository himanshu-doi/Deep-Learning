
# coding: utf-8

# In[2]:


from __future__ import division

import cv2
import glob
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.utils import shuffle
import tensorflow as tf
get_ipython().magic(u'matplotlib inline')
import os as os

# random.seed(42)
# np.random.seed(42)
# tf.reset_default_graph()


# In[3]:


os.chdir('/home/himanshu/dogs')
labels = pd.read_csv('Y_train_final.csv')
Y_cv = pd.read_csv('eval_data.csv')


# In[4]:


labels = labels.drop(['Unnamed: 0'], axis = 1)


# In[5]:


labels.head()


# In[7]:


## One hot encoding the labels

def one_hot(y): 
    Y = pd.get_dummies(y, columns=['breed'], drop_first=False)
    Y.columns = Y.columns[1:121].str[6:].insert(0,'id')    ## str[6:] because get dummies adds the original column name before each new column created 
    Y = pd.concat([Y, y.breed], axis = 1)
    return Y


# In[8]:


## One-hot encoding the training labels
data = one_hot(labels)


# In[9]:


## Loading batches from training data

def train_data(data, batch_size):
    batch = []
    target = []
    temp = data[['id','breed']].sample(n = batch_size, replace = False)
    for i,b in temp.values:  
        os.chdir('/home/himanshu/dogs/train/%s'%(b))
#         os.chdir('/home/himanshu/dogs/eval/')
        img = []
        img = cv2.imread('%s'%(i)+'.jpg') 
        img = cv2.resize(img, (w,w))/255
        batch.append(img)
    
    batch = np.asarray(batch)
    target = data.iloc[temp.index,1:-1]
    batch_x = batch
    batch_y = np.asarray(target)
    return batch_x, batch_y


# In[10]:


## Training Parameters 
learning_rate = 0.01
num_epochs = 4
w = 128      ## height and width of image
l = 3       ## num of channels in input image
batch_size = 128
display_step = 10
epsilon = 10e-8

## Network Parameters
num_classes = 120     # Num of classes in dog breeds
dropout = 0.80        # Dropout, probability to keep units

## tf Graph input
X = tf.placeholder(tf.float32, shape = [batch_size,w,w,l])
Y = tf.placeholder(tf.float32, [None, num_classes])
phase = tf.placeholder(tf.bool, name='phase')
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# In[54]:


## Defining wrappers for convolution functions
def conv2D(x,W,b,stride = 2):
    x = tf.nn.conv2d(x, W, strides = [1,stride,stride,1], padding='VALID')
    x = tf.nn.bias_add(x,b)
    x = tf.cast(x,dtype='float32')
    #with tf.variable_scope('bn', reuse=tf.AUTO_REUSE) as scope:
#     x_BN = tf.contrib.layers.batch_norm(x,center=True,scale=True,is_training=phase)
    return tf.nn.relu(x)

def max_pool(x,k=2,s=2):
    return tf.nn.max_pool(x, ksize=[1,k,k,1],strides=[1,s,s,1], padding='SAME')
    
    
## Creating convolution model
def conv_net(x, weights, biases, dropout):
    
    ## tensor input of 4D [Batch_size, height, width, channel]
    
    ## First convolution layer
    conv1 = conv2D(x, weights['W1'], biases['b1'], stride = 2)
#     conv1 = max_pool(conv1, k=2, s=2)
    
    ## Second convolution layer
    conv2 = conv2D(conv1, weights['W2'], biases['b2'], stride = 2)
#     conv2 = max_pool(conv2, k=2, s=2)
    
    ## Third convolution layer
    conv3 = conv2D(conv2, weights['W3'], biases['b3'], stride = 2)
#     conv3 = max_pool(conv3, k=2, s=2) 
    
    ## Fully connected layer
    ## Reshaping the output from 3D image shape to 1D vectors for fully connected layers
    fc1 = tf.reshape(conv3, [-1, weights['fw1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['fw1']), biases['fb1'])
    fc1 = tf.nn.relu(fc1)

    ## Applying Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    
    ## Output class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# In[14]:


tf.set_random_seed(42)
## Initializing layer weights & biases
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'W1': tf.Variable(tf.random_normal([5, 5, 3, 16]), name='W1'),
    # 5x5 conv, 32 inputs, 64 outputs
    'W2': tf.Variable(tf.random_normal([5, 5, 16, 32]), name='W2'),
#     3x3 conv, 64 inputs, 64 outputs
    'W3': tf.Variable(tf.random_normal([3, 3, 32, 64]),name='W3'),
    # fully connected, 6*6*64 inputs, 1024 outputs
    'fw1': tf.Variable(tf.random_normal([14*14*64, 1024]),name='fw1'),
    # 1024 inputs, 120 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]),name='w_out')
}

biases = {
    'b1': tf.Variable(tf.random_normal([16]),name='b1'),
    'b2': tf.Variable(tf.random_normal([32]),name='b2'),
    'b3': tf.Variable(tf.random_normal([64]),name='b3'),
    'fb1': tf.Variable(tf.random_normal([1024]),name='fb1'),
    'out': tf.Variable(tf.random_normal([num_classes]),name='b_out')
}

## Obtaining predictions using softmax layer
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

## Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


## Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
## Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

tf.summary.scalar('accuracy',accuracy)
tf.summary.scalar('loss',loss_op)
merged = tf.summary.merge_all()
    
writer = tf.summary.FileWriter('/home/himanshu/dogs/chkpt', tf.get_default_graph())


# In[15]:


## Viewing trainable variables
tf.trainable_variables()


# In[16]:


## Start training
with tf.Session() as sess:

 
    ## Run the initializer
    sess.run(init)
    
    for epoch in range(num_epochs):
        for step in range(1, int((len(data)/batch_size)+1)):
            batch_x, batch_y = train_data(data, batch_size)
            ## Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout,'phase:0': 1})
            if step % display_step == 0 or step == 1:
                ## Calculate batch loss and accuracy
                loss, acc,summary = sess.run([loss_op, accuracy,merged], feed_dict={X: batch_x,
                                                                     Y: batch_y,
                                                                     keep_prob: 1.0,'phase:0': 1})
                writer.add_summary(summary,step)
                print("Epoch " + str(epoch)+", Step " + str(step) + ", Minibatch Loss= " +                       "{:.4f}".format(loss) + ", Training Accuracy= " +                       "{:.3f}".format(acc))

                ##saving checkpoints after every 10 iterations
                save_path = tf.train.Saver().save(sess, '/home/himanshu/dogs/chkpt/chkpts.ckpt', global_step=step)

    
    ## Calculate accuracy for 256 MNIST test images
#     print("Testing Accuracy:", \
#         sess.run(accuracy, feed_dict={X: ,
#                                       Y: mnist.test.labels[:256],
#                                       keep_prob: 1.0,'phase:0': 0}))
writer.close()


# In[17]:


## Loading parameters from best converged check point
os.chdir('/home/himanshu/dogs/chkpt')
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('chkpts.ckpt-30.meta')
    new_saver.restore(sess, '/home/himanshu/dogs/chkpt/chkpts.ckpt-30')
    w_new1 = sess.run('W1:0')
    w_new2 = sess.run('W2:0')
    w_new3 = sess.run('W3:0')
    b_new1 = sess.run('b1:0')
    b_new2 = sess.run('b2:0')
    b_new3 = sess.run('b3:0')
    fw_new1 = sess.run('fw1:0')
    fb_new1 = sess.run('fb1:0')
    w_out1 = sess.run('w_out:0')
    b_out1 = sess.run('b_out:0')


# ### Model testing

# In[18]:


Y_cv = Y_cv.drop('Unnamed: 0', axis = 1)
Y_cv.head()


# In[19]:


Y_cv = one_hot(Y_cv)


# In[20]:


Y_cv.head()


# In[21]:


## Loading batches from training data

def test_data(data, batch_size):
    batch = []
    target = []
    temp = data[['id','breed']].sample(n = batch_size, replace = False)
    for i,b in temp.values:  
        os.chdir('/home/himanshu/dogs/eval/')
        img = []
        img = cv2.imread('%s'%(i)+'.jpg') 
        img = cv2.resize(img, (w,w))/255
        batch.append(img)
    
    batch = np.asarray(batch)
    target = data.iloc[temp.index,1:-1]
    batch_x = batch
    batch_y = np.asarray(target)
    return batch_x, batch_y


# In[64]:


def predict(x, weights, biases, dropout):
    
    ## tensor input of 4D [Batch_size, height, width, channel]
    
    ## First convolution layer
    conv1 = conv2D(x, weights['W1'], biases['b1'], stride = 2)
#     conv1 = max_pool(conv1, k=2, s=2)
    
    ## Second convolution layer
    conv2 = conv2D(conv1, weights['W2'], biases['b2'], stride = 2)
#     conv2 = max_pool(conv2, k=2, s=2)
    
    ## Third convolution layer
    conv3 = conv2D(conv2, weights['W3'], biases['b3'], stride = 2)
#     conv3 = max_pool(conv3, k=2, s=2) 
    
    ## Fully connected layer
    ## Reshaping the output from 3D image shape to 1D vectors for fully connected layers
    fc1 = tf.reshape(conv3, [-1, weights['fw1'].shape[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['fw1']), biases['fb1'])
    fc1 = tf.nn.relu(fc1)

    ## Applying Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    
    ## Output class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    
    test_logit = tf.nn.softmax(out)
    return test_logit


# In[67]:


## Training weights with lowest minibatch loss

weights_new = {
    # 5x5 conv, 1 input, 32 outputs
    'W1': w_new1,
    # 5x5 conv, 32 inputs, 64 outputs
    'W2': w_new2,
#     3x3 conv, 64 inputs, 64 outputs
    'W3': w_new3,
    # fully connected, 6*6*64 inputs, 1024 outputs
    'fw1': fw_new1,
    # 1024 inputs, 120 outputs (class prediction)
    'out': w_out1
}

biases_new = {
    'b1': b_new1,
    'b2': b_new2,
    'b3': b_new3,
    'fb1': fb_new1,
    'out': b_out1
}
place = tf.placeholder(tf.float32, shape = [batch_size,w,w,l])
keep_prob1 = tf.placeholder(tf.float32)


# In[102]:


from sklearn.metrics import confusion_matrix
test_pred = []
y_true = []
test_init = tf.global_variables_initializer()
p = predict(place,weights_new,biases_new,keep_prob1) 
with tf.Session() as sess:
    sess.run(test_init)
    
    for step in range(1, int((len(Y_cv)/batch_size)+1)):
        test_x, test_y = test_data(Y_cv, batch_size)
        pred = sess.run(p, feed_dict={place:test_x, keep_prob1:1})
        test_pred.append(pred)
        y_true.append(test_y)

    test_pred = np.asarray(test_pred)   
    y_true = np.asarray(y_true)   
    conf_mat = confusion_matrix(np.argmax(y_true, axis = -1).reshape(-1,1), np.argmax(test_pred, axis = -1).reshape(-1,1))
    print conf_mat    


# In[81]:


## Writing the confusion matrix to local disk
os.chdir('/home/himanshu/dogs')
pd.DataFrame(conf_mat).to_csv('conf_mat.csv')

