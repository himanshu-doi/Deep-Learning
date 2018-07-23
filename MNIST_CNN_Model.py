
# coding: utf-8

# In[2]:


from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from sklearn.model_selection import train_test_split


# In[3]:


import os as os
os.chdir('/home/himanshu/Downloads')


# In[4]:


data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[5]:


data.head()


# In[6]:


test.shape


# In[7]:


train = data.drop(['label'], axis = 1)


# In[8]:


label = data.label


# In[9]:


target = pd.get_dummies(label, columns=['label'], drop_first=False)
target.head()


# In[10]:


train.shape


# In[123]:


#x_train,x_cv,y_train,y_cv = train_test_split(train,y,test_size = 0.2, random_state = 42)
x_train,x_cv,y_train,y_cv = train_test_split(train,target,test_size = 0.2, random_state = 42)


# In[124]:


print x_train.shape
print x_cv.shape


# In[125]:


## Reshaping the dataframe to image
x_arr = np.array(x_train)
x_cv_arr = np.array(x_cv)
X = x_arr.reshape(33600,28,28,1)
X_cv = x_cv_arr.reshape(8400,28,28,1)


# ### Defining CNN architecture and required functions

# In[126]:


def idxargmax(a):
    idx = np.argmax(a, axis=None)
    multi_idx = np.unravel_index(idx, a.shape)
    if np.isnan(a[multi_idx]):
        nan_count = np.sum(np.isnan(a))
        idx = np.argpartition(a, -nan_count-1, axis=None)[-nan_count-1]
        idx = np.argsort(a, axis=None)[-nan_count-1]
        multi_idx = np.unravel_index(idx, a.shape)
    return multi_idx


# In[127]:


## PADDING function
def zero_pad(data, pad):
    data_pad = np.pad(data,((0,0),(pad,pad),(pad,pad),(0,0)), 'constant')
    
    return data_pad


# In[128]:


## Maxpooling function
def max_pool(X,f,stride):
    (m, w, w, c) = X.shape
    pool = np.zeros((m,int((w-f)/stride+1),int((w-f)/stride+1), c))
    for e in range(0,m):
        for k in range(0,c):
            i=0
            i = int(i)
            while(i<w):
                j=0
                j = int(j)
                while(j<w):
                    pool[e,int(i/2),int(j/2),k] = np.max(X[e,i:i+f,j:j+f,k])
                    j+=stride
                i+=stride
    return pool


# In[129]:


def softmax_cost(out,y):
    eout = np.exp(out, dtype=np.float)  
    probs = eout/np.sum(eout, axis = 1)[:,None]
    
    p = np.sum(np.multiply(y,probs), axis = 1)
    prob_label = np.argmax(np.array(probs), axis = 1)    ## taking out the arguments of max values
    cost = -np.log(p)    ## (Only data loss. No regularised loss)
    
    return p,cost,probs,prob_label


# In[130]:


def conv_net(input_data, Y, W1, W2, b1, b2, theta3,bias3):
    
    #####################################################################################################
    ###################################### Forward Propagation ##########################################
    #####################################################################################################
    
    ## Input shape
    m, n_Hi, n_Wi, n_Ci = input_data.shape
    
    ## no. of filters in layer_1 and layer_2 
    l1 = len(W1)    ## All filters for conv layer 1 horizontally stacked in W1  
    l2 = len(W2)    ## All filters for conv layer 2 horizontally stacked in W2
    
    (f, f, _) = W1[0].shape   ## Shape of the filter used
    pad = 1
    ## stride = 1
    
    ## Convolution layer1 output dimensions
    n_H1 = n_Hi+(2*pad)-f + 1
    n_W1 = n_Wi+(2*pad)-f + 1
   
    ## Convolution layer2 output dimensions
    n_H2 = n_H1-f + 1
    n_W2 = n_W1-f + 1
    
    ## Initializing output image matrices after both convolutions
    conv1 = np.zeros((m,n_H1,n_W1,l1))
    conv2 = np.zeros((m,n_H2,n_W2,l2))
    
    ## Padding the input images
    input_pad = zero_pad(input_data,pad)
    
    ## First convolution layer
    for i in range(0,m):                                                  ##looping over the no. of examples
        for j in range(0,l1):                                             ##looping over the no. of filters
            for x in range(0,n_H1):                                       ##looping over the height of one image 
                for y in range(0,n_W1):                                   ##looping over the width of one image
                    conv1[i,x,y,j] = np.sum(input_pad[i,x:x+f,y:y+f]*W1[j])+b1[j]
        conv1[i,:,:,:][conv1[i,:,:,:] <= 0] = 0                           ##relu activation
        
    ## Second Convolution layer
    for i in range(0,m):                                                  ##looping over the no. of examples
        for j in range(0,l2):                                             ##looping over the no. of filters
            for x in range(0,n_H2):                                       ##looping over the height of one image 
                for y in range(0,n_W2):                                   ##looping over the width of one image
                    conv2[i,x,y,j] = np.sum(np.multiply(conv1[i,x:x+f,y:y+f,:],W2[j]))+b2[j]
        conv2[i,:,:,:][conv2[i,:,:,:] <= 0] = 0                           ##relu activation
        
    
    ## Pooling layer after max_pooling filter size of 2x2 and stride 2
    pooled_layer = max_pool(conv2, 2, 2)  
    
    
    ## Fully connected layer of neurons
    fc1 = pooled_layer.reshape(m,int((n_H2/2)*(n_W2/2)*l2))
    
    ## Output layer of mx10 activation units
#    theta3 = initialize_theta3(params = int((n_H2/2)*(n_W2/2)*l2))
    out = np.dot(fc1,theta3) + bias3
        
    ## Using softmax to get the cost    
    p, cost, probs, prob_label = softmax_cost(out, Y)   ## change it to y_train or batch
    
    acc = []
    for i in range(0,len(Y)):
        if prob_label[i]==np.argmax(np.array(Y)[i,:]):
            acc.append(1)
        else:
            acc.append(0)

    ########################################################################################################
    ############################# Backpropagation to compute gradients #####################################
    ########################################################################################################
    
    d_out = probs - Y

    dtheta3 = np.dot(d_out.T, fc1)
    dbias3 = np.mean(d_out, axis = 0).reshape(1,10)    

    dfc1 = np.dot(theta3,d_out.T)
    
    dpool = dfc1.T.reshape((m, int(n_H2/2), int(n_W2/2), l2))
    dconv2 = np.zeros((m, n_H2, n_W2, l2))
    
    for k in range(0,m):
        for c in range(0,l2):
            i=0
            while(i<n_H2):
                j=0
                while(j<n_W2):
                    (a,b) = idxargmax(conv2[k,i:i+2,j:j+2,c]) ## Getting indexes of maximum value in the array
                    dconv2[k,i+a,j+b,c] = dpool[k,int(i/2),int(j/2),c]
                    j+=2
                i+=2

        dconv2[conv2<=0]=0


    dconv1 = np.zeros((m, n_H1, n_W1, l1))
    dW2_stack = {}
    db2_stack = {}
    
    dW2_stack = np.zeros((m,l2,f,f,l1))
    db2_stack = np.zeros((m,l2,1))

    dW1_stack = {}
    db1_stack = {}
    
    dW1_stack = np.zeros((m,l1,f,f,1))
    db1_stack = np.zeros((m,l1,1))

    dW2 = {}
    dW2 = np.zeros((l2,f,f,l1))
    db2 = np.zeros((l2,1))
    bW1 = {}
    dW1 = np.zeros((l1,f,f,1))
    db1 = np.zeros((l1,1))
    
    
    
    for i in range(0,m):                     ## looping through the one batch of 32 examples
        for c in range(0,l2):
            for x in range(0,n_H2):
                for y in range(0,n_W2):
                    dW2_stack[i,:,:,c] += dconv2[i,x,y,c]*conv1[i,x:x+f,y:y+f,:]
                    dconv1[i,x:x+f,y:y+f,:] += dconv2[i,x,y,c]*W2[c]
            db2_stack[i,c] = np.sum(dconv2[i,:,:,c])
        dconv1[conv1<=0]=0
        
        dW2 = np.mean(dW2_stack, axis = 0)        ## calculating the mean gradient for single batch 
        db2 = np.mean(db2_stack, axis = 0)
    
    for i in range(0,m):                          ## ## looping through the one batch of 32 examples
        for c in range(0,l1):
            for x in range(0,n_H1):
                for y in range(0,n_W1):
                    dW1_stack[i,:,:,c] += dconv1[i,x,y,c]*input_pad[i,x:x+f,y:y+f,:]

            db1_stack[i,c] = np.sum(dconv1[i,:,:,c])
            
        dW1 = np.mean(dW1_stack, axis = 0)
        db1 = np.mean(db1_stack, axis = 0)

        
        
    return dW1, dW2, db1, db2, dtheta3, dbias3, cost, probs, prob_label, acc 
        
        
        
        
        


# In[ ]:


## testing conv_net forward and backward prop
grads = conv_net(t,W1,W2,b1,b2,bias3)
[dW1, dW2, db1, db2, dtheta3, dbias3, cost, probs, prob_label, acc] = grads


# ### Gradient Descent optimizer

# In[131]:


def optimizer(batch,learning_rate,W1,W2,b1,b2,theta3,bias3):
    
    ## Slicing train data and labels from batch
    X = batch[:,0:-10]
    X = X.reshape(len(batch), w, w, l)
    Y = batch[:,784:794]
    
    
    batch_size = len(batch)
    
    ## Initializing gradient matrices 
    dW2 = {}
    dW2 = np.zeros((l2,f,f,l1))
    db2 = np.zeros((l2,1))
    bW1 = {}
    dW1 = np.zeros((l1,f,f,1))
    db1 = np.zeros((l1,1))
    
    dtheta3 = np.zeros(theta3.shape)
    dbias3 = np.zeros(bias3.shape)
    
    grads = conv_net(X,Y,W1,W2,b1,b2,theta3,bias3)
    [dW1, dW2, db1, db2, dtheta3, dbias3, cost_, probs_, prob_label, acc_] = grads
    
    W1 = W1-learning_rate*(dW1)
    b1 = b1-learning_rate*(db1)
    W2 = W2-learning_rate*(dW2)
    b2 = b2-learning_rate*(db2)
    theta3 = theta3-learning_rate*(dtheta3.T)
    bias3 = bias3-learning_rate*(dbias3)
    
    batch_cost = np.mean(cost_)
    #print dW1    ## checking if the gradients are calculated or not (yes)
    batch_accuracy = sum(acc_)/len(acc_)
    
    return W1, W2, b1, b2, theta3, bias3, batch_cost, acc_, batch_accuracy
    


# In[132]:


## Initializing weights and bias for convolution layers

W1 = 0.1*np.random.rand(3,3,3,1)
W2 = 0.1*np.random.rand(3,3,3,3)

b1 = 0.1*np.random.rand(3,1)
b2 = 0.1*np.random.rand(3,1)

## Initializing weights and bias for fully connected layer
#def initialize_theta3():
theta3 = 0.1*np.random.rand(507,10)
#    return theta3

bias3 = 0.1*np.random.rand(1,10)

## Normalizing input data
x_arr -= int(np.mean(x_arr))
x_arr = x_arr.astype(float)
x_arr /= int(np.std(x_arr))

train_data = np.hstack((x_arr,np.array(y_train)))     ## horizontally stacking the features and labels 
t = train_data[0:320]      ## training the model on only 300 examples images due to heavy computation issue

## Normalizing cross-validation data
x_cv_arr -= int(np.mean(x_cv_arr))
x_cv_arr = x_cv_arr.astype(float)
x_cv_arr /= int(np.std(x_cv_arr))

cv_data = np.hstack((x_cv_arr,np.array(y_cv)))
test_data = x_cv_arr[0:100]    ## cross-validating the model on only 100 examples images due to heavy computation issue   
Y_cv = np.array(y_cv)[0:100]

np.random.shuffle(train_data)

## Assigning hyperparameter values
learning_rate = 0.001
batch_size = 32
num_epochs = 10
num_images = len(t)   ##Number of the input training examples
w = 28
l = 1
l1 = len(W1)    ## no. opf filters in W1 and W2 
l2 = len(W2)    
f = len(W1[0])    


# ### Main_initiation_function

# In[133]:


def main_init(train_data,W1,W2,b1,b2,theta3,bias3):
    cost = []
    accuracy = []
    for epoch in range(0, num_epochs):
        batches = [train_data[k:k + batch_size] for k in xrange(0, len(train_data), batch_size)]
        x=0
        i = 1
        for batch in batches:
            
            output = optimizer(batch,learning_rate,W1,W2,b1,b2,theta3,bias3)
            [W1, W2, b1, b2, theta3, bias3, batch_cost,acc_,batch_acc] = output
            
            #epoch_acc = round(np.sum(accuracy[epoch*num_images/batch_size:])/(x+1),2)
            
            cost.append(batch_cost)
            accuracy.append(batch_acc)

            print 'ep:%d, batch_num = %f, batch_cost = %f, batch_acc = %f' %(epoch,i,batch_cost,batch_acc) 
            i+=1
    plt.subplot(121),plt.plot(range(int(num_epochs*num_images/batch_size)), cost),plt.title('Batch_cost')        
#     plt.plot(range(int(epoch*num_images/batch_size)), cost)
    plt.ylabel('Batch Training loss')
    plt.xlabel('Iteration')
    plt.subplot(122),plt.plot(range(int(num_epochs*num_images/batch_size)), accuracy),plt.title('Batch_accuracy')       
    plt.ylabel('Batch Training Accuracy')
    plt.xlabel('Iteration')
    plt.show()
    
    return W1,W2,b1,b2,theta3,bias3,cost,accuracy


# In[134]:


W1_t,W2_t,b1_t,b2_t,theta3_t,bias3_t,cost_t,accuracy_t = main_init(t,W1,W2,b1,b2,theta3,bias3)


# ### Predicting on cross-validation set

# In[141]:


def predict(test_data, Y_cv, W1, W2, b1, b2, theta3, bias3, cost_t):
    
    #####################################################################################################
    ###################################### Forward Propagation ##########################################
    #####################################################################################################
    
    test_data = test_data.reshape(len(test_data),w,w,l)
    
    ## test shape
    m, n_Hi, n_Wi, n_Ci = test_data.shape
    
    ## no. of filters in layer_1 and layer_2 
    l1 = len(W1)    ## All filters for conv layer 1 horizontally stacked in W1  
    l2 = len(W2)    ## All filters for conv layer 2 horizontally stacked in W2
    
    (f, f, _) = W1[0].shape   ## Shape of the filter used
    pad = 1
    ## stride = 1
    
    ## Convolution layer1 output dimensions
    n_H1 = n_Hi+(2*pad)-f + 1
    n_W1 = n_Wi+(2*pad)-f + 1
   
    ## Convolution layer2 output dimensions
    n_H2 = n_H1-f + 1
    n_W2 = n_W1-f + 1
    
    ## Initializing output image matrices after both convolutions
    conv1 = np.zeros((m,n_H1,n_W1,l1))
    conv2 = np.zeros((m,n_H2,n_W2,l2))
    
    ## Padding the test images
    test_pad = zero_pad(test_data,pad)
    
    ## First convolution layer
    for i in range(0,m):                                                  ##looping over the no. of examples
        for j in range(0,l1):                                             ##looping over the no. of filters
            for x in range(0,n_H1):                                       ##looping over the height of one image 
                for y in range(0,n_W1):                                   ##looping over the width of one image
                    conv1[i,x,y,j] = np.sum(test_pad[i,x:x+f,y:y+f]*W1[j])+b1[j]
        conv1[i,:,:,:][conv1[i,:,:,:] <= 0] = 0                           ##relu activation
        
    ## Second Convolution layer
    for i in range(0,m):                                                  ##looping over the no. of examples
        for j in range(0,l2):                                             ##looping over the no. of filters
            for x in range(0,n_H2):                                       ##looping over the height of one image 
                for y in range(0,n_W2):                                   ##looping over the width of one image
                    conv2[i,x,y,j] = np.sum(np.multiply(conv1[i,x:x+f,y:y+f,:],W2[j]))+b2[j]
        conv2[i,:,:,:][conv2[i,:,:,:] <= 0] = 0                           ##relu activation
        
    
    ## Pooling layer after max_pooling filter size of 2x2 and stride 2
    pooled_layer = max_pool(conv2, 2, 2)  
    
    
    ## Fully connected layer of neurons
    fc1 = pooled_layer.reshape(m,int((n_H2/2)*(n_W2/2)*l2))
    
    ## Output layer of mx10 activation units
#    theta3 = initialize_theta3(params = int((n_H2/2)*(n_W2/2)*l2))
    out_t = np.dot(fc1,theta3) + bias3
        
    ## Using softmax to get the cost    
    p_pred, cost_pred, probs_pred, prob_label_pred = softmax_cost(out_t, Y_cv)  
    
    cv_acc = []
    for i in range(0,len(Y_cv)):
        if prob_label_pred[i]==np.argmax(np.array(Y_cv)[i,:]):
            cv_acc.append(1)
        else:
            cv_acc.append(0)
            
    cv_accuracy = sum(cv_acc)/len(cv_acc)
    print 'cv_acc = %f' %(cv_accuracy)
            
    plt.subplot(121),plt.plot(range(m), cost_pred),plt.title('cv_cost')       
    plt.ylabel('CV cost')
    plt.xlabel('Iteration')
    plt.subplot(122),plt.plot(range(int(num_epochs*num_images/batch_size)), cost_t),plt.title('Training_cost')        
    plt.ylabel('Batch Training cost')
    plt.xlabel('Iteration')
    
    plt.show()
    
    
    return cost_pred, probs_pred, prob_label_pred


# In[142]:


predict(test_data, Y_cv, W1_t, W2_t, b1_t, b2_t, theta3_t, bias3_t, cost_t)

