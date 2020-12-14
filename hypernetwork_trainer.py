import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense

import networks

# I used tensorflow 2.3.0 to run this so just check if its the same
print('tensorflow version: {}'.format(tf.__version__))

# clear the backend before starting the training
tf.keras.backend.clear_session()

#load the MNIST dataset and devide it into test and train dataset
#TODO: create validation dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# convert to float32 and normalize. 
x_train = x_train.astype('float32') /255
x_test = x_test.astype('float32')   /255

# one-hot encode the labels 
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# add a channel dimension to the images
x_train = x_train.reshape(x_train.shape[0], 28, 28,1)
x_test = x_test.reshape(x_test.shape[0], 28, 28,1)


#TODO: Figure out why the gradients of the target network can only be calculated when its a functional 
target_network = networks.MNIST_Model()


#TODO: make a good and easily scalable version of this function

def set_parameters(mnist_model, weights_pred):

    #these are just the indices for the target network
    index_w1 = 800
    index_b1 = 832

    index_w2 = index_b1 + 12800
    index_b2 = index_b1 + 12816

    index_w3 = index_b2 + 6272
    index_b3 = index_b2 + 6280

    index_w4 = index_b3 + 80
    index_b4 = index_b3 + 90

    #These are the weights for the first layer: input is 15*32=480 and output is 26*32=832
    #So for the kernel we use 800 which has to be reshaped to (5,5,1,32) of the weights and for the bias 32
    kernel_1 = weights_pred[:index_w1]
    kernel_1 = tf.reshape(kernel_1,(5,5,1,32))
    b_1 = weights_pred[index_w1:index_b1]
    
    #These are the weights for the second layer: input here is 15*16=240 and the output is 16*801=12816
    #The kernel has to be reshaped to (5,5,1,32) and 16 for the bias
    kernel_2 = weights_pred[index_b1:index_w2]
    kernel_2 = tf.reshape(kernel_2,(5,5,32,16))
    b_2 = weights_pred[index_w2:index_b2]
    

    #These are the weights for the third layer: takes as input 15*8=120 and the output is 8*785=6280
    #The bias is 8 and the kernel shape is

    kernel_3 = weights_pred[index_b2:index_w3]
    kernel_3 = tf.reshape(kernel_3,(784,8))
    b_3 = weights_pred[index_w3:index_b3]
    
    #And these are the weights for the output layer

    kernel_4 = weights_pred[index_b3:index_w4]
    kernel_4 = tf.reshape(kernel_4,(8,10))
    b_4 = weights_pred[index_w4:index_b4]

    #assign the weights to the kernel and biases
    mnist_model.layers[0].kernel = kernel_1
    mnist_model.layers[0].bias = b_1

    mnist_model.layers[2].kernel = kernel_2
    mnist_model.layers[2].bias = b_2

    mnist_model.layers[5].kernel = kernel_3
    mnist_model.layers[5].bias = b_3

    mnist_model.layers[6].kernel = kernel_4
    mnist_model.layers[6].bias = b_4

    flattened_network = gauge_fixing(kernel_1,b_1,kernel_2,b_2,kernel_3,b_3,kernel_4,b_4)
     

    return flattened_network




initialization_std = 0.1
bias_initialization = 0.1

image_height = 28
image_width = 28

# layer1 : convolutional
layer1_filter_size = 5
layer1_size = 32
layer1_pool_size = 2

# layer2 : convolutional
layer2_filter_size = 5
layer2_size = 16
layer2_pool_size = 2

# layer3 : fully connected
layer3_size = 8

# layer4 : fully connected (but has no params)



# optimization params
learning_rate = 1e-3
learning_rate_rate = 0.9999
batch_size = 256

number_of_channels = 1

zero_fixer = 1e-8

lamBda = 10000

input_noise_size = 300

noise_batch_size = tf.identity(1,name='noise_batch_size') #TODO: check why this is actually 32 

def ExpandDims(tensor:tf.Tensor,axis,name=None):
    """
    perform multiple tf.expand_dims at once
    :param tensor:
    :param axis:
    :param name:
    :return:
    """
    for i in np.sort(axis):
        tensor = tf.expand_dims(tensor,i)
    tensor = tf.identity(tensor,name)
    return tensor


def gauge_fixing(w1_not_gauged,b1_not_gauged,w2_not_gauged,b2_not_gauged,w3_not_gauged,b3_not_gauged,w4_not_gauged,b4_not_gauged):

  #layer 1
  required_scale = layer1_filter_size * layer1_filter_size * number_of_channels + 1
  scale_factor = tf.math.sqrt(tf.math.reduce_sum(tf.square(w1_not_gauged), [0,1,2]) + tf.math.square(b1_not_gauged)) / required_scale + zero_fixer

  w1 = w1_not_gauged / (ExpandDims(scale_factor, [0,1,2]) + zero_fixer)
  b1 = b1_not_gauged / (scale_factor + zero_fixer)
  w1 = tf.identity(w1, 'w1')
  b1 = tf.identity(b1, 'b1')
  
  w2 = w2_not_gauged * ExpandDims(scale_factor, [0,1,3])

  #layer 2
  required_scale = layer2_filter_size * layer2_filter_size * layer1_size + 1
  scale_factor = tf.math.sqrt(tf.math.reduce_sum(tf.square(w2), [0,1,2]) + tf.math.square(b2_not_gauged)) / required_scale + zero_fixer

  w2 = w2 / (ExpandDims(scale_factor, [0,1,2]) + zero_fixer)
  b2 = b2_not_gauged / (scale_factor + zero_fixer)
  w2 = tf.identity(w1, 'w1')
  b2 = tf.identity(b1, 'b1')
  w3_not_gauged = tf.reshape(w3_not_gauged, (7,7,16,8))
  w3 = w3_not_gauged * ExpandDims(scale_factor, [0,1,3])

  #layer 3
  required_scale = (image_height / (layer1_pool_size * layer2_pool_size)) * (image_width / (layer1_pool_size * layer2_pool_size)) * layer2_size + 1
  scale_factor = tf.math.sqrt(tf.math.reduce_sum(tf.square(w3), [0,1,2]) + tf.math.square(b3_not_gauged)) / required_scale + zero_fixer

  w3 = w3 / (ExpandDims(scale_factor, [0,1,2]) + zero_fixer)
  b3 = b3_not_gauged / (scale_factor + zero_fixer)
  w3 = tf.identity(w1, 'w1')
  b3 = tf.identity(b1, 'b1')
  
  w4 = w4_not_gauged * ExpandDims(scale_factor, [1])

  #layer 4
  required_softmax_bias = 0.0
  softmax_bias_diff = tf.math.reduce_sum(b4_not_gauged, 0) - required_softmax_bias
  b4 = b4_not_gauged - softmax_bias_diff
  w4 = tf.identity(w4, 'w4')
  b4 = tf.identity(b4, 'b4')

  flattened_network = tf.concat(axis=1, values=[tf.reshape(w1, [noise_batch_size, -1]),tf.reshape(b1, [noise_batch_size, -1]),tf.reshape(w2, [noise_batch_size, -1]),tf.reshape(b2, [noise_batch_size, -1]),tf.reshape(w3, [noise_batch_size, -1]),tf.reshape(b3, [noise_batch_size, -1]),tf.reshape(w4, [noise_batch_size, -1]),tf.reshape(b4, [noise_batch_size, -1])],name='flattened_network')

  return flattened_network


#Import the hypernetwork
hypernetwork = networks.Hypernetwork()

# Define accuracy metrics for training and validation
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

#create a loss function and assign a optimizer
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(1e-3) 

#null the accumulated loss and define a batchsize
#TODO: find out how to find optimal hyperparameters for the simple classifier-network

batch_size = 8

#the actual training loop
#TODO: define a different end-criteria instead of a high number
for step in range(1, 60000):

  #randomly chose a batch
  #TODO: epochs should be evaluated
  loss_accum = 0.0
  flattened_networks = []
  
  
  
  with tf.GradientTape() as tape:
    

    for i in range(batch_size):
      #create the random vector of dimension 300 as input of the hypernetwork and reshape it
      z = np.random.uniform(low = -1, high = 1, size = 300)
      z = z[np.newaxis,:]
      idx = np.random.randint(low=0, high=x_train.shape[0], size=batch_size)
      x, y = x_train[idx], y_train[idx]
      
      #predict the weights of the target network
      target_weights = hypernetwork(z)
      
      #set the weights of the target network
      flattened_network = set_parameters(target_network, target_weights)   
      flattened_networks.append(flattened_network)
      
      #make a prediction with the new weights
      preds = target_network(x)

      #calculate the loss according to the prior defined loss function
      #TODO: add the diversity term to the loss function
      loss_acc = loss_fn( y, preds)

      


      loss_accum += loss_acc
      #print(loss, i)
      #check the accuracy of the predictions
      train_acc_metric( y, tf.expand_dims(preds, 0)) 
    
    flattened_networks = tf.stack(flattened_networks) 

    # entropy estimated using  Kozachenko-Leonenko estimator, with l1 distances
    mutual_distances = tf.math.reduce_sum(tf.math.abs(tf.expand_dims(flattened_network, 0) - tf.expand_dims(flattened_network, 1)), 1,name='mutual_squared_distances') # all distances between weight vector samples
    #mutual_distances = tf.math.reduce_sum(tf.math.abs(flattened_network - tf.reshape(flattened_network, -1)), 1,name='mutual_squared_distances') # all distances between weight vector samples

    nearest_distances = tf.identity(-1*tf.math.top_k(-1 * mutual_distances, k=2)[0][:, 1] ,name='nearest_distances') # distance to nearest neighboor for each weight vector sample
    entropy_estimate = tf.identity(input_noise_size * tf.math.reduce_mean(tf.math.log(nearest_distances + zero_fixer)) + tf.math.digamma(tf.cast(noise_batch_size, tf.float32)), name='entropy_estimate')

    loss_div = tf.identity( - 1 * entropy_estimate, name='diversity_loss')

    lamBda = tf.Variable(lamBda, dtype=tf.float32, trainable=False, name='lambda')
    loss = tf.identity(lamBda*loss_accum + loss_div,name='loss')
    loss_accum += loss
    # Train only hyper model
    grads = tape.gradient(loss_accum, hypernetwork.trainable_weights)
    optimizer.apply_gradients(zip(grads, hypernetwork.trainable_weights))
    

    if step % 10 == 0:

      
      var = target_weights.numpy()
      print('statistics of the generated parameters: '+'Mean, {:2.3f}, var {:2.3f}, min {:2.3f}, max {:2.3f}'.format(var.mean(), var.var(), var.min(), var.max()))
      print('\n Step: {}, validation set accuracy: {:2.2f}     loss: {:2.2f}'.format(step, float(train_acc_metric.result()), loss_accum))
      loss_accum = 0.0
      

    if step % 5000 == 0:
      hypernetwork.save('hyper_model5000', overwrite=True)
      break
  
#TODO: create a function to evaluate and test the network
