import numpy as np
import tensorflow as tf
import keras

import networks


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

target_network = networks.MNIST_Model()

hypernetwork = keras.models.load_model('my_model')
#hypernetwork = networks.Hypernetwork()
#print(hypernetwork.summary())


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


z1 = np.random.uniform(low = -1, high = 1, size = 300)


z2 = np.random.uniform(low = -1, high = 1, size = 300)

batch_size = 32
idx = np.random.randint(low=0, high=x_train.shape[0], size=batch_size)
x, y = x_train[idx], y_train[idx]

# Define accuracy metrics for training and validation
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

#create a loss function and assign a optimizer
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(1e-3) 

resolution = np.arange(0,1,0.1)
print(resolution)

accuracy = []

for i in resolution:

    z = z1*i + z2*(1-i)
    z = z[np.newaxis, :]
    

    target_weights = hypernetwork(z)
      
    #set the weights of the target network
    set_parameters(target_network, target_weights)

    preds = target_network(x)

      
    loss_acc = loss_fn( y, preds)

    acc = train_acc_metric( y, tf.expand_dims(preds, 0)) 

    accuracy.append([i, loss_acc.numpy(), acc.numpy()])

print(accuracy)


'''
from tensorflow.examples.tutorials.mnist import input_data
from scipy import stats
import pickle
import sys

sys.path.append('../')
sys.path.append('../MNF/')
from train import GetImages

from params import GeneralParameters
general_params = GeneralParameters()

noise_batch_size = 40
number_of_paths = 10

from hypernetwork import HyperNetwork

# exactly one of these two sets of lines need to be commented - depending in whether we want to use HyperNetwork or MnfNetwork

# hnet = HyperNetwork()
# file_name = 'checkpoint-13000'
# noise_size =hnet.generator_hparams.input_noise_size

from mnf_network import MnfNetwork
hnet = MnfNetwork()
file_name = '../MNF/results/models/mnf_lenet_mnist_fq2_fr2_usezTrue_thres0.5/model/model'
noise_size = hnet.noise_size



net = HyperNetwork(use_generator=False)

x,y = GetImages('test')
labels = np.nonzero(y[0, :, :])[1] # convert one-hot to regular representation

with tf.Session() as sess:
    hnet.Restore(sess,file_name)

    acc1 = []
    acc2 = []
    zs = []


    for i in range(number_of_paths):

        z = hnet.SampleInput(2)
        w1,b1, w2,b2, w3,b3, w4,b4= hnet.GenerateWeights(sess,z)

        zs.append(z)

        interp = lambda q: (q[[-1]]-q[[0]])*np.reshape(np.linspace(0,1,noise_batch_size),[-1]+[1]*(q.ndim-1))+q[[0]]
        
        z = interp(z)
        w1 = interp(w1)
        b1 = interp(b1)
        w2 = interp(w2)
        b2 = interp(b2)
        w3 = interp(w3)
        b3 = interp(b3)
        w4 = interp(w4)
        b4 = interp(b4)

        acc1.append(hnet.GetAccuracy(sess,x,y,z))
        acc2.append(net.GetAccuracyWithForcedWeights(sess,x,y,w1,b1, w2,b2, w3,b3, w4,b4))


data = {'accuracy_direct':acc2,'accuracy_interp':acc1,'z':zs}
with open("path.pickle","wb") as f:
    pickle.dump(data, f)




import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rc('font', size=8)
plt.rc('axes', titlesize=8)
plt.rc('axes', labelsize=8)
plt.rc('axes', linewidth=0.6)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('legend', fontsize=8)
plt.rc('figure', titlesize=8)

# open file created with path_data_creator.py
with open('path_mnf.pickle','rb') as f:
    data = pickle.load(f)

direct = data['accuracy_direct']
non_direct = data['accuracy_interp']
zs = data['z']

t = np.linspace(0, 1, len(direct[0]))

cmap = plt.get_cmap('Dark2', 2)

for i in range(len(direct)):
    fig = plt.figure(figsize=(1.2, 1), facecolor='white')
    ax = plt.axes()

    ax.set_color_cycle([cmap(i) for i in range(2)])
    plt.plot(t,direct[i]*100,'--',lw=1.0,label='direct path')
    plt.plot(t,non_direct[i]*100,lw=1.0,label='interpolated path')
    plt.xlabel('t')
    plt.ylabel('accuracy [%]')
    ax.get_xaxis().set_ticks(np.arange(0,1.01,0.5))
    ax.get_yaxis().set_ticks(np.arange(0,104,5))
    plt.ylim(np.min(direct[i]*100)-5,108)
    # ax.legend(loc='lower center',labelspacing=0.1,borderpad=0.2)
    plt.gca().set_ylim(top=105)
    plt.tight_layout(0,0,0)


'''