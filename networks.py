import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense


def MNIST_Model():

    model = tf.keras.models.Sequential([
        Conv2D(32, (5, 5), padding="same", activation="relu", input_shape = (28,28,1)),
        MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="same"),
        Conv2D(16, (5, 5), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="same"),
        tf.keras.layers.Flatten(),
        Dense(8, activation= "relu"),
        Dense(10, activation= "softmax")
    ])
    
    return model

class Hypernetwork(keras.Model):

    def __init__(self, name=None, **kwargs):
        super().__init__(name = name, **kwargs)

        self.dense_1 = Dense(300, activation="relu")
        self.dense_2 = Dense(855, activation="relu")

        self.w1_dense_1 = Dense(40, activation="relu")
        self.w1_dense_2 = Dense(26, activation="relu") 
        
        self.w2_dense_1 = Dense(100, activation="relu")
        self.w2_dense_2 = Dense(801, activation="relu")

        self.w3_dense_1 = Dense(100, activation="relu")
        self.w3_dense_2 = Dense(785, activation="relu")

        self.w4_dense_1 = Dense(60, activation="relu")
        self.w4_dense_2 = Dense(90, activation="relu")


    def call(self, inputs):

        index_1 = 32*15
        index_2 = index_1 + 16*15
        index_3 = index_2 + 8*15
        index_4 = index_3 + 1*15

        layer_1 = 32
        layer_2 = 16
        layer_3 = 8
        layer_4 = 1

        output = []

        x = self.dense_1(inputs)
        x = self.dense_2(x)

        input_w1 = x[:,:index_1]
        input_w1 = tf.reshape(input_w1,(32,-1))


        for step in range(layer_1):

            w1 = input_w1[step,:]
            w1 = tf.reshape(w1,(-1,1))

            w1 = self.w1_dense_1(w1)
            w1 = self.w1_dense_2(w1)

            output = tf.concat([output,w1[1]], 0)
            
        input_w2 = x[:,index_1:index_2]
        input_w2 = tf.reshape(input_w1,(16,-1))
        
        for step in range(layer_2):
           
            w2 = input_w2[step,:]
            w2 = tf.reshape(w2,(-1,1))
            w2 = self.w2_dense_1(w2)
            w2 = self.w2_dense_2(w2)
            output = tf.concat([output,w2[1]], 0)

        input_w3 = x[:,index_2:index_3]
        input_w3 = tf.reshape(input_w3,(8,-1))

        for step in range(layer_3):

            w3 = input_w3[step,:]
            w3 = tf.reshape(w3,(-1,1))
            w3 = self.w3_dense_1(w3)
            w3 = self.w3_dense_2(w3)
            output = tf.concat([output,w3[1]], 0)

        input_w4 = x[:,index_3:index_4]
        input_w4 = tf.reshape(input_w4,(1,-1))

        for step in range(layer_4):
            
            w4 = input_w4[step,:]
            w4 = tf.reshape(w4,(-1,1))
            w4 = self.w4_dense_1(w4)
            w4 = self.w4_dense_2(w4)
            output = tf.concat([output,w4[1]], 0)

        return output
