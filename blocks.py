import tensorflow as tf
import tensorflow.keras as keras

class ConvBlock(keras.layers.Layer):
    def __init__(self, filters, kernel, strides):
        super(ConvBlock, self).__init__()
        self.conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel, strides=1, padding='same')
        self.bn = keras.layers.BatchNormalization()
        self.leaky = keras.layers.LeakyReLU()
        
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        output = self.leaky(x)
        return output

class DenseBlock(keras.layers.Layer):
    def __init__(self, filters, activation):
        super(DenseBlock, self).__init__()
        self.dense = keras.layers.Dense(filters, activation)
        self.bn = keras.layers.BatchNormalization()
        self.relu = keras.layers.LeakyReLU()
        self.dropout = keras.layers.Dropout(0.5)
        
    def call(self, inputs):
        x = self.dense(inputs)
        x = self.bn(x)
        x = self.relu(x)
        output = self.dropout(x)
        return output