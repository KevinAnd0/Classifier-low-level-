import tensorflow as tf
import tensorflow.keras as keras
from blocks import ConvBlock, DenseBlock

class CNN_Model(keras.Model):
    def __init__(self, n_classes):
        super(CNN_Model, self).__init__()
        self.rescaling = keras.layers.experimental.preprocessing.Rescaling(1./255)
        self.pooling = keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.flat = keras.layers.Flatten()
        self.conv_block1 = ConvBlock(32, 3, 1)
        self.conv_block2 = ConvBlock(64, 3, 1)
        self.conv_block3 = ConvBlock(128, 3, 1)
        self.conv_block4 = ConvBlock(256, 3, 1)
        self.conv_block5 = ConvBlock(512, 3, 1)
        self.dense_block1 = DenseBlock(1024, activation='relu')
        self.global_layer = keras.layers.AveragePooling2D()
        self.output_layer = keras.layers.Dense(n_classes, activation='softmax')
        
    def call(self, inputs, training=False):
        x = self.rescaling(inputs)
        x = self.conv_block1(x)
        x = self.pooling(x)
        x = self.conv_block2(x)
        x = self.pooling(x)
        x = self.conv_block3(x)
        x = self.pooling(x)
        x = self.conv_block4(x)
        x = self.pooling(x)
        x = self.conv_block5(x)
        x = self.global_layer(x)
        x = self.flat(x)
        x = self.dense_block1(x)
        output = self.output_layer(x)
        return output
    
    def build_graph(self, shape):
        x = tf.keras.layers.Input(shape=shape)
        return keras.Model(inputs=[x], outputs=self.call(x))