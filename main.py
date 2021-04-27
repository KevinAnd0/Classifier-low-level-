import tensorflow as tf
import tensorflow.keras as keras
from model import CNN_Model
from preprocess import ProcessImage

train_path = ""

img_height = 160
img_width = 160
batch_size = 32

train_processor = ProcessImage(train_path, img_height, img_width, batch_size)
train_ds = train_processor.process()

model = CNN_Model(6)

optimizer = keras.optimizers.Adam(learning_rate=0.002)
loss_f = keras.losses.CategoricalCrossentropy()


for epoch in range(1):

    for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
        y_batch_train = tf.keras.utils.to_categorical(y_batch_train, num_classes=6)   

        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_f(y_batch_train, logits)  

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))